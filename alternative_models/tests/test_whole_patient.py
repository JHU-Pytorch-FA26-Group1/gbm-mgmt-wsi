"""Real autograd equivalence and patient-level safeguards on CPU."""
import copy
import unittest

import torch
from torch import nn

from alternative_models.whole_patient import backward_patient, finetune_parameters, Plateau
from alternative_models.patient_metrics import summarize
from training.models import BagModel


class Encoder(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.stem = nn.Linear(3, 4)
        self.stages = nn.ModuleList([nn.Sequential(nn.Tanh(), nn.Dropout(dropout), nn.Linear(4, 4))])
        self.head = nn.LayerNorm(4)

    def forward(self, images):
        return self.head(self.stages[0](self.stem(images)))


def model(dropout=0):
    return BagModel(Encoder(dropout), 4, {"attention_hidden": 5, "attention_dropout": dropout})


class WholePatient(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.images = torch.randn(11, 3)
        self.target = torch.tensor([1.])
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.7))

    def compare(self, chunk_size, dropout=0):
        reference = model(dropout).train()
        actual = copy.deepcopy(reference)
        chunks = list(self.images.split(chunk_size))
        state = torch.get_rng_state()
        # Keep graphs for all chunks: the straightforward, memory-heavy reference.
        embeddings = torch.cat([reference.encode(chunk) for chunk in chunks])
        logit, _ = reference.pool(embeddings.unsqueeze(0))
        loss = self.loss_fn(logit, self.target)
        loss.backward()
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(state)
        result = backward_patient(actual, chunks, lambda i: chunks[i], self.target, self.loss_fn, "cpu")
        self.assertAlmostEqual(result["loss"], loss.item(), places=6)
        self.assertAlmostEqual(result["logit"], logit.item(), places=6)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for (name, first), (_, second) in zip(reference.named_parameters(), actual.named_parameters()):
            self.assertTrue(torch.allclose(first.grad, second.grad, rtol=2e-5, atol=2e-6), name)
        before, after = torch.optim.Adam(reference.parameters(), lr=.001), torch.optim.Adam(actual.parameters(), lr=.001)
        before.step()
        after.step()
        for (name, first), (_, second) in zip(reference.named_parameters(), actual.named_parameters()):
            self.assertTrue(torch.allclose(first, second, rtol=2e-5, atol=2e-6), name)

    def test_matches_global_loss_gradients_and_update_with_uneven_chunks(self):
        for size in (1, 4, 11, 32):
            with self.subTest(chunk=size):
                self.compare(size)

    def test_dropout_replays_identical_rng_and_gradients(self):
        self.compare(4, dropout=.25)

    def test_chunk_order_does_not_change_global_attention_without_dropout(self):
        first, second = model(), None
        second = copy.deepcopy(first)
        chunks = list(self.images.split(4))
        a = backward_patient(first, chunks, lambda i: chunks[i], self.target, self.loss_fn, "cpu")
        chunks.reverse()
        b = backward_patient(second, chunks, lambda i: chunks[i], self.target, self.loss_fn, "cpu")
        self.assertAlmostEqual(a["loss"], b["loss"], places=6)
        for p, q in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.allclose(p.grad, q.grad, rtol=2e-5, atol=2e-6))

    def test_full_patient_loss_is_not_average_of_independent_chunk_losses(self):
        net = model()
        chunks = list(self.images.split(4))
        logit, _ = net.pool(net.encode(self.images).unsqueeze(0))
        whole = self.loss_fn(logit, self.target)
        averaged = sum(len(c)/len(self.images) * self.loss_fn(net.pool(net.encode(c).unsqueeze(0))[0], self.target) for c in chunks)
        self.assertGreater(abs(whole.item()-averaged.item()), 1e-6)

    def test_activation_graphs_only_exist_for_one_encoder_chunk_at_a_time(self):
        net, events = model(), []
        net.backbone.register_forward_hook(lambda module, args, output: events.append((torch.is_grad_enabled(), len(args[0]))))
        chunks = list(self.images.split(4))
        backward_patient(net, chunks, lambda i: chunks[i], self.target, self.loss_fn, "cpu")
        self.assertEqual(events, [(False, 4), (False, 4), (False, 3), (True, 4), (True, 4), (True, 3)])

    def test_changed_pixels_and_stateful_batchnorm_are_refused(self):
        net, calls = model(), []
        def load(index):
            calls.append(index)
            return self.images if len(calls) == 1 else self.images + 1
        with self.assertRaisesRegex(ValueError, "replay changed"):
            backward_patient(net, [self.images], load, self.target, self.loss_fn, "cpu")
        net.backbone.stages[0].append(nn.BatchNorm1d(4))
        with self.assertRaisesRegex(ValueError, "BatchNorm"):
            backward_patient(net, [self.images], lambda _: self.images, self.target, self.loss_fn, "cpu")

    def test_only_last_stage_and_final_normalization_are_unfrozen(self):
        net = model()
        selected = finetune_parameters(net, "convnext_v2_t")
        self.assertTrue(selected)
        self.assertTrue(all(not p.requires_grad for p in net.backbone.stem.parameters()))
        self.assertTrue(all(p.requires_grad for p in net.backbone.stages[-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in net.backbone.head.parameters()))

    def test_plateau_does_not_reset_for_insignificant_improvement(self):
        monitor = Plateau(patience=2, min_delta=.001)
        self.assertEqual(monitor.observe(.5), (True, False))
        self.assertEqual(monitor.observe(.4999), (True, False))
        self.assertEqual(monitor.observe(.4998), (True, True))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            monitor.observe(float("nan"))

    def test_confusion_matrix_orientation_and_patient_predictions(self):
        rows = [dict(patient_id=str(i), label=y, score=p) for i,(y,p) in enumerate([(0,.1),(0,.8),(1,.4),(1,.9),(1,.7)])]
        result = summarize(rows)
        self.assertEqual(result["confusion_matrix"], [[1,1],[1,2]])
        self.assertEqual(result["accuracy"], .6)
        self.assertAlmostEqual(result["balanced_accuracy"], (1/2+2/3)/2)
        self.assertEqual([r["correct"] for r in rows], [True,False,False,True,True])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(rows+rows[:1])


if __name__ == "__main__":
    unittest.main()
