"""Score patients: encode each patient's whole pool in chunks, pool it once, one prediction each."""
import torch
from torch.utils.data import DataLoader

from training import data


def predict(model, tiles, patients, settings, pipeline, *, device="cpu",  # pylint: disable=too-many-arguments
            batch_size=64, num_workers=0):
    """[(patient_id, logit, probability)] in the order of `patients`; leaves the model in eval.

    With `num_workers` above 0, workers read and normalize the chunks; the scores do not change.
    """
    if len(set(patients)) != len(patients):
        raise ValueError("a patient is listed twice; its tiles would be pooled together")
    model = model.to(device).eval()
    chunks = DataLoader(data.EvalChunks(tiles, patients, settings, pipeline, batch_size),
                        batch_size=None, num_workers=num_workers,
                        multiprocessing_context=data.worker_context(num_workers))
    embeddings = {patient: [] for patient in patients}
    with torch.no_grad():
        for patient, chunk in chunks:
            embeddings[patient].append(model.encode(chunk.to(device)).cpu())
        rows = []
        for patient in patients:
            if not embeddings[patient]:
                raise RuntimeError(f"patient {patient} has no tiles to score")
            logit, _ = model.pool(torch.cat(embeddings[patient])[None].to(device))
            rows.append((patient, logit.item(), torch.sigmoid(logit).item()))
    return rows
