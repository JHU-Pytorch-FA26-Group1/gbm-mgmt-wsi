"""Score patients: encode each patient's whole pool in chunks, pool it once, one prediction each."""
import torch

from training import data


def predict(model, tiles, patients, settings, pipeline, *, device="cpu",  # pylint: disable=too-many-arguments
            batch_size=64):
    """[(patient_id, logit, probability)] in the order of `patients`; leaves the model in eval."""
    model = model.to(device).eval()
    rows = []
    with torch.no_grad():
        for patient in patients:
            chunks = [model.encode(chunk.to(device)).cpu()
                      for chunk in data.evaluation_chunks(tiles, patient, settings, pipeline,
                                                         batch_size)]
            if not chunks:
                raise RuntimeError(f"patient {patient} has no tiles to score")
            logit, _ = model.pool(torch.cat(chunks)[None].to(device))
            rows.append((patient, logit.item(), torch.sigmoid(logit).item()))
    return rows
