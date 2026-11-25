"""Deterministic patient bags: exact cycle coverage and integer real/synthetic quotas.

No replacement, padding, dropped remainder, global RNG, or torch dependency.
The proportion is that patient's selected synthetic count / selected total count.
"""
from collections import defaultdict, deque
import hashlib
import random


def rng(*parts):
    return random.Random(int.from_bytes(hashlib.sha256(
        "|".join(map(str, parts)).encode()).digest(), "big"))


def diverse_order(rows, seed, block_px=1792):
    """Round robin slides, spatial blocks, then parent families before taking siblings.

    Spreading is a selection rule, not a guarantee of biological independence.
    Synthetic coordinates refer to their original parent tile.
    """
    generator = rng(seed)
    slides = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        key = (int(float(row.get("x_target", 0))) // block_px,
               int(float(row.get("y_target", 0))) // block_px)
        slides[row["slide_id"]][key][row["parent_patch_id"]].append(row)

    def drain(queues):
        active = list(queues)
        result = []
        while active:
            generator.shuffle(active)
            next_active = []
            for key in active:
                result.append(queues[key].popleft())
                if queues[key]:
                    next_active.append(key)
            active = next_active
        return result

    slide_queues = {}
    for slide in sorted(slides):
        block_queues = {}
        for block in sorted(slides[slide]):
            families = {}
            for parent, family in sorted(slides[slide][block].items()):
                family = sorted(family, key=lambda r: r["tile_id"])
                generator.shuffle(family)
                families[parent] = deque(family)
            block_queues[block] = deque(drain(families))
        slide_queues[slide] = deque(drain(block_queues))
    return drain(slide_queues)


def coverage_bags(rows, bag_size, seed, cycle):
    """Each tile exactly once, with cumulative synthetic discrepancy < 1 tile.

    A complete 5,000-tile pool with 1,000 synthetics has exactly 20% synthetic
    exposure per cycle, including the partial last bag. Full bags use the two
    nearest integer quotas (e.g. 12 or 13 of 64); no probabilistic omission.
    """
    if type(bag_size) is not int or bag_size < 1 or type(cycle) is not int or cycle < 0:
        raise ValueError("Positive integer bag size and nonnegative integer cycle required")
    if not rows or len({r["tile_id"] for r in rows}) != len(rows):
        raise ValueError("A nonempty pool of unique tile IDs is required")
    if len({r["patient_id"] for r in rows}) != 1:
        raise ValueError("A bag schedule belongs to exactly one patient")
    if any(r["kind"] not in ("real", "synthetic") for r in rows):
        raise ValueError("Unknown tile kind")
    patient = rows[0]["patient_id"]
    real = diverse_order([r for r in rows if r["kind"] == "real"],
                         (seed, patient, cycle, "real"))
    synthetic = diverse_order([r for r in rows if r["kind"] == "synthetic"],
                              (seed, patient, cycle, "synthetic"))
    count, syn_count = len(rows), len(synthetic)
    result, real_pos, syn_pos = [], 0, 0
    for start in range(0, count, bag_size):
        end = min(start + bag_size, count)
        take_syn = end * syn_count // count - start * syn_count // count
        take_real = end - start - take_syn
        bag = real[real_pos:real_pos + take_real] + synthetic[syn_pos:syn_pos + take_syn]
        real_pos += take_real
        syn_pos += take_syn
        rng(seed, patient, cycle, start, "within-bag").shuffle(bag)
        result.append(bag)
    assert real_pos == len(real) and syn_pos == len(synthetic)
    assert sum(map(len, result)) == count
    return result


def select_pools(real, variants, cap, seed):
    """Nested real selections; use all accepted variants up to an explicit real-retaining cap.

    All arms have min(cap, real_count) tiles. Refuse overflow rather than silently
    truncate variants or create an all-synthetic patient. Patients lacking any
    variants are retained with an unchanged all-real bag in each arm.
    """
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("A positive integer patient cap is required")
    ordered = diverse_order(real, (seed, "pool"))
    count = len(ordered) if cap is None else min(cap, len(ordered))
    if not count:
        raise ValueError("No real tiles for patient")
    pools = {"real": ordered[:count]}
    for arm, synthetic in variants.items():
        if len(synthetic) >= count and synthetic:
            raise ValueError(f"{arm}: {len(synthetic)} variants leave no real tile within cap {count}; decide a larger cap or explicit subsampling")
        pools[arm] = ordered[:count - len(synthetic)] + list(synthetic)
    return pools
