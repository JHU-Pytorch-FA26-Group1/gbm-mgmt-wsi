import unittest
from collections import Counter

from alternative_models.sampling import coverage_bags, diverse_order, select_pools


def rows(real=4000, synthetic=1000, patient="P"):
    result = []
    for kind, count in (("real", real), ("synthetic", synthetic)):
        for i in range(count):
            result.append(dict(tile_id=f"{kind}-{i}", patient_id=patient, kind=kind,
                               slide_id=f"S{i%3}", parent_patch_id=f"parent-{i//8}",
                               x_target=(i%20)*224, y_target=(i//20)*224))
    return result


class Sampling(unittest.TestCase):
    def test_full_5000_coverage_and_exact_cumulative_ratio(self):
        pool = rows()
        for size, expected_bags in ((64, 79), (256, 20)):
            bags = coverage_bags(pool, size, 42, 0)
            self.assertEqual(len(bags), expected_bags)
            self.assertEqual(Counter(r["tile_id"] for b in bags for r in b),
                             Counter(r["tile_id"] for r in pool))
            seen, synthetic = 0, 0
            for bag in bags:
                seen += len(bag)
                synthetic += sum(r["kind"] == "synthetic" for r in bag)
                self.assertEqual(synthetic, seen*1000//5000)
                self.assertLess(abs(synthetic-seen*.2), 1)
            self.assertEqual(synthetic, 1000)

    def test_reproducibility_order_independence_and_new_cycle(self):
        pool = rows(80, 20)
        first = coverage_bags(pool, 64, 10, 0)
        self.assertEqual(first, coverage_bags(list(reversed(pool)), 64, 10, 0))
        self.assertNotEqual(first, coverage_bags(pool, 64, 10, 1))

    def test_small_pools_are_not_padded_or_dropped(self):
        for real, synthetic in ((1, 0), (2, 1), (63, 1), (64, 1), (0, 5), (19, 0)):
            pool = rows(real, synthetic)
            bags = coverage_bags(pool, 64, 0, 0)
            self.assertEqual(sum(map(len, bags)), len(pool))
            self.assertTrue(all(0 < len(b) <= 64 for b in bags))

    def test_bad_ids_patient_mixing_and_bag_size_are_refused(self):
        pool = rows(4, 2)
        for bad in ([], pool+[pool[0]], pool+rows(1, 0, "Q")):
            with self.assertRaises(ValueError):
                coverage_bags(bad, 64, 0, 0)
        for size in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                coverage_bags(pool, size, 0, 0)

    def test_spreads_families_before_repeating_siblings_in_one_stratum(self):
        pool = rows(0, 32)
        for row in pool:
            row.update(slide_id="S", x_target=0, y_target=0)
        order = diverse_order(pool, 0)
        for start in range(0, 32, 4):
            self.assertEqual(len({r["parent_patch_id"] for r in order[start:start+4]}), 4)

    def test_nested_real_pools_keep_every_variant_with_equal_total(self):
        real = rows(20, 0)
        resize, crop = rows(0, 4), rows(0, 6)
        pools = select_pools(real, {"resize": resize, "crop": crop}, 15, 0)
        self.assertEqual({len(p) for p in pools.values()}, {15})
        first = {r["tile_id"] for r in pools["real"]}
        second = {r["tile_id"] for r in pools["resize"] if r["kind"] == "real"}
        third = {r["tile_id"] for r in pools["crop"] if r["kind"] == "real"}
        self.assertTrue(third < second < first)
        self.assertEqual(pools["resize"][-4:], resize)

    def test_overflow_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            select_pools(rows(10, 0), {"resize": rows(0, 10)}, 10, 0)

    def test_patient_without_variants_is_retained(self):
        pools = select_pools(rows(20, 0), {"resize": []}, 10, 0)
        self.assertEqual(pools["real"], pools["resize"])


if __name__ == "__main__":
    unittest.main()
