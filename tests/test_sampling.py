# pylint: disable=protected-access

import unittest

import polars as pl

from vdc.sampling.allocation import LRMAllocator
from vdc.sampling.allocation import WaterFillingAllocator
from vdc.sampling.cluster import ClusterInfo


class TestClusterInfo(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_data = {
            "sample": [f"sample_{i:02d}" for i in range(10)],
            "level_0": [101, 101, 102, 102, 103, 103, 103, 104, 104, 105],
            "level_1": [201, 201, 201, 202, 202, 202, 203, 203, 203, 203],
            "level_2": [301, 301, 301, 301, 302, 302, 302, 302, 302, 302],
        }
        self.assignments_df = pl.DataFrame(self.sample_data)

        self.single_level_data = {"sample": [f"s{i}" for i in range(5)], "level_0": [1, 1, 2, 2, 3]}
        self.single_level_df = pl.DataFrame(self.single_level_data)

    def test_attributes_correctness(self) -> None:
        cluster_info = ClusterInfo(self.assignments_df)
        self.assertEqual(cluster_info.total_samples, 10)
        self.assertEqual(cluster_info.max_level, 2)

        expected_num_clusters = {
            0: 5,  # level_0: 101, 102, 103, 104, 105
            1: 3,  # level_1: 201, 202, 203
            2: 2,  # level_2: 301, 302
        }
        for level, count in expected_num_clusters.items():
            with self.subTest(level=level):
                self.assertEqual(cluster_info._num_clusters_per_level[level], count)

    def test_level0_samples(self) -> None:
        cluster_info = ClusterInfo(self.assignments_df)
        self.assertEqual(cluster_info._level0_samples[101], ["sample_00", "sample_01"])
        self.assertEqual(cluster_info._level0_samples[103], ["sample_04", "sample_05", "sample_06"])
        self.assertEqual(cluster_info._level0_samples[105], ["sample_09"])

        self.assertEqual(cluster_info.get_samples_in_level0_cluster(101), ["sample_00", "sample_01"])

    def test_cluster_sizes(self) -> None:
        cluster_info = ClusterInfo(self.assignments_df)
        self.assertEqual(cluster_info._cluster_sizes[0][101], 2)
        self.assertEqual(cluster_info._cluster_sizes[1][201], 3)
        self.assertEqual(cluster_info._cluster_sizes[2][302], 6)

        self.assertEqual(cluster_info.get_cluster_size(0, 101), 2)
        self.assertEqual(cluster_info.get_cluster_size(1, 202), 3)
        self.assertEqual(cluster_info.get_cluster_size(2, 301), 4)

    def test_child_clusters(self) -> None:
        cluster_info = ClusterInfo(self.assignments_df)

        # level_2 cluster 301 has children at level_1: 201, 202
        self.assertIn(201, cluster_info.get_child_cluster_ids(2, 301))
        self.assertIn(202, cluster_info.get_child_cluster_ids(2, 301))
        self.assertEqual(len(cluster_info.get_child_cluster_ids(2, 301)), 2)
        self.assertCountEqual(cluster_info.get_child_cluster_ids(2, 301), [201, 202])

        # level_2 cluster 302 has children at level_1: 202, 203
        self.assertIn(202, cluster_info.get_child_cluster_ids(2, 302))
        self.assertIn(203, cluster_info.get_child_cluster_ids(2, 302))
        self.assertEqual(len(cluster_info.get_child_cluster_ids(2, 302)), 2)
        self.assertCountEqual(cluster_info.get_child_cluster_ids(2, 302), [202, 203])

        # level_1 cluster 201 has children at level_0: 101, 102
        self.assertCountEqual(cluster_info.get_child_cluster_ids(1, 201), [101, 102])
        # level_1 cluster 202 has children at level_0: 102, 103
        self.assertCountEqual(cluster_info.get_child_cluster_ids(1, 202), [102, 103])
        # level_1 cluster 203 has children at level_0: 103, 104, 105
        self.assertCountEqual(cluster_info.get_child_cluster_ids(1, 203), [103, 104, 105])

        with self.assertRaisesRegex(ValueError, "Level 0 clusters do not have child cluster IDs"):
            cluster_info.get_child_cluster_ids(0, 101)


class TestDistributeSamplesLRM(unittest.TestCase):
    def setUp(self) -> None:
        self.distribute_samples_lrm = LRMAllocator()

    def test_basic_distribution(self) -> None:
        capacity = {1: 5, 2: 3, 3: 2}
        total = 10
        expected = {1: 5, 2: 3, 3: 2}
        self.assertEqual(self.distribute_samples_lrm.allocate(total, capacity), expected)

    def test_distribution_with_remainders(self) -> None:
        capacity = {1: 33, 2: 33, 3: 34}
        total = 10

        # Allocate 1 remaining to cluster 3
        expected = {1: 3, 2: 3, 3: 4}
        self.assertEqual(self.distribute_samples_lrm.allocate(total, capacity), expected)

        capacity = {1: 3, 2: 3, 3: 4}
        total_2 = 7

        # Allocate 1 remaining to cluster 3
        expected_2 = {1: 2, 2: 2, 3: 3}
        self.assertEqual(self.distribute_samples_lrm.allocate(total_2, capacity), expected_2)


class TestWaterFillingAllocator(unittest.TestCase):
    def setUp(self) -> None:
        self.allocator = WaterFillingAllocator()

    def test_empty_capacities_map(self) -> None:
        self.assertEqual(self.allocator.allocate(10, {}), {})

    def test_zero_total_to_distribute(self) -> None:
        capacities = {1: 10, 2: 20}
        self.assertEqual(self.allocator.allocate(0, capacities), {1: 0, 2: 0})

    def test_total_less_than_sum_of_capacities(self) -> None:
        capacities = {1: 10, 2: 20, 3: 5}
        total = 25
        # threshold = 10, should not have any remainders
        expected = {1: 10, 2: 10, 3: 5}
        self.assertEqual(self.allocator.allocate(total, capacities), expected)

    def test_total_equal_to_sum_of_capacities(self) -> None:
        capacities = {1: 10, 2: 20, 3: 5}
        total = 35  # Sum of capacities
        expected = {1: 10, 2: 20, 3: 5}
        self.assertEqual(self.allocator.allocate(total, capacities), expected)

    def test_total_exceeds_sum_of_capacities(self) -> None:
        capacities = {1: 10, 2: 20, 3: 5}
        total = 50  # More than sum of capacities
        expected = {1: 10, 2: 20, 3: 5}
        self.assertEqual(self.allocator.allocate(total, capacities), expected)

    def test_uneven_capacities_with_remainders(self) -> None:
        capacities = {1: 10, 2: 3, 3: 20, 4: 7}
        total = 25
        # threshold = 7, initial allocation 7, 3, 7, 7 => reminder = 1
        expected = {1: 7, 2: 3, 3: 8, 4: 7}
        self.assertEqual(self.allocator.allocate(total, capacities), expected)

    def test_zero_threshold(self) -> None:
        capacities = {1: 10, 2: 20}
        total = 1
        # threshold = 0, distribute reminder according to size
        expected = {1: 0, 2: 1}
        self.assertEqual(self.allocator.allocate(total, capacities), expected)
