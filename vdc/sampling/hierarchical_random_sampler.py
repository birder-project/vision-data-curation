import logging
import random
from collections import defaultdict
from collections import deque
from typing import Optional

from birder.common.lib import set_random_seeds

from vdc.sampling.base_sampler import BaseSampler
from vdc.sampling.cluster import ClusterInfo

logger = logging.getLogger(__name__)


class HierarchicalRandomSampler(BaseSampler):
    """
    Implements a hierarchical random sampling strategy

    Samples are distributed proportionally across the cluster hierarchy using
    the Largest Remainder Method, and then randomly selected at the lowest level.
    """

    # pylint: disable=too-many-locals,too-many-branches
    def sample(self, cluster_info: ClusterInfo, total_samples: int, random_seed: Optional[int] = None) -> list[str]:
        if random_seed is not None:
            set_random_seeds(random_seed)
            logger.debug(f"Using random seed: {random_seed}")

        if total_samples >= cluster_info.total_samples:
            raise ValueError(
                "Total samples is greater than or equal to the total number of available samples "
                f"({total_samples:,} >= {cluster_info.total_samples:,})"
            )

        # Dictionary to store how many samples to take from each (level, cluster_id): num_samples_to_take
        samples_to_take_per_cluster: dict[tuple[int, int], int] = defaultdict(int)

        top_level_cluster_ids = cluster_info.get_top_level_cluster_ids()
        top_level_capacities_map: dict[int, int] = {}
        for cluster_id in top_level_cluster_ids:
            size = cluster_info.get_cluster_size(cluster_info.max_level, cluster_id)
            if size > 0:
                top_level_capacities_map[cluster_id] = size

        if len(top_level_capacities_map) == 0:
            logger.warning("No samples found in top-level clusters")
            return []

        logger.info(f"Distributing {total_samples} samples across {len(top_level_cluster_ids)} top-level clusters.")
        top_level_allocations = self.allocator.allocate(total_samples, top_level_capacities_map)
        for cluster_id, count in top_level_allocations.items():
            samples_to_take_per_cluster[(cluster_info.max_level, cluster_id)] = count

        # Distribute down the hierarchy using a queue
        queue = deque(
            [(cluster_info.max_level, cid) for cid in top_level_cluster_ids if top_level_allocations.get(cid, 0) > 0]
        )
        while len(queue) > 0:
            (current_level, parent_cluster_id) = queue.popleft()
            if current_level == 0:  # Reached bottom, level 0 clusters contain samples directly
                continue

            num_samples_for_parent = samples_to_take_per_cluster.get((current_level, parent_cluster_id), 0)
            if num_samples_for_parent == 0:
                continue

            child_level = current_level - 1
            child_cluster_ids = cluster_info.get_child_cluster_ids(current_level, parent_cluster_id)
            if len(child_cluster_ids) == 0:  # Parent has no children, this branch ends
                continue

            child_capacities_map: dict[int, int] = {}
            total_size_of_children_in_parent = sum(
                cluster_info.get_cluster_size(child_level, cid) for cid in child_cluster_ids
            )
            if total_size_of_children_in_parent == 0:
                logger.debug(
                    f"Parent cluster (level {current_level}, id {parent_cluster_id}) has no samples in its "
                    f"children at level {child_level}. Cannot distribute."
                )
                continue

            for child_id in child_cluster_ids:
                child_size = cluster_info.get_cluster_size(child_level, child_id)
                if child_size > 0:
                    child_capacities_map[child_id] = child_size

            if len(child_capacities_map) == 0:
                continue

            child_allocations = self.allocator.allocate(num_samples_for_parent, child_capacities_map)
            for child_id, count in child_allocations.items():
                samples_to_take_per_cluster[(child_level, child_id)] += count
                if count > 0:
                    queue.append((child_level, child_id))

        # Collect actual samples from level 0 clusters
        selected_samples: list[str] = []
        level0_clusters_to_sample = [
            (cluster_id, count)
            for (level, cluster_id), count in samples_to_take_per_cluster.items()
            if level == 0 and count > 0
        ]

        logger.info(f"Collecting samples from {len(level0_clusters_to_sample):,} level 0 clusters")
        for cluster_id, num_to_take in level0_clusters_to_sample:
            available_samples = cluster_info.get_samples_in_level0_cluster(cluster_id)
            actual_num_to_take = min(num_to_take, len(available_samples))
            if actual_num_to_take == 0:
                continue

            # Randomly select samples from the current level 0 cluster
            sampled_from_cluster = random.sample(available_samples, actual_num_to_take)
            selected_samples.extend(sampled_from_cluster)

        # Cap the final list if, by chance, more samples were collected than requested
        if len(selected_samples) > total_samples:
            selected_samples = random.sample(selected_samples, total_samples)

        return selected_samples
