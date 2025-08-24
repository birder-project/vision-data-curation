import numpy as np
import numpy.typing as npt


class BaseAllocator:
    """
    Abstract Base Class for all sample allocation strategies
    """

    def allocate(self, total_to_distribute: int, cluster_capacities_map: dict[int, int]) -> dict[int, int]:
        """
        Distributes a total number of samples among items based on their absolute reference values.

        Parameters
        ----------
        total_to_distribute
            The total number of samples to distribute.
        cluster_capacities_map
            A dictionary where keys are cluster IDs and values are their capacity constraints
            (i.e., the maximum number of samples each cluster can receive).

        Returns
        -------
        A dictionary mapping cluster IDs to the integer number of samples allocated to them.
        """

        raise NotImplementedError


class LRMAllocator(BaseAllocator):
    """
    Allocates samples using the Largest Remainder Method (LRM)
    """

    def allocate(self, total_to_distribute: int, cluster_capacities_map: dict[int, int]) -> dict[int, int]:
        """
        Distributes a total number of samples among items based on their proportions using the Largest Remainder Method

        This method is commonly used for proportional representation, ensuring that integers are allocated
        as proportionally as possible to given quotas, while distributing any remaining items based on
        the largest fractional remainders.

        Parameters
        ----------
        total_to_distribute
            The total number of samples to distribute.
        cluster_capacities_map
            A dictionary where keys are cluster IDs and values are their capacity constraints
            (i.e., the maximum number of samples each cluster can receive).

        Returns
        -------
        A dictionary mapping cluster IDs to the integer number of samples allocated to them.
        """

        if len(cluster_capacities_map) == 0:
            return {}
        if total_to_distribute == 0:
            return {cluster_id: 0 for cluster_id in cluster_capacities_map}

        total_capacity = sum(cluster_capacities_map.values())
        cluster_proportions_map = {
            cluster_id: value / total_capacity for cluster_id, value in cluster_capacities_map.items()
        }

        final_counts: dict[int, int] = {}
        remainders: dict[int, float] = {}
        assigned_count = 0

        # Calculate initial quotas and integer allocations
        for cluster_id, proportion in cluster_proportions_map.items():
            quota = total_to_distribute * proportion
            final_counts[cluster_id] = int(quota)
            remainders[cluster_id] = quota - final_counts[cluster_id]
            assigned_count += final_counts[cluster_id]

        remaining_to_distribute = total_to_distribute - assigned_count

        # Distribute remainders
        sorted_remainders = sorted(remainders.items(), key=lambda item: item[1], reverse=True)
        for cluster_id, _ in sorted_remainders:
            if remaining_to_distribute <= 0:
                break

            final_counts[cluster_id] += 1
            remaining_to_distribute -= 1

        return final_counts


class WaterFillingAllocator(BaseAllocator):
    """
    Allocates samples using the Water-Filling algorithm
    """

    def allocate(self, total_to_distribute: int, cluster_capacities_map: dict[int, int]) -> dict[int, int]:
        """
        Distributes a total number of samples among clusters using the water-filling algorithm with capacity constraints

        This method ensures fair allocation by first filling smaller capacity requirements completely,
        then distributing remaining resources equally among larger-capacity clusters up to a threshold.
        Any remaining samples after threshold allocation are distributed to the largest capacity clusters first.

        Parameters
        ----------
        total_to_distribute
            The total number of samples to distribute across all clusters.
        cluster_capacities_map
            A dictionary where keys are cluster IDs and values are their capacity constraints
            (i.e., the maximum number of samples each cluster can receive).

        Returns
        -------
        A dictionary mapping cluster IDs to the integer number of samples allocated to them.
        No cluster will receive more samples than its specified capacity.

        Notes
        -----
        The algorithm works in three phases:
        1. Water-filling phase: Allocates samples up to a computed threshold, ensuring smaller
        clusters get their full capacity if below threshold
        2. Remainder distribution: Any remaining samples are allocated to clusters with
        remaining capacity, prioritizing those with largest capacities
        3. Final assignment: Returns the allocation map
        """

        if len(cluster_capacities_map) == 0:
            return {}
        if total_to_distribute == 0:
            return {cluster_id: 0 for cluster_id in cluster_capacities_map}

        cluster_ids = list(cluster_capacities_map.keys())
        capacities_array = np.array(list(cluster_capacities_map.values()))

        # Find water-filling threshold and perform initial allocation
        threshold = _find_water_filling_threshold(capacities_array, total_to_distribute)
        allocated_samples = np.minimum(threshold, capacities_array)

        total_allocated = np.sum(allocated_samples)
        remaining_to_distribute = total_to_distribute - total_allocated

        # Distribute remainders
        if remaining_to_distribute > 0:
            available_capacity = capacities_array - allocated_samples
            candidates_mask = available_capacity > 0
            candidate_indices = np.where(candidates_mask)[0]

            if len(candidate_indices) > 0:
                # Sort candidates by their original capacity (descending) to prioritize larger clusters
                sorted_candidate_indices = candidate_indices[np.argsort(capacities_array[candidate_indices])[::-1]]
                for idx in sorted_candidate_indices:
                    if remaining_to_distribute <= 0:
                        break

                    allocated_samples[idx] += 1
                    remaining_to_distribute -= 1

        final_allocations = {cluster_id: int(allocated_samples[i]) for i, cluster_id in enumerate(cluster_ids)}

        return final_allocations


def _find_water_filling_threshold(capacities: npt.NDArray[np.int64], budget: int) -> int:
    """
    Find the threshold value for water-filling (threshold allocation) algorithm

    Computes the maximum threshold T such that allocating min(capacity_i, T) to each
    item i results in a total allocation less than or equal to the budget.

    Parameters
    ----------
    capacities
        Array of capacity constraints for each item. Each element represents the
        maximum amount that can be allocated to that item.
    budget
        The total amount to be allocated across all items.

    Returns
    -------
    The threshold value T such that sum(min(capacity_i, T)) <= target.
    Items with capacity < T receive their full capacity, while items with
    capacity >= T receive exactly T.

    Notes
    -----
    The water-filling algorithm ensures fairness by:
    1. Satisfying smaller capacity requirements first (in full)
    2. Distributing the remaining resources equally among larger-capacity items
    3. Using a single threshold value to determine allocations
    """

    total_sum = np.sum(capacities)
    if total_sum <= budget:
        return np.max(capacities).item()

    sorted_caps = np.sort(capacities)
    n = len(sorted_caps)
    cumsum = np.cumsum(sorted_caps)
    for i in range(n):
        prev_sum = cumsum[i - 1] if i > 0 else 0
        remaining_items = n - i
        if prev_sum + remaining_items * sorted_caps[i] >= budget:
            threshold = (budget - prev_sum) // remaining_items
            return threshold

    return 0  # Shouldn't reach here if inputs are valid
