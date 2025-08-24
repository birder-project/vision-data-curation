from typing import Optional

from vdc.sampling.allocation import BaseAllocator
from vdc.sampling.cluster import ClusterInfo


class BaseSampler:
    """
    Abstract Base Class for all sampling strategies.
    Defines the interface that all concrete samplers must implement.
    """

    def __init__(self, allocator: BaseAllocator) -> None:
        self._allocator = allocator

    @property
    def allocator(self) -> BaseAllocator:
        return self._allocator

    def sample(self, cluster_info: ClusterInfo, total_samples: int, random_seed: Optional[int] = None) -> list[str]:
        """
        Executes the sampling strategy to select a specified number of samples from the given cluster information

        Parameters
        ----------
        cluster_info
            An instance of ClusterInfo containing the hierarchical clustering assignments.
        total_samples
            The total number of samples to select.
        random_seed
            An optional integer seed for reproducibility.

        Returns
        -------
        A list of strings, where each string is the path/identifier of a selected sample.
        """

        raise NotImplementedError
