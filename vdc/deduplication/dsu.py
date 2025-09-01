class DSU:
    """
    Disjoint Set Union (Union-Find) data structure

    This class implements a Disjoint Set Union data structure with path compression
    and union by rank optimizations.
    """

    def __init__(self, elements: list[str]):
        """
        Initialize the DSU

        Parameters
        ----------
        elements
            List of elements to initialize the disjoint sets with.
            Each element will initially be in its own separate set.
        """

        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, i: str) -> str:
        root = i
        # Find the root
        while self.parent[root] != root:
            root = self.parent[root]

        # Path compression
        while i != root:
            parent = self.parent[i]
            self.parent[i] = root
            i = parent

        return root

    def union(self, i: str, j: str) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by rank
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1

            return True

        return False

    def connected(self, element1: str, element2: str) -> bool:
        """
        Check if two elements are in the same connected component
        """

        return self.find(element1) == self.find(element2)

    def get_components(self) -> dict[str, list[str]]:
        """
        Get all connected components as a dictionary
        """

        components: dict[str, list[str]] = {}
        for element in self.parent:
            root = self.find(element)
            if root not in components:
                components[root] = []

            components[root].append(element)

        return components
