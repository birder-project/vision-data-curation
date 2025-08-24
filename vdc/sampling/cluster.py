import polars as pl


class ClusterInfo:
    def __init__(self, assignments_df: pl.DataFrame) -> None:
        # Validate input
        if "sample" not in assignments_df.columns:
            raise ValueError("Assignments DataFrame must contain a 'sample' column")

        level_cols = [col for col in assignments_df.columns if col.startswith("level_")]
        if len(level_cols) == 0:
            raise ValueError("Assignments DataFrame must contain 'level_' columns (e.g., level_0, level_1)")

        self.total_samples = len(assignments_df)

        # Sort level columns to ensure proper ordering
        level_cols = sorted(level_cols, key=lambda x: int(x.removeprefix("level_")))
        self.max_level = len(level_cols) - 1

        # Mapping from level 0 cluster IDs to sample paths
        grouped = assignments_df.group_by("level_0").agg(pl.col("sample").alias("samples"))
        self._level0_samples: dict[int, list[str]] = {
            row["level_0"]: row["samples"] for row in grouped.iter_rows(named=True)
        }

        # Cluster sizes for all levels
        self._cluster_sizes: dict[int, dict[int, int]] = {}
        self._num_clusters_per_level: dict[int, int] = {}
        for level in range(self.max_level + 1):
            level_col = f"level_{level}"
            sizes = assignments_df.group_by(level_col).agg(pl.len().alias("size"))
            self._cluster_sizes[level] = {row[level_col]: row["size"] for row in sizes.iter_rows(named=True)}
            self._num_clusters_per_level[level] = len(self._cluster_sizes[level])

        # Parent-child cluster relationships
        self._child_clusters: dict[int, dict[int, list[int]]] = {}
        for level in range(1, self.max_level + 1):
            current_col = f"level_{level}"  # This is the parent level
            prev_col = f"level_{level-1}"  # This is the child level

            relationships = (
                assignments_df.select([current_col, prev_col])
                .unique()
                .group_by(current_col)
                .agg(pl.col(prev_col).alias("children"))
            )
            self._child_clusters[level] = {
                row[current_col]: row["children"] for row in relationships.iter_rows(named=True)
            }

    def get_max_level(self) -> int:
        return self.max_level

    def get_top_level_cluster_ids(self) -> list[int]:
        return list(self._cluster_sizes[self.max_level].keys())

    def get_cluster_size(self, level: int, cluster_id: int) -> int:
        return self._cluster_sizes[level][cluster_id]

    def get_child_cluster_ids(self, level: int, parent_cluster_id: int) -> list[int]:
        if level == 0:
            raise ValueError("Level 0 clusters do not have child cluster IDs, they contain samples")
        if level < 1 or level > self.max_level:
            raise ValueError(f"Level must be between 1 and {self.max_level}")

        return self._child_clusters[level][parent_cluster_id]

    def get_samples_in_level0_cluster(self, level0_cluster_id: int) -> list[str]:
        return self._level0_samples[level0_cluster_id]

    def get_num_clusters_at_level(self, level: int) -> int:
        return self._num_clusters_per_level[level]

    def __len__(self) -> int:
        return self.total_samples

    def __repr__(self) -> str:
        clusters_info = []
        for level in range(self.max_level + 1):
            num_clusters = len(self._cluster_sizes[level])
            clusters_info.append(f"level_{level}: {num_clusters} clusters")

        clusters_str = ", ".join(clusters_info)

        return f"ClusterInfo(total_samples={self.total_samples}, max_level={self.max_level}, {clusters_str})"
