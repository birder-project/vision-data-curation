import importlib.resources
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow.dataset as ds
import torch
import torch.utils.data


def load_default_bundled_config() -> Optional[dict[str, Any]]:
    try:
        resource_path = importlib.resources.files("vdc.conf").joinpath("config.json")

        if resource_path.is_file() is True:
            with resource_path.open("r", encoding="utf-8") as handle:
                config: dict[str, Any] = json.load(handle)

            return config

    except ModuleNotFoundError:
        # Module not found, try at alternate locations
        pass

    for file_path in ["config.json", "vdc/conf/config.json"]:
        if os.path.exists(file_path) is True:
            return read_json("config.json")

    return None


def read_json(json_path: str) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)

    return data


def read_embeddings(path: str) -> npt.NDArray[np.float32]:
    schema = pl.read_csv(path, n_rows=1).schema
    schema_overrides = {name: (pl.Float32 if dtype == pl.Float64 else dtype) for name, dtype in schema.items()}

    df_lazy = pl.scan_csv(path, schema_overrides=schema_overrides).select(pl.exclude(["sample"]))
    return df_lazy.collect().to_numpy()


def csv_iter(path: str, batch_size: Optional[int] = None, batches_per_yield: int = 100) -> Iterator[pl.DataFrame]:
    """
    Reads a CSV file in batches and yields concatenated Polars DataFrames

    This generator function provides an efficient way to process large CSV files
    by reading them in configurable batches, concatenating these batches into
    single Polars DataFrames, and yielding each consolidated DataFrame.
    This helps in managing memory when dealing with datasets that do not fit
    entirely into RAM.

    Parameters
    ----------
    path
        The path to the CSV file.
    batch_size
        The 'batch_size' argument passed directly to 'polars.read_csv_batched()'.
        This hints to Polars how large its internal chunks should be.
        If None, Polars will determine an optimal size.
        Warning:
          As of Polars versions prior to the merge of https://github.com/pola-rs/polars/pull/23996
          (GitHub issue link: https://github.com/pola-rs/polars/issues/19978),
          this 'batch_size' might not be strictly respected by Polars for the individual internal batches.
    batches_per_yield
        The number of internal Polars batches to read and concatenate
        before yielding a single DataFrame.

    Yields
    ------
    A concatenated DataFrame containing data from the current set of batches.

    Examples
    --------
    >>> total_rows = 0
    >>> for df_chunk in csv_iter(dummy_csv_path):
    ...     # Perform operations on df_chunk, e.g., filter, aggregate, write to another file
    ...     total_rows += len(df_chunk)
    >>> print(f"Total rows processed: {total_rows}")
    """

    kwargs = {}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size

    schema = pl.read_csv(path, n_rows=1).schema
    schema_overrides = {name: (pl.Float32 if dtype == pl.Float64 else dtype) for name, dtype in schema.items()}

    reader = pl.read_csv_batched(path, schema_overrides=schema_overrides, **kwargs)  # type: ignore[arg-type]

    while True:
        batches: Optional[list[pl.DataFrame]] = reader.next_batches(batches_per_yield)

        if batches is None or len(batches) == 0:
            break

        yield pl.concat(batches, how="vertical")


class InferenceCSVDataset(torch.utils.data.IterableDataset):  # pylint: disable=abstract-method
    """
    PyTorch IterableDataset for loading numeric CSV data for inference

    This dataset loads CSV files using PyArrow and yields individual rows as PyTorch tensors.
    All CSV columns are assumed to be numeric and are converted to float32 tensors. The dataset
    supports multi-worker data loading by distributing file fragments across workers.

    Notes
    -----
    - All CSV columns must contain numeric data only
    - All numeric types are converted to float32
    - Supports multi-worker data loading with automatic fragment distribution
    - Empty batches are automatically skipped
    """

    def __init__(
        self,
        file_paths: str | list[str],
        pyarrow_batch_size: int = 1024,
        columns_to_drop: Optional[list[str]] = None,
        metadata_columns: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.file_paths = file_paths
        self.pyarrow_batch_size = pyarrow_batch_size
        self._columns_to_drop = set(columns_to_drop) if columns_to_drop is not None else set()
        self._metadata_columns = metadata_columns if metadata_columns is not None else []

        overlap = self._columns_to_drop.intersection(set(self._metadata_columns))
        if len(overlap) > 0:
            raise ValueError(f"Overlap found between 'columns_to_drop' and 'metadata_columns': {overlap}")

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """
        Iterate over individual rows from the CSV files as PyTorch tensors

        Each yielded item is a tuple where the first element is the numeric row tensor
        and subsequent elements are the string values for that row, in the order specified
        by 'metadata_columns'. If no metadata columns are specified, the tuple will contain
        only the numeric row tensor.

        Yields
        ------
        A tuple (numeric_row_tensor, metadata_value_1, metadata_value_2, ...)
        where the numeric tensor has a dtype float32.

        Notes
        -----
        When using multiple workers, fragments are distributed using round-robin
        assignment based on worker ID. Workers with no assigned fragments will
        return early without yielding any data.
        """

        worker_info = torch.utils.data.get_worker_info()
        base_dataset = ds.dataset(self.file_paths, format="csv")

        if worker_info is None:
            worker_fragments = list(base_dataset.get_fragments())
        else:
            all_fragments = list(base_dataset.get_fragments())
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            worker_fragments = []
            for i, fragment in enumerate(all_fragments):
                if i % num_workers == worker_id:
                    worker_fragments.append(fragment)

        if len(worker_fragments) == 0:
            # Worker has no fragments assigned (e.g., more workers than actual fragments)
            return

        numeric_columns = []
        for col_name in base_dataset.schema.names:
            if col_name not in self._columns_to_drop and col_name not in self._metadata_columns:
                numeric_columns.append(col_name)

        columns_for_scanner = numeric_columns + self._metadata_columns
        for fragment in worker_fragments:
            scanner = fragment.scanner(batch_size=self.pyarrow_batch_size, columns=columns_for_scanner)

            for batch in scanner.to_batches():
                if batch is None or batch.num_rows == 0:
                    continue

                numeric_batch = batch.select(numeric_columns)
                tensor_batch = torch.from_numpy(numeric_batch.to_tensor().to_numpy()).to(torch.float32)

                metadata_values_per_column: dict[str, list[str]] = {}
                for col_name in self._metadata_columns:
                    metadata_values_per_column[col_name] = batch.column(col_name).to_pylist()

                for i in range(batch.num_rows):
                    row_tensor = tensor_batch[i]

                    row_metadata_values = [
                        metadata_values_per_column[col_name][i] for col_name in self._metadata_columns
                    ]

                    yield (row_tensor, *row_metadata_values)


def build_backup_path(source: Path | str, backup_root: Path | str) -> Path:
    """
    Generate a safe backup path that preserves the directory structure of the source

    For absolute paths (e.g., /foo/bar.txt), the path is reconstructed relative to its root or drive.
    For relative paths (e.g., foo/bar.txt or ../baz.txt), ".." components are replaced to prevent
    path traversal issues.


    Parameters
    ----------
    source
        The original path of the file to be backed up.
    backup_root
        The root directory where backups should be stored.

    Returns
    -------
    The full path to the safe backup location.

    Raises
    ------
    ValueError
        If the source path is empty or resolves to a problematic structure
        that cannot be safely represented (e.g., a root directory by itself).
    """

    source = Path(source)
    backup_root = Path(backup_root)
    if source.is_absolute() is True:
        sanitized_parts = []
        for part in source.parts:
            # Skip Windows drive letters like "C:"
            if part in ("/", "\\") or (len(part) == 2 and part.endswith(":") is True):
                continue

            sanitized_parts.append(part)

    else:
        sanitized_parts = []
        for part in source.parts:
            if part == "..":
                sanitized_parts.append("parent")
            elif part in (".", ""):
                continue
            else:
                sanitized_parts.append(part)

    if len(sanitized_parts) == 0:
        raise ValueError(f"Invalid source path: {source}")

    return backup_root / Path(*sanitized_parts)
