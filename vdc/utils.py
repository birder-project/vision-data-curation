import importlib.resources
import json
import logging
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Optional

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


def load_default_bundled_config() -> Optional[dict[str, Any]]:
    # Configuration loading order:
    # 1. Local override in the current working directory.
    # 2. Bundled configuration file in a cloned repository.
    # 3. Bundled configuration file within the installed Python package (vdc.conf).

    for file_path in ["config.json", "vdc/conf/config.json"]:
        if os.path.exists(file_path) is True:
            return read_json(file_path)

    try:
        resource_path = importlib.resources.files("vdc.conf").joinpath("config.json")

        if resource_path.is_file() is True:
            with resource_path.open("r", encoding="utf-8") as handle:
                config: dict[str, Any] = json.load(handle)

            return config

    except ModuleNotFoundError:
        # Module not found
        pass

    return None


def read_json(json_path: str) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)

    return data


def read_vector_file(path: str) -> npt.NDArray[np.float32]:
    """
    Load embeddings or logits from a Parquet or CSV file

    Parameters
    ----------
    path
        Path to the input file. Supported formats are:
        - Parquet (.parquet): may contain either embeddings or logits.
        - CSV (.csv): may contain embeddings or logits.

    Returns
    -------
    The loaded vectors with shape of (n_samples, n_features).

    Raises
    ------
    ValueError
        If the file format is not supported.

    Notes
    -----
    This function expects data files to conform to the "Birder format".
    A detailed specification:
    https://gitlab.com/birder/birder/-/blob/main/docs/inference.md#classification---output-files
    """

    if path.endswith(".parquet"):
        schema = pl.read_parquet_schema(path)
        if len(schema) == 2 and "embedding" in schema:
            # Embeddings Parquet
            return pl.read_parquet(path, columns=["embedding"]).to_series().to_numpy()

        # Logits Parquet
        return pl.read_parquet(path).select(pl.exclude(["sample"])).to_numpy()

    if path.endswith(".csv"):
        schema = pl.read_csv(path, n_rows=1).schema
        schema_overrides = {name: (pl.Float32 if dtype == pl.Float64 else dtype) for name, dtype in schema.items()}

        # Both logits and embeddings file have the same schema as CSV
        df_lazy = pl.scan_csv(path, schema_overrides=schema_overrides).select(pl.exclude(["sample"]))
        return df_lazy.collect().to_numpy()

    raise ValueError(f"Unsupported file format for '{path}', only .parquet and .csv files are supported")


def get_file_samples(path: str) -> pl.Series:
    if path.endswith(".parquet"):
        return pl.read_parquet(path, columns=["sample"]).to_series()
    if path.endswith(".csv"):
        return pl.scan_csv(path).select(["sample"]).collect().to_series()

    raise ValueError(f"Unsupported file format for '{path}', only .parquet and .csv files are supported")


def data_file_iter(path: str, batch_size: int = 1024) -> Iterator[pl.DataFrame]:
    """
    Reads CSV or Parquet files in batches using PyArrow and yields Polars DataFrames

    This generator function provides an efficient way to process large CSV or Parquet files
    by leveraging PyArrow's dataset scanning capabilities. It reads data in configurable
    batches, converts each batch to a Polars DataFrame, and yields it. This approach
    helps in managing memory when dealing with datasets that do not fit entirely into RAM.
    It also automatically casts Float64 columns to Float32 for memory optimization.

    Parameters
    ----------
    path
        The path to the CSV or Parquet file.
    batch_size
        The maximum number of rows for each batch.

    Yields
    ------
    A Polars DataFrame containing data from the current batch.

    Raises
    ------
    ValueError
        If the file format (determined by the file extension) is not 'csv' or 'parquet'.
    """

    file_format = Path(path).suffix[1:]
    if file_format not in {"csv", "parquet"}:
        raise ValueError(f"Unsupported file_format '{file_format}', must be 'csv' or 'parquet'")

    if file_format == "csv":
        # The PyArrow "scanner" is very slow, use Polars
        reader = pl.read_csv_batched(path, batch_size=batch_size)
        batches = reader.next_batches(100)
        while batches is not None:
            df_current_batches = pl.concat(batches)
            yield df_current_batches.cast({pl.Float64: pl.Float32})
            batches = reader.next_batches(100)

    else:
        base_dataset = ds.dataset(source=path, format=file_format)
        scanner = base_dataset.scanner(batch_size=batch_size)

        for batch in scanner.to_batches():
            if batch is None or batch.num_rows == 0:
                continue

            yield pl.from_arrow(batch).cast({pl.Float64: pl.Float32})  # type: ignore


def batch_to_tensor(batch: pa.RecordBatch, numeric_columns: list[str]) -> torch.Tensor:
    """
    Convert a PyArrow RecordBatch with numeric and fixed_size_list columns into a single torch.Tensor
    """

    if batch.num_rows == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    arrays_to_stack = []
    for col_name in numeric_columns:
        pa_array = batch.column(col_name)
        if isinstance(pa_array, pa.ChunkedArray):
            pa_array = pa_array.combine_chunks()
            np_array = pa_array.to_numpy()
        elif pa.types.is_fixed_size_list(pa_array.type) is True:
            np_array = pa_array.values.to_numpy(zero_copy_only=True)
            np_array = np_array.reshape(-1, pa_array.type.list_size)
        elif pa.types.is_list(pa_array.type) is True or pa.types.is_large_list(pa_array.type) is True:
            np_array = np.array(pa_array.to_pylist(), dtype=np.float32)
        else:
            np_array = pa_array.to_numpy()

        if np_array.ndim == 1:
            arrays_to_stack.append(np_array.reshape(-1, 1))
        else:
            arrays_to_stack.append(np_array)

    if len(arrays_to_stack) == 0:
        return torch.empty((batch.num_rows, 0), dtype=torch.float32)

    return torch.from_numpy(np.hstack(arrays_to_stack)).to(torch.float32)


class InferenceDataset(torch.utils.data.IterableDataset):  # pylint: disable=abstract-method
    """
    PyTorch IterableDataset for loading numeric CSV/Parquet data for inference

    This dataset loads files using PyArrow and yields individual rows as PyTorch tensors.
    All non-metadata columns are assumed to be numeric and converted to float32 tensors.
    Metadata columns (strings, ids, etc.) are returned alongside the numeric tensor.
    The dataset supports multi-worker data loading by distributing file fragments across workers.

    Notes
    -----
    - All non-metadata columns must contain numeric data only
    - All numeric types are converted to float32
    - Supports CSV and Parquet backends via 'file_format'
    - Supports multi-worker data loading with automatic fragment distribution
    - Empty batches are automatically skipped
    """

    def __init__(
        self,
        file_paths: str | list[str],
        file_format: Literal["csv", "parquet"] = "csv",
        pyarrow_batch_size: int = 4096,
        columns_to_drop: Optional[list[str]] = None,
        metadata_columns: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.file_paths = file_paths
        self.file_format = file_format.lower()
        if self.file_format not in {"csv", "parquet"}:
            raise ValueError(f"Unsupported file_format '{file_format}', must be 'csv' or 'parquet'")

        self.pyarrow_batch_size = pyarrow_batch_size
        self._columns_to_drop = set(columns_to_drop) if columns_to_drop is not None else set()
        self._metadata_columns = metadata_columns if metadata_columns is not None else []

        overlap = self._columns_to_drop.intersection(set(self._metadata_columns))
        if len(overlap) > 0:
            raise ValueError(f"Overlap found between 'columns_to_drop' and 'metadata_columns': {overlap}")

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """
        Iterate over individual rows from the dataset files as PyTorch tensors

        Each yielded item is a tuple where the first element is the numeric row tensor
        and subsequent elements are the metadata values for that row, in the order specified
        by 'metadata_columns'. If no metadata columns are specified, the tuple will contain
        only the numeric row tensor.

        Yields
        ------
        A tuple (numeric_row_tensor, metadata_value_1, metadata_value_2, ...)
        where the numeric tensor has dtype float32.

        Notes
        -----
        When using multiple workers, fragments are distributed using round-robin
        assignment based on worker ID. Workers with no assigned fragments will
        return early without yielding any data.
        """

        worker_info = torch.utils.data.get_worker_info()
        base_dataset = ds.dataset(self.file_paths, format=self.file_format)

        if worker_info is None:
            worker_fragments = list(base_dataset.get_fragments())
        else:
            all_fragments = list(base_dataset.get_fragments())
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            worker_fragments = [f for i, f in enumerate(all_fragments) if i % num_workers == worker_id]

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

                # numeric_batch = batch.select(numeric_columns)
                tensor_batch = batch_to_tensor(batch, numeric_columns)
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


def perform_file_deletion_with_backup(file_path_str: str, backup_dir: Optional[str]) -> str:
    """
    Performs file deletion with an optional backup

    Parameters
    ----------
    file_path_str
        The path to the file to be potentially deleted.
    backup_dir
        An optional path to a directory where the file should be backed up
        before deletion. If None, no backup is performed.

    Returns
    -------
    A string indicating the remediation status:
    - "deleted": File was successfully backed up (if specified) and deleted.
    - "file_not_found": The original file did not exist.
    - "skipped_backup_exists": Backup was skipped because the target backup file already existed.
    - "backup_failed_error": An error occurred during backup, so deletion was prevented.
    - "error": An error occurred during deletion.
    """

    original_path = Path(file_path_str)

    if original_path.exists() is False:
        logger.debug(f"File not found: {original_path}")
        return "file_not_found"

    backup_successful = False
    if backup_dir is not None:
        backup_path = build_backup_path(original_path, backup_dir)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if backup_path.exists() is True:
            logger.error(f"Backup file already exists: {backup_path}. Skipping deletion of {original_path}.")
            return "skipped_backup_exists"
        try:
            shutil.copy2(original_path, backup_path)
            logger.debug(f"Backed up {original_path} to {backup_path}")
            backup_successful = True
        except OSError as e:
            logger.error(
                f"Failed to backup {original_path} to {backup_path} (Error: {type(e).__name__} - {e}). "
                "Deletion will NOT proceed for this file to prevent data loss."
            )
            return "backup_failed_error"

    try:
        original_path.unlink()
        logger.debug(f"DELETED: {file_path_str} (backup created: {backup_successful})")
        return "deleted"
    except OSError as e:
        logger.error(f"Error deleting {file_path_str} (Error: {type(e).__name__} - {e}). Check backup if available.")
        return "error"
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            f"An unexpected error occurred during deletion of {file_path_str} (Error: {type(e).__name__} - {e}). "
            "Original file state is now potentially corrupted or partially modified. Check backup if available."
        )
        return "error"
