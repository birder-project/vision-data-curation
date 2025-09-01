import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import torch

from vdc import utils


class TestInferenceDatasetCSV(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.csv_file_path_1 = os.path.join(self.temp_dir, "test_data1.csv")

        # Create a simple CSV file
        with open(self.csv_file_path_1, "w", encoding="utf-8") as f:
            f.write("col1,col2,col3\n")
            f.write("1.0,2.0,3.0\n")
            f.write("4.0,5.0,6.0\n")
            f.write("7.0,8.0,9.0\n")

        self.csv_file_path_2 = os.path.join(self.temp_dir, "test_data2.csv")
        with open(self.csv_file_path_2, "w", encoding="utf-8") as f:
            f.write("col1,col2,col3\n")
            f.write("10.0,11.0,12.0\n")
            f.write("13.0,14.0,15.0\n")

        self.csv_file_path_metadata = os.path.join(self.temp_dir, "test_data_with_metadata.csv")
        with open(self.csv_file_path_metadata, "w", encoding="utf-8") as f:
            f.write("id_col,feature1,feature2,extra_info\n")
            f.write("id_A,100.0,200.0,some_text_A\n")
            f.write("id_B,101.0,201.0,some_text_B\n")
            f.write("id_C,102.0,202.0,some_text_C\n")

    def tearDown(self) -> None:
        os.remove(self.csv_file_path_1)
        os.remove(self.csv_file_path_2)
        os.remove(self.csv_file_path_metadata)
        os.rmdir(self.temp_dir)

    def test_basic_iteration(self) -> None:
        dataset = utils.InferenceDataset(self.csv_file_path_1)
        collected_data = list(dataset)

        expected_data = [
            (torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),),
            (torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32),),
            (torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32),),
        ]

        self.assertEqual(len(collected_data), len(expected_data))

        for i, (actual, expected) in enumerate(zip(collected_data, expected_data)):
            with self.subTest(row_index=i):
                self.assertEqual(len(actual), 1)
                actual_tensor = actual[0]
                expected_tensor = expected[0]

                torch.testing.assert_close(actual_tensor, expected_tensor)
                self.assertEqual(actual_tensor.dtype, torch.float32)
                self.assertEqual(expected_tensor.shape, (3,))

    def test_columns_to_drop(self) -> None:
        dataset = utils.InferenceDataset(file_paths=self.csv_file_path_1, columns_to_drop=["col2"])
        collected_data = list(dataset)

        expected_dropped_data = [
            (torch.tensor([1.0, 3.0], dtype=torch.float32),),
            (torch.tensor([4.0, 6.0], dtype=torch.float32),),
            (torch.tensor([7.0, 9.0], dtype=torch.float32),),
        ]

        self.assertEqual(len(collected_data), len(expected_dropped_data))

        for i, (actual, expected) in enumerate(zip(collected_data, expected_dropped_data)):
            with self.subTest(row_index=i):
                self.assertEqual(len(actual), 1)
                actual_tensor = actual[0]
                expected_tensor = expected[0]

                torch.testing.assert_close(actual_tensor, expected_tensor)
                self.assertEqual(actual_tensor.dtype, torch.float32)
                self.assertEqual(expected_tensor.shape, (2,))

    def test_dataloader_multi_worker(self) -> None:
        all_file_paths = [self.csv_file_path_1, self.csv_file_path_2]
        dataset = utils.InferenceDataset(file_paths=all_file_paths)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)
        collected_data = []
        for row_tuple in dataloader:
            self.assertEqual(len(row_tuple), 1)
            collected_data.append(row_tuple)

        # Sort the collected data to make comparison deterministic, as worker order is not guaranteed
        collected_data_list = [t[0].tolist() for t in collected_data]
        collected_data_list.sort()

        expected_total_data = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]

        self.assertEqual(len(collected_data), len(expected_total_data))

        for i, (actual, expected) in enumerate(zip(collected_data_list, expected_total_data)):
            with self.subTest(row_index=i):
                self.assertEqual(actual, expected)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, num_workers=1)
        for row_tuple in dataloader:
            self.assertEqual(len(row_tuple), 1)
            self.assertEqual(row_tuple[0].size(), (5, 3))

    def test_metadata_columns(self) -> None:
        dataset = utils.InferenceDataset(
            file_paths=self.csv_file_path_metadata, metadata_columns=["id_col", "extra_info"]
        )
        collected_data = list(dataset)

        expected_data = [
            (torch.tensor([100.0, 200.0], dtype=torch.float32), "id_A", "some_text_A"),
            (torch.tensor([101.0, 201.0], dtype=torch.float32), "id_B", "some_text_B"),
            (torch.tensor([102.0, 202.0], dtype=torch.float32), "id_C", "some_text_C"),
        ]

        self.assertEqual(len(collected_data), len(expected_data))

        for i, (actual, expected) in enumerate(zip(collected_data, expected_data)):
            with self.subTest(row_index=i):
                self.assertIsInstance(actual, tuple)
                self.assertEqual(len(actual), len(expected))

                (actual_tensor, *actual_metadata) = actual
                (expected_tensor, *expected_metadata) = expected

                torch.testing.assert_close(actual_tensor, expected_tensor)
                self.assertEqual(actual_tensor.dtype, torch.float32)
                self.assertEqual(actual_tensor.shape, (2,))

                # Check the metadata strings
                self.assertEqual(actual_metadata, expected_metadata)

    def test_metadata_reverse_columns(self) -> None:
        dataset = utils.InferenceDataset(
            file_paths=self.csv_file_path_metadata, metadata_columns=["extra_info", "id_col"]
        )
        collected_data = list(dataset)

        expected_data = [
            (torch.tensor([100.0, 200.0], dtype=torch.float32), "some_text_A", "id_A"),
            (torch.tensor([101.0, 201.0], dtype=torch.float32), "some_text_B", "id_B"),
            (torch.tensor([102.0, 202.0], dtype=torch.float32), "some_text_C", "id_C"),
        ]

        self.assertEqual(len(collected_data), len(expected_data))

        for i, (actual, expected) in enumerate(zip(collected_data, expected_data)):
            with self.subTest(row_index=i):
                self.assertIsInstance(actual, tuple)
                self.assertEqual(len(actual), len(expected))

                (actual_tensor, *actual_metadata) = actual
                (expected_tensor, *expected_metadata) = expected

                torch.testing.assert_close(actual_tensor, expected_tensor)
                self.assertEqual(actual_tensor.dtype, torch.float32)
                self.assertEqual(actual_tensor.shape, (2,))

                # Check the metadata strings
                self.assertEqual(actual_metadata, expected_metadata)


class TestInferenceDatasetParquet(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

        # Embeddings Parquet (fixed size)
        self.parquet_embeddings_path = os.path.join(self.temp_dir, "embeddings.parquet")
        df_embeddings = pl.DataFrame(
            {
                "sample": ["id1", "id2", "id3"],
                "embedding": [
                    np.array([0.1, 0.2, 0.3]),
                    np.array([0.4, 0.5, 0.6]),
                    np.array([0.7, 0.8, 0.9]),
                ],
            }
        )
        df_embeddings.write_parquet(self.parquet_embeddings_path)

        # Embeddings Parquet
        self.parquet_embeddings_list_path = os.path.join(self.temp_dir, "embeddings_list.parquet")
        df_embeddings = pl.DataFrame(
            {
                "sample": ["id1", "id2", "id3"],
                "embedding": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9],
                ],
            }
        )
        df_embeddings.write_parquet(self.parquet_embeddings_list_path)

        # Logits Parquet
        self.parquet_logits_path = os.path.join(self.temp_dir, "logits.parquet")
        df_logits = pl.DataFrame(
            {
                "sample": ["a", "b", "c"],
                "class1": np.array([1.0, 2.0, 3.0]),
                "class2": np.array([4.0, 5.0, 6.0]),
            }
        )
        df_logits.write_parquet(self.parquet_logits_path)

    def tearDown(self) -> None:
        os.remove(self.parquet_embeddings_path)
        os.remove(self.parquet_embeddings_list_path)
        os.remove(self.parquet_logits_path)
        os.rmdir(self.temp_dir)

    def test_embeddings_parquet_iteration(self) -> None:
        dataset = utils.InferenceDataset(
            file_paths=self.parquet_embeddings_path, file_format="parquet", columns_to_drop=["sample"]
        )
        collected = list(dataset)

        expected = [
            (torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),),
            (torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32),),
            (torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32),),
        ]

        self.assertEqual(len(collected), len(expected))
        for (actual_tensor,), (expected_tensor,) in zip(collected, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)
            self.assertEqual(actual_tensor.dtype, torch.float32)

    def test_embeddings_list_parquet_iteration(self) -> None:
        dataset = utils.InferenceDataset(
            file_paths=self.parquet_embeddings_list_path, file_format="parquet", columns_to_drop=["sample"]
        )
        collected = list(dataset)

        expected = [
            (torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),),
            (torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32),),
            (torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32),),
        ]

        self.assertEqual(len(collected), len(expected))
        for (actual_tensor,), (expected_tensor,) in zip(collected, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)
            self.assertEqual(actual_tensor.dtype, torch.float32)

    def test_logits_parquet_iteration(self) -> None:
        dataset = utils.InferenceDataset(
            file_paths=self.parquet_logits_path,
            file_format="parquet",
            metadata_columns=["sample"],
        )
        collected = list(dataset)

        expected = [
            (torch.tensor([1.0, 4.0], dtype=torch.float32), "a"),
            (torch.tensor([2.0, 5.0], dtype=torch.float32), "b"),
            (torch.tensor([3.0, 6.0], dtype=torch.float32), "c"),
        ]

        self.assertEqual(len(collected), len(expected))
        for (actual_tensor, actual_sample), (expected_tensor, expected_sample) in zip(collected, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)
            self.assertEqual(actual_sample, expected_sample)


class TestDataFileIter(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

        # Embeddings Parquet
        self.embeddings_path = os.path.join(self.temp_dir, "embeddings.parquet")
        pl.DataFrame({"embedding": [[0.11, 0.12], [0.21, 0.22], [0.31, 0.32]]}).write_parquet(self.embeddings_path)

        # Logits Parquet
        self.logits_path = os.path.join(self.temp_dir, "logits.parquet")
        pl.DataFrame(
            {
                "sample": ["x", "y"],
                "logit1": [0.5, 1.5],
                "logit2": [1.0, 2.0],
            }
        ).write_parquet(self.logits_path)

    def tearDown(self) -> None:
        os.remove(self.embeddings_path)
        os.remove(self.logits_path)
        os.rmdir(self.temp_dir)

    def test_embeddings_iter(self) -> None:
        batches = list(utils.data_file_iter(self.embeddings_path, batch_size=2))
        self.assertTrue(all(isinstance(b, pl.DataFrame) for b in batches))

        concatenated = pl.concat(batches)
        expected = pl.DataFrame({"embedding": [[0.11, 0.12], [0.21, 0.22], [0.31, 0.32]]})
        self.assertEqual(concatenated.shape, expected.shape)
        self.assertEqual(concatenated.to_dict(as_series=False), expected.to_dict(as_series=False))

    def test_logits_iter(self) -> None:
        batches = list(utils.data_file_iter(self.logits_path, batch_size=1))
        self.assertTrue(all(isinstance(b, pl.DataFrame) for b in batches))

        concatenated = pl.concat(batches)
        expected = pl.DataFrame({"sample": ["x", "y"], "logit1": [0.5, 1.5], "logit2": [1.0, 2.0]})
        self.assertEqual(concatenated.shape, expected.shape)
        self.assertEqual(concatenated.to_dict(as_series=False), expected.to_dict(as_series=False))


class TestCreateSafeBackupPath(unittest.TestCase):
    def test_relative_paths(self) -> None:
        backup_root = Path("backup")
        test_cases = [
            ("data/scraped_data/img1.jpg", "data/scraped_data/img1.jpg"),
            ("data/scraped_data/img2.png", "data/scraped_data/img2.png"),
            ("data/scraped_data/subfolder/img3.jpg", "data/scraped_data/subfolder/img3.jpg"),
        ]
        for source, expected_relative in test_cases:
            with self.subTest(source=source):
                expected_path = backup_root / Path(expected_relative)
                self.assertEqual(utils.build_backup_path(source, backup_root), expected_path)

    def test_relative_paths_with_parent_and_current_references(self) -> None:
        backup_root = Path("backup")
        test_cases = [
            ("../other_project/data/scraped_data/img1.jpg", "parent/other_project/data/scraped_data/img1.jpg"),
            ("../../shared_data/images/img5.png", "parent/parent/shared_data/images/img5.png"),
            ("../parent_dir/../sibling_dir/img6.jpg", "parent/parent_dir/parent/sibling_dir/img6.jpg"),
            ("./data/scraped_data/img9.jpg", "data/scraped_data/img9.jpg"),
            ("data/../data/scraped_data/img10.png", "data/parent/data/scraped_data/img10.png"),
            ("data/./scraped_data/img11.jpg", "data/scraped_data/img11.jpg"),
            ("../../parent/../another_parent/path/img.jpg", "parent/parent/parent/parent/another_parent/path/img.jpg"),
        ]
        for source, expected_relative in test_cases:
            with self.subTest(source=source):
                expected_path = backup_root / Path(expected_relative)
                self.assertEqual(utils.build_backup_path(source, backup_root), expected_path)

    def test_absolute_paths_unix_like(self) -> None:
        backup_root = Path("backup")
        test_cases = [
            ("/mnt/data/scraped_data/img1.jpg", "mnt/data/scraped_data/img1.jpg"),
            ("/mnt/data/scraped_data/dataset.csv", "mnt/data/scraped_data/dataset.csv"),
            ("/home/user/project/data/img4.jpg", "home/user/project/data/img4.jpg"),
        ]
        for source, expected_relative in test_cases:
            with self.subTest(source=source):
                expected_path = backup_root / Path(expected_relative)
                self.assertEqual(utils.build_backup_path(source, backup_root), expected_path)

    def test_absolute_paths_windows_like(self) -> None:
        backup_root = Path("backup")
        test_cases = [
            ("C:\\Users\\Data\\scraped_data\\img7.jpg", "C:\\Users\\Data\\scraped_data\\img7.jpg"),
            ("D:\\Projects\\data\\img8.png", "D:\\Projects\\data\\img8.png"),
        ]
        for source, expected_relative in test_cases:
            with self.subTest(source=source):
                expected_path = backup_root / Path(expected_relative)
                self.assertEqual(utils.build_backup_path(source, backup_root), expected_path)

    def test_edge_cases_and_errors(self) -> None:
        backup_root = Path("backup")

        # Empty source path
        with self.subTest(source=""):
            with self.assertRaises(ValueError):
                utils.build_backup_path("", backup_root)

        # Source path is just '.'
        with self.subTest(source="."):
            with self.assertRaises(ValueError):
                utils.build_backup_path(".", backup_root)

        # Source path is an empty string part (e.g., from `//` in `Path` creation)
        with self.subTest(source="data//file.txt"):
            expected_path = backup_root / Path("data/file.txt")
            self.assertEqual(utils.build_backup_path("data//file.txt", backup_root), expected_path)

        # Source path is an absolute root
        if os.name == "posix":
            with self.subTest(source="/"):
                with self.assertRaises(ValueError):
                    utils.build_backup_path("/", backup_root)

            with self.subTest(source="/mnt/"):
                expected_path = backup_root / Path("mnt")
                self.assertEqual(utils.build_backup_path("/mnt/", backup_root), expected_path)
