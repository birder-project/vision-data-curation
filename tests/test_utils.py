import os
import tempfile
import unittest

import torch

from vdc import utils


class TestInferenceCSVDataset(unittest.TestCase):
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
        dataset = utils.InferenceCSVDataset(self.csv_file_path_1)
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
        dataset = utils.InferenceCSVDataset(file_paths=self.csv_file_path_1, columns_to_drop=["col2"])
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
        dataset = utils.InferenceCSVDataset(file_paths=all_file_paths)

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
        dataset = utils.InferenceCSVDataset(
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
        dataset = utils.InferenceCSVDataset(
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
