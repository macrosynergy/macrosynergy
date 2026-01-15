import unittest 

import torch
import torch.nn as nn 

from macrosynergy.learning.forecasting.torch.samplers.timeseries_sampler import TimeSeriesSampler

import itertools
from parameterized import parameterized

class TestTimeSeriesSampler(unittest.TestCase):
    @classmethod 
    def setUpClass(self):
        # Create a valid torch dataset for testing
        tensor = torch.randn(64,3)
        self.valid_dataset = torch.utils.data.TensorDataset(tensor)

    def test_types_init(self):
        """ Test correct input validation checks """
        """ dataset """
        with self.assertRaises(TypeError):
            TimeSeriesSampler(
                dataset = "invalid_dataset",
                batch_size = 4
            )
        with self.assertRaises(TypeError):
            TimeSeriesSampler(
                dataset = torch.randn(32,3),
                batch_size = 4
            )
        """ batch_size """
        with self.assertRaises(TypeError):
            TimeSeriesSampler(
                dataset = self.valid_dataset,
                batch_size = "invalid_batch_size"
            )
        with self.assertRaises(TypeError):
            TimeSeriesSampler(
                dataset = self.valid_dataset,
                batch_size = 3.2
            )
        with self.assertRaises(ValueError):
            TimeSeriesSampler(
                dataset = self.valid_dataset,
                batch_size = 0
            )
        with self.assertRaises(ValueError):
            TimeSeriesSampler(
                dataset = self.valid_dataset,
                batch_size = -7
            )
        """ shuffle """
        with self.assertRaises(TypeError):
            TimeSeriesSampler(
                dataset = self.valid_dataset,
                batch_size = 4,
                shuffle = "invalid_shuffle"
            )
        with self.assertRaises(TypeError):
            TimeSeriesSampler(
                dataset = self.valid_dataset,
                batch_size = 4,
                shuffle = 3
            )
        """ aggregate_last """
        with self.assertRaises(TypeError):
            TimeSeriesSampler(
                dataset = self.valid_dataset,
                batch_size = 4,
                aggregate_last = "invalid_aggregate_last"
            )
        with self.assertRaises(TypeError):
            TimeSeriesSampler(
                dataset = self.valid_dataset,
                batch_size = 4,
                aggregate_last = 3
            )
        """ drop_last """
        with self.assertRaises(TypeError):
            TimeSeriesSampler(
                dataset = self.valid_dataset,
                batch_size = 4,
                drop_last = "invalid_drop_last"
            )
        with self.assertRaises(TypeError):
            TimeSeriesSampler(
                dataset = self.valid_dataset,
                batch_size = 4,
                drop_last = 3
            )
        """ Test that drop_last and aggregate_last cannot both be True """
        with self.assertRaises(ValueError):
            TimeSeriesSampler(
                dataset = self.valid_dataset,
                batch_size = 4,
                drop_last = True,
                aggregate_last = True
            )

    @parameterized.expand(itertools.product([True, False], [True, False], [True, False]))
    def test_valid_init(self, shuffle, aggregate_last, drop_last):
        """ Test that valid initialization works """
        # Try with a batch size that divides dataset size evenly
        if not (aggregate_last and drop_last):
            # Check instantiation works
            try:
                sampler = TimeSeriesSampler(
                    dataset = self.valid_dataset,
                    batch_size = 4,
                    shuffle = shuffle,
                    aggregate_last = aggregate_last,
                    drop_last = drop_last,
                )
            except Exception as e:
                self.fail(f"TimeSeriesSampler with parameters shuffle = {shuffle}, aggregate_last = {aggregate_last}, drop_last = {drop_last} raised an exception unexpectedly: {e}")
            
            # Check attributes set correctly
            self.assertIsInstance(sampler, TimeSeriesSampler)
            self.assertIsInstance(sampler, torch.utils.data.Sampler)
            self.assertEqual(sampler.dataset, self.valid_dataset)
            self.assertEqual(sampler.batch_size, 4)
            self.assertEqual(sampler.shuffle, shuffle)
            self.assertEqual(sampler.aggregate_last, aggregate_last)
            self.assertEqual(sampler.drop_last, drop_last)
            self.assertEqual(sampler.dataset_size, len(self.valid_dataset))

            # Check length of the sampler is correct
            num_batches = len(sampler.batches)
            if drop_last or aggregate_last:
                expected_num_batches = len(self.valid_dataset) // 4
            else:
                expected_num_batches = len(self.valid_dataset) // 4 # 64 / 4 is a whole number

            self.assertEqual(num_batches, expected_num_batches)

        # Try with a batch size that doesn't divides the dataset size evenly
        if not (aggregate_last and drop_last):
            # Check instantiation works
            try:
                sampler = TimeSeriesSampler(
                    dataset = self.valid_dataset,
                    batch_size = 3,
                    shuffle = shuffle,
                    aggregate_last = aggregate_last,
                    drop_last = drop_last,
                )
            except Exception as e:
                self.fail(f"TimeSeriesSampler with parameters shuffle = {shuffle}, aggregate_last = {aggregate_last}, drop_last = {drop_last} raised an exception unexpectedly: {e}")
            
            # Check attributes set correctly
            self.assertIsInstance(sampler, TimeSeriesSampler)
            self.assertIsInstance(sampler, torch.utils.data.Sampler)
            self.assertEqual(sampler.dataset, self.valid_dataset)
            self.assertEqual(sampler.batch_size, 3)
            self.assertEqual(sampler.shuffle, shuffle)
            self.assertEqual(sampler.aggregate_last, aggregate_last)
            self.assertEqual(sampler.drop_last, drop_last)
            self.assertEqual(sampler.dataset_size, len(self.valid_dataset))

            # Check length of the sampler is correct
            num_batches = len(sampler.batches)
            if drop_last or aggregate_last:
                expected_num_batches = len(self.valid_dataset) // 3
            else:
                expected_num_batches = len(self.valid_dataset) // 3 + 1 # 64 / 4 is a whole number

            self.assertEqual(num_batches, expected_num_batches)

    