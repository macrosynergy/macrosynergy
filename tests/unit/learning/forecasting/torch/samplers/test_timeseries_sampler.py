import unittest 

import torch
import torch.nn as nn 

from macrosynergy.learning.forecasting.torch.samplers.timeseries_sampler import TimeSeriesSampler

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

    def test_valid_init(self):
        pass