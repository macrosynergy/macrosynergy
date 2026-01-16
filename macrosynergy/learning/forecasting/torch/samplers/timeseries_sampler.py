
import torch
import torch.nn as nn

from torch.utils.data import Sampler

import numbers

class TimeSeriesSampler(Sampler):
    """
    Batch sampler for datasets indexed by time, to ensure that batches are comprised of 
    samples from contiguous time periods.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The PyTorch dataset to sample from.
    batch_size : int
        Number of samples per batch.
    shuffle : bool, optional
        Whether to shuffle the order of batches. Default is True.
    aggregate_last : bool, optional
        Whether to aggregate the last batch with the previous one if it has length 
        smaller than batch_size. Default is True.
    drop_last : bool, optional
        Whether to drop the last batch if it has length smaller than batch_size.
        Default is False.
    """
    def __init__(self, dataset, batch_size, shuffle = True, aggregate_last = True, drop_last = False):
        # Checks
        self._check_init_params(
            dataset,
            batch_size,
            shuffle,
            aggregate_last,
            drop_last,
        )

        # Attributes
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aggregate_last = aggregate_last
        self.drop_last = drop_last
        
        self.dataset_size = len(dataset)

        # Determine batches
        self.batches = self._create_batches(
            self.batch_size,
            self.dataset_size,
            self.aggregate_last,
            self.drop_last,
        )

    def _create_batches(self, batch_size, dataset_size, aggregate_last, drop_last):
        """ Create list of batches """
        batches = [
            list(range(start, min(start + batch_size, dataset_size)))
            for start in range(0, dataset_size, batch_size)
        ]
        if aggregate_last:
            if len(batches) > 1 and len(batches[-1]) < batch_size:
                batches[-2].extend(batches[-1])
                batches = batches[:-1]

        if drop_last:
            if len(batches) > 1 and len(batches[-1]) < batch_size:
                    batches = batches[:-1]
        
        return batches

    def __iter__(self):
        """
        Generator for batch indices.
        """
        if self.shuffle:
            batch_indices = torch.randperm(len(self.batches)).tolist()
        else:
            batch_indices = range(len(self.batches))
            
        for idx in batch_indices:
            yield self.batches[idx]

    def __len__(self):
        """ Returns number of batches """
        return len(self.batches)
    
    def _check_init_params(
        self,
        dataset,
        batch_size,
        shuffle,
        aggregate_last,
        drop_last,
    ):
        # dataset
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise TypeError("dataset must be a torch.utils.data.Dataset instance.")
        # batch_size
        if not isinstance(batch_size, numbers.Integral):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        # shuffle
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean.")
        # aggregate_last
        if not isinstance(aggregate_last, bool):
            raise TypeError("aggregate_last must be a boolean.")
        # drop_last
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a boolean.")
        if aggregate_last and drop_last:
            raise ValueError("aggregate_last and drop_last cannot both be True.")