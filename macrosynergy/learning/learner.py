import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm.auto import tqdm

class PanelBaseLearner:
    """
    Base class for a general learning process to iteratively perform model and 
    hyperparameter selection over a panel. The selected models and hyperparameters
    are stored, along with cross-validation/validation scores. If the underlying model 
    is linear, the coefficients are also stored.
    """
    def __init__(self, X, y, blacklist = None):
        self.X = X
        self.y = y
        self.blacklist = blacklist 

    def run(
        self,
        name,
        outer_splitter,
        inner_splitters, # List of splitters for hyperparameter tuning
        models,
        hyperparameters,
        scorers, # List or string of scorers, not metrics
        search_type = "grid", # Should accept grid, random or bayes
        n_iter = 100, # Number of iterations for random or bayes
        splits_function = None, # Brainstorm this
        use_variance_correction = False, 
        n_jobs_outer = -1,
        n_jobs_inner = 1,
    ):
        train_test_splits = list(outer_splitter.split(self.X, self.y))

        # Return nested dictionary with results
        optim_results = Parallel(n_jobs=n_jobs_outer)(
            delayed(self._worker)(
                name = name,
                train_idx = train_idx,
                test_idx = test_idx,
                inner_splitters = inner_splitters,
                models = models,
                hyperparameters = hyperparameters,
                scorers = scorers,
                use_variance_correction = use_variance_correction,
                search_type = search_type,
                n_iter = n_iter,
                splits_function = splits_function,
                n_jobs_inner = n_jobs_inner,
            )
            for idx, (train_idx, test_idx) in tqdm(enumerate(train_test_splits), total=len(train_test_splits))
        )

        return optim_results



    