import torch 
import torch.nn as nn

import numpy as np 
import pandas as pd 

from macrosynergy.learning.forecasting.torch import TorchEmbedding, LinearEncoderDecoder

class LinearFactorLearning(TorchEmbedding):
    def __init__(
        self,
        hidden_dim = 3,
        lr = 1e-3,
        decoder_sparsity = 0,
        epochs = 10000,
        pct_train = 0.7,
        patience = 10,
        min_xs_samples = 36,
        verbose = 0,
    ):
        # Model attributes
        self.model = None
        self.hidden_dim = hidden_dim

        # Training dynamics
        self.lr = lr
        self.epochs = epochs
        self.decoder_sparsity = decoder_sparsity

        # Early stopping dynamics
        self.pct_train = pct_train 
        self.patience = patience

        # Other attributes
        self.min_xs_samples = min_xs_samples
        self.verbose = verbose

    def fit(self, X, y):
        # Split X and y into training and validation sets 
        # for early stopping. Create dictionaries of indices for 
        # each asset in training and validation
        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            train_cids,
            valid_cids,
            train_cids_samples,
            valid_cids_samples,
        ) = self.create_train_valid_splits_(X, y)

        # Convert to numpy 
        X_train_np, y_train_np = X_train.values, np.reshape(y_train.values, (-1,1))
        X_valid_np, y_valid_np = X_valid.values, np.reshape(y_valid.values, (-1,1))

        # Convert to tensor 
        X_train_torch, y_train_torch = torch.Tensor(X_train_np), torch.Tensor(y_train_np)
        X_valid_torch, y_valid_torch = torch.Tensor(X_valid_np), torch.Tensor(y_valid_np)

        # Define a model with linear encoder and country-specific decoders.
        # Only countries with more samples than min_xs_samples have heads
        torch.manual_seed(42)
        self.model = LinearEncoderDecoder(
            init_dim = X_train.shape[1],
            hidden_dim = self.hidden_dim,
            valid_cids = train_cids,
        )
    
        best_state_dict, best_train_loss, best_valid_loss = self.fit_(
            model = self.model,
            # Hyperparameters
            lr = self.lr,
            epochs = self.epochs,
            decoder_sparsity = self.decoder_sparsity,
            # Data
            X_train = X_train_torch, 
            y_train = y_train_torch,
            X_valid = X_valid_torch,
            y_valid = y_valid_torch,
            train_cids = train_cids,
            valid_cids = valid_cids,
            train_cids_samples = train_cids_samples,
            valid_cids_samples = valid_cids_samples,
            patience = self.patience,
            verbose = self.verbose,
        )

        self.model.load_state_dict(best_state_dict)

        return self

    def transform(self, X):
        self.model.eval()

        # Convert to numpy 
        X_numpy = X.values 

        # Convert to tensor
        X_tensor = torch.Tensor(X_numpy)

        # Get embeddings
        X_embeddings = self.model.embeddings(X_tensor)

        # Wrap in dataframe
        X_embeddings_pandas = pd.DataFrame(
            data = X_embeddings,
            index = X.index,
            columns = [f"Factor{i+1}" for i in range(self.hidden_dim)]
        )

        return X_embeddings_pandas
    
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    from macrosynergy.management.simulate import make_qdf

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2012-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2012-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2012-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2012-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2015-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75, seed=42)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {
        "GBP": (
            pd.Timestamp(year=2009, month=1, day=1),
            pd.Timestamp(year=2012, month=6, day=30),
        ),
        "CAD": (
            pd.Timestamp(year=2015, month=1, day=1),
            pd.Timestamp(year=2016, month=1, day=1),
        ),
    }

    from macrosynergy.learning import SignalOptimizer
    so = SignalOptimizer(
        df=dfd,
        xcats=["CRY", "GROWTH", "INFL", "XR"],
        cids=cids,
        blacklist=black,
        drop_nas = True
    )

    X = so.X.copy()
    y = so.y.copy()

    model = LinearFactorLearning(
        hidden_dim=2,
        lr = 1e-3,
        decoder_sparsity=10,
        verbose = 1
    ).fit(X, y)
    # Print embeddings
    embeddings = model.transform(X)
    print(embeddings)
