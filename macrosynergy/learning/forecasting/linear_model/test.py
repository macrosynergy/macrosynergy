from macrosynergy.management.simulate import make_qdf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import copy

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

dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
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

# Define the Encoder-Decoder model
class EncoderDecoder(nn.Module):
    def __init__(self, init_dim, hidden_dim, valid_cids = None):
        super().__init__()
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.valid_cids = valid_cids

        self.encoder = nn.Linear(
            in_features=self.init_dim,
            out_features=self.hidden_dim,
            bias = False,
        )
        self.decoder = nn.ModuleDict({
            cid : nn.Linear(
                in_features = self.hidden_dim,
                out_features = self.init_dim,
                bias = False
            )
            for cid in self.valid_cids
        })

    def forward(self, x, cid = None):
        x_embedding = self.encoder(x)
        if cid == None:
            # return the average over cids
            outputs = [
                self.decoder[cid](x_embedding)
                for cid in self.valid_cids
            ]
            x_reconstr = torch.mean(torch.stack(outputs,dim=0),dim = 0)
        else:
            x_reconstr = self.decoder[cid](x_embedding)
        
        return x_reconstr
    
    @torch.no_grad()
    def embeddings(self, x):
        return self.encoder(x)


from sklearn.base import BaseEstimator, TransformerMixin

class TorchEmbedding(BaseEstimator, TransformerMixin):
    def create_train_valid_splits_(self, X, y):
        real_dates = sorted(X.index.get_level_values(1).unique())
        train_dates = real_dates[:int(self.pct_train * len(real_dates))]
        valid_dates = real_dates[int(self.pct_train * len(real_dates)):]

        # Create training and validation sets
        X_train, y_train = X[X.index.get_level_values(1).isin(train_dates)], y[y.index.get_level_values(1).isin(train_dates)]
        X_valid, y_valid = X[X.index.get_level_values(1).isin(valid_dates)], y[y.index.get_level_values(1).isin(valid_dates)]

        # Get training and validation cross-sections
        train_cids = sorted(X_train.index.get_level_values(0).unique())
        valid_cids = sorted(X_valid.index.get_level_values(0).unique())

        # Create sample dictionaries for each training and validation cross-section.
        train_cids_samples, valid_cids_samples = self.get_cid_indices_(X_train, X_valid, train_cids, valid_cids)
        
        return (
            X_train,
            y_train,
            X_valid,
            y_valid,
            train_cids,
            valid_cids,
            train_cids_samples,
            valid_cids_samples
        )
    
    def get_cid_indices_(self, X_train, X_valid, train_cids, valid_cids):
        train_cids_samples = {
            cid : torch.Tensor(np.where(X_train.index.get_level_values(0) == cid)[0]).int()
            for cid in train_cids
        }
        valid_cids_samples = {
            cid : torch.Tensor(np.where(X_valid.index.get_level_values(0) == cid)[0]).int()
            for cid in valid_cids
        }
        return train_cids_samples, valid_cids_samples
    
    def fit_(
        self,
        model,
        lr,
        epochs,
        decoder_sparsity,
        X_train,
        y_train,
        X_valid,
        y_valid,
        train_cids,
        valid_cids,
        train_cids_samples,
        valid_cids_samples,
        patience,
        verbose,
    ):
        # Set up optimizer and loss
        optim = torch.optim.Adam(
            params = model.parameters(),
            lr = lr,
            weight_decay = 0,
        )
        loss_func = nn.MSELoss()

        # Set up training and validation statistic lists
        train_losses = []
        valid_losses = []

        # Set up early stopping statistics
        counter = 0
        best_state_dict  = None
        best_valid_loss = np.inf
        best_train_loss = np.inf

        for epoch in range(epochs):
            model.train()
            for cid in train_cids:
                optim.zero_grad()
                # Get country batch
                indices = train_cids_samples[cid]
                X_batch = X_train[indices]
                # Forward pass
                if cid in model.valid_cids:
                    X_pred = model(X_batch, cid = cid)
                else:
                    X_pred = model(X_batch)
                # Evaluate loss
                loss = loss_func(X_batch, X_pred) + decoder_sparsity * self.decoder_shrinkage(model.decoder)
                # Backward pass
                loss.backward()
                optim.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                # Training loss 
                train_loss_epoch = 0
                for cid in train_cids:
                    # Get country batch
                    indices = train_cids_samples[cid]
                    X_batch = X_train[indices]
                    # Forward pass
                    if cid in model.valid_cids:
                        X_pred = model(X_batch, cid = cid)
                    else:
                        X_pred = model(X_batch)
                    # Evaluate cross-section loss
                    loss = loss_func(X_batch, X_pred)
                    train_loss_epoch += loss.item()
                train_losses.append(train_loss_epoch / len(train_cids))

                # Validation loss
                valid_loss_epoch = 0
                for cid in valid_cids:
                    # Get country batch
                    indices = valid_cids_samples[cid]
                    X_batch = X_valid[indices]
                    # Forward pass
                    if cid in model.valid_cids:
                        X_pred = model(X_batch, cid = cid)
                    else:
                        X_pred = model(X_batch)
                    # Evaluate cross-section loss
                    loss = loss_func(X_batch, X_pred)
                    valid_loss_epoch += loss.item()
                valid_losses.append(valid_loss_epoch / len(valid_cids))

            # Print progress
            if verbose > 0 and (epoch % 5 == 0):
                print(f"Epoch {epoch + 1}: training loss = {train_losses[-1]}, validation loss = {valid_losses[-1]}")

            # Early stopping
            if valid_losses[-1] < best_valid_loss:
                counter = 0
                best_train_loss = train_losses[-1]
                best_valid_loss = valid_losses[-1]
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    break 

        return best_state_dict, best_train_loss, best_valid_loss
    
    def decoder_shrinkage(self, decoder):
        l1_penalty = sum(
            p.abs().sum() for p in decoder.parameters()
        )
        return l1_penalty

class FactorLearning(TorchEmbedding):
    def __init__(
            self,
            hidden_dim = 3,
            lr = 4e-3,
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
        self.model = EncoderDecoder(
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
    model = FactorLearning(
        hidden_dim=2,
        lr = 1e-3,
        decoder_sparsity=10,
        verbose = 1
    ).fit(X, y)
    # Print embeddings
    embeddings = model.transform(X)
    print(embeddings)
