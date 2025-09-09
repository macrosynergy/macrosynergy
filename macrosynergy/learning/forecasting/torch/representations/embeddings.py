import torch 
import torch.nn as nn 

import numpy as np 
import pandas as pd 

import copy

from sklearn.base import BaseEstimator, TransformerMixin

class LinearEncoderDecoder(nn.Module):
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