import numpy as np

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from typing import Optional

from macrosynergy.learning.forecasting.torch.samplers.timeseries_sampler import TimeSeriesSampler
from macrosynergy.learning.forecasting.torch.models.mlps import MultiLayerPerceptron

class MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # Neural network structure
        n_latent = 256,
        fit_encoder_intercept = False,
        fit_head_intercept = False, 
        encoder_activation = "relu",
        head_activation = "identity",
        # Neural network training dynamics
        loss_func = torch.nn.MSELoss(),
        optimizer: str = "AdamW",
        scheduler: Optional[str] = None, # TODO: consider default of one cycle LR
        batch_size = 32,
        learning_rate = 3e-4,
        weight_decay = 1e-4,
        reg_turnover = 0,
        use_ts_sampler = True,
        aggregate_last = True,
        drop_last = False,
        epochs = 10000,
        patience = 1000,
        train_pct = 0.7,
        x_scaler = StandardScaler(with_mean=False), # Can be made optional due to panel z scoring
        y_scaler = StandardScaler(with_mean=False), # Can be made optional due to vol targeted returns and sharpe loss
        # Other stuff 
        verbose = False,
        random_state = 42,
        inverse_transform_preds = False
    ):
        self.n_latent = n_latent
        self.fit_encoder_intercept = fit_encoder_intercept
        self.fit_head_intercept = fit_head_intercept
        self.encoder_activation = encoder_activation
        self.head_activation = head_activation
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        #self.reg_turnover = reg_turnover
        self.use_ts_sampler = use_ts_sampler
        self.aggregate_last = aggregate_last
        self.drop_last = drop_last
        self.epochs = epochs
        self.patience = patience
        self.train_pct = train_pct
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.verbose = verbose
        self.random_state = random_state
        self.inverse_transform_preds = inverse_transform_preds

        self.model = None

    def fit(self, X, y, sample_weight=None):
        torch.manual_seed(self.random_state)

        # Initialize model 
        model = self.initialize_model(
            n_inputs = X.shape[1],
            n_latent = self.n_latent,
            n_outputs = y.shape[1],
            encoder_activation = self.encoder_activation,
            head_activation = self.head_activation,
            fit_encoder_intercept = self.fit_encoder_intercept,
            fit_head_intercept = self.fit_head_intercept
        )

        # Create training and validation splits 
        X_train, X_valid, y_train, y_valid = self.create_train_valid_splits(X, y, self.train_pct)

        # Scale training and validation splits 
        X_train_s, X_valid_s, y_train_s, y_valid_s = self.scale_data(X_train, X_valid, y_train, y_valid, self.x_scaler, self.y_scaler)

        # Make tensor datasets 
        train_dataset, valid_dataset = self.make_tensor_datasets(X_train_s, X_valid_s, y_train_s, y_valid_s)

        # Make torch dataloaders
        train_loader, valid_loader = self.make_dataloaders(train_dataset, valid_dataset, self.batch_size, self.use_ts_sampler, self.aggregate_last, self.drop_last)

        # Set up optimizer 
        optimizer = self.make_optimizer(self.model, self.optimizer, self.learning_rate, self.weight_decay)

        # Set up scheduler
        if self.scheduler is not None:
            scheduler = self.make_scheduler(optimizer, self.scheduler, self.epochs, len(train_loader))
        
        # Train model
        self.model = self.train_model(
            model = model, 
            train_loader = train_loader, 
            valid_loader = valid_loader, 
            optimizer = optimizer, 
            scheduler = scheduler,
            loss_func = self.loss_func, 
            #reg_turnover = self.reg_turnover, 
            patience = self.patience, 
            verbose = self.verbose
        )

        return self

    def initialize_model(
        self,
        n_inputs,
        n_latent,
        n_outputs,
        encoder_activation,
        head_activation,
        fit_encoder_intercept,
        fit_head_intercept,
    ):
        model = MultiLayerPerceptron(
            n_inputs=n_inputs,
            n_latent=n_latent,
            n_outputs=n_outputs,
            encoder_activation=encoder_activation,
            head_activation=head_activation,
            fit_encoder_intercept=fit_encoder_intercept,
            fit_head_intercept=fit_head_intercept,
        )

        return model
    
    def create_train_valid_splits(self, X, y, train_pct):
        dates = sorted(X.index.get_level_values(1).unique())
        cut = int(train_pct * len(dates))
        train_dates, valid_dates = dates[:cut], dates[cut:] # TODO: upfront check this doesn't create empty splits

        X_train = X[X.index.get_level_values(1).isin(train_dates)]
        y_train = y[y.index.get_level_values(1).isin(train_dates)]
        X_valid = X[X.index.get_level_values(1).isin(valid_dates)]
        y_valid = y[y.index.get_level_values(1).isin(valid_dates)]

        return X_train, X_valid, y_train, y_valid
    
    def scale_data(
        self,
        X_train,
        X_valid, 
        y_train,
        y_valid,
        x_scaler,
        y_scaler,
    ):
        X_train_s = x_scaler.fit_transform(X_train)
        X_valid_s = x_scaler.transform(X_valid)
        y_train_s = y_scaler.fit_transform(y_train) # TODO: ensure y is 2d for this to work
        y_valid_s = y_scaler.transform(y_valid)

        return X_train_s, X_valid_s, y_train_s, y_valid_s
    
    def make_tensor_datasets(
        self,
        X_train_s,
        X_valid_s,
        y_train_s,
        y_valid_s,
    ):
        train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train_s), torch.Tensor(y_train_s))
        valid_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_valid_s), torch.Tensor(y_valid_s))

        return train_dataset, valid_dataset

    def make_dataloaders(
        self,
        train_dataset,
        valid_dataset,
        batch_size,
        use_ts_sampler,
        aggregate_last,
        drop_last,
    ):
        """
        TODO: run through aggregate last and drop last logic 
        """
        if not use_ts_sampler:
            train_loader = torch.utils.data.DataLoader(
                dataset = train_dataset,
                batch_size = self.batch_size,
                shuffle = True,
            )
            train_loader_eval = torch.utils.data.DataLoader(
                dataset = train_dataset,
                batch_size = self.batch_size,
                shuffle = False,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset = train_dataset,
                batch_sampler = TimeSeriesSampler(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle = True,
                )
            )
            train_loader_eval = torch.utils.data.DataLoader(
                dataset = train_dataset,
                batch_sampler = TimeSeriesSampler(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle = False,
                )
            )

        valid_loader = torch.utils.data.DataLoader(
            dataset = valid_dataset,
            batch_sampler=TimeSeriesSampler(
                dataset=valid_dataset, batch_size=self.batch_size, shuffle=False
            ),
        )

        return train_loader, valid_loader
    
    def make_optimizer(self, model, optimizer_name, learning_rate, weight_decay):
        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            # TODO: add more optimizers later
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def make_scheduler(self, optimizer, scheduler_name, epochs, steps_per_epoch):
        if scheduler_name == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                anneal_strategy='cos',
            )
        else:
            # TODO: add more schedulers later
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        return scheduler
    
    def train_model(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        loss_func,
        #reg_turnover,
        patience,
        verbose,
    ):
        best_score = np.inf
        best_state = None 
        counter = 0

        for epoch in range(self.epochs):
            model.train()
            for X_i, y_i in train_loader:
                self._fit_one_batch(
                    model = model,
                    X_i = X_i,
                    y_i = y_i,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    loss_func = loss_func,
                    #reg_turnover = reg_turnover
                )
            
            train_loss = self._eval_loss(model, train_loader, loss_func)
            valid_loss = self._eval_loss(model, valid_loader, loss_func)

            best_score, best_state, counter = self.update_es_stats(
                train_loss, valid_loss, best_score, best_state, counter, patience
            )

            if verbose and (epoch % 5 == 0):
                print(
                    f"Epoch {epoch+1}: train_loss={train_loss:.6g}, valid_loss={valid_loss:.6g}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            # TODO: handle this case later
            pass

        return model


    def predict(self, X):
        # Scale data 
        X_s = self.x_scaler.transform(X)

        # Switch to evaluation mode 
        self.model.eval()
        with torch.no_grad():
            # Convert to tensor and pass through network
            X_s_torch = torch.Tensor(X_s)
            preds = self.model(X_s_torch).numpy()

            # Inverse scale predictions
            if self.inverse_transform_preds:
                preds = self.y_scaler.inverse_transform(preds)

        return preds

if __name__ == "__main__":
    print("foo")