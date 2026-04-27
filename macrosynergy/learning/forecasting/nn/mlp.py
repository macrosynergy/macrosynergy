import numpy as np

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from typing import Optional
import inspect

from macrosynergy.learning.forecasting.torch.samplers.timeseries_sampler import TimeSeriesSampler
from macrosynergy.learning.forecasting.torch.models.mlps import MultiLayerPerceptron

from copy import deepcopy

import numbers

class MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # Neural network structure
        # TODO: alternatively could just take in a model object and skip this whole section
        # But this would need some careful methods for cross-validating model hparams
        n_latent = 32,
        fit_encoder_intercept = True,
        fit_head_intercept = True,
        encoder_activation = "relu",
        head_activation = "identity",
        # Neural network training dynamics
        loss_func = torch.nn.MSELoss(),
        optimizer: str = "AdamW", # TODO: option for multiple optimizers as well as lars and ability to pass in a custom optimizer. 
        scheduler: Optional[str] = None, # TODO: options for other schedulers, option for custom scheduler object
        batch_size = 32,
        learning_rate = 3e-4,
        weight_decay = 1e-4,
        reg_turnover = 0,
        use_ts_sampler = True, # TODO: turn this into an optional sampler object
        aggregate_last = True,
        drop_last = False,
        epochs = 10000, # NOTE: when a scheduler is used, the epochs default is way too high unless the patience is high
        patience = 1000,
        train_pct = 0.7,
        x_scaler = StandardScaler(with_mean=False),
        y_scaler = StandardScaler(with_mean=False), 
        # Other stuff 
        verbose = False,
        random_state = 42,
        inverse_transform_preds = False
    ):
        # Checks 
        self._check_init_params(
            n_latent,
            fit_encoder_intercept,
            fit_head_intercept,
            encoder_activation,
            head_activation,
            loss_func,
            optimizer,
            scheduler,
            batch_size,
            learning_rate,
            weight_decay,
            reg_turnover,
            use_ts_sampler,
            aggregate_last,
            drop_last,
            epochs,
            patience,
            train_pct,
            x_scaler,
            y_scaler,
            verbose,
            random_state,
            inverse_transform_preds
        )

        # Attributes
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
        self.reg_turnover = reg_turnover

        self.use_ts_sampler = use_ts_sampler
        self.aggregate_last = aggregate_last
        self.drop_last = drop_last
        self.epochs = epochs
        self.patience = patience
        self.train_pct = train_pct
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.verbose = verbose
        self.random_states = [random_state] if not isinstance(random_state, list) else random_state
        self.inverse_transform_preds = inverse_transform_preds

        self.models = []

    def fit(self, X, y, sample_weight=None):
        # Additional checks 
        sample_weight_strategy = self._check_fit_params(X, y, sample_weight)

        # Create training and validation splits
        X_train, X_valid, y_train, y_valid = self.create_train_valid_splits(X, y, self.train_pct)

        # Scale training and validation splits
        X_train_s, X_valid_s, y_train_s, y_valid_s = self.scale_data(X_train, X_valid, y_train, y_valid, self.x_scaler, self.y_scaler)

        # Make tensor datasets
        train_dataset, valid_dataset = self.make_tensor_datasets(X_train_s, X_valid_s, y_train_s, y_valid_s, sample_weight)

        # Iterate through random states
        for random_state in self.random_states:
            torch.manual_seed(random_state)

            # Make torch dataloaders
            train_loader, train_loader_eval, valid_loader = self.make_dataloaders(train_dataset, valid_dataset, self.batch_size, self.use_ts_sampler, self.aggregate_last, self.drop_last)

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

            # Set up optimizer 
            optimizer = self.make_optimizer(model, self.optimizer, self.learning_rate, self.weight_decay)

            # Set up scheduler
            if self.scheduler is not None:
                scheduler = self.make_scheduler(optimizer, self.scheduler, self.epochs, len(train_loader))
            else:
                scheduler = None
        
            # Train model
            trained_model = self.train_model(
                model = model, 
                train_loader = train_loader,
                train_loader_eval = train_loader_eval,
                valid_loader = valid_loader, 
                optimizer = optimizer, 
                scheduler = scheduler,
                loss_func = self.loss_func,
                sample_weight = sample_weight,
                sample_weight_strategy = sample_weight_strategy,
                #reg_turnover = self.reg_turnover, 
                patience = self.patience, 
                verbose = self.verbose
            )
            self.models.append(trained_model)

        return self
    
    def predict(self, X):
        # Scale data 
        X_s = self.x_scaler.transform(X)
        model_preds = []

        with torch.no_grad():
            # Convert to tensor and pass through each network
            X_s_torch = torch.Tensor(X_s)
            for model in self.models:
                model.eval()
                preds = model(X_s_torch).numpy()

                # Inverse scale predictions
                if self.inverse_transform_preds:
                    preds = self.y_scaler.inverse_transform(preds)
                model_preds.append(preds)

        return np.mean(np.concatenate(model_preds, axis=1), axis = 1)

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
        # Scale independent variables
        if x_scaler:
            x_scaler.fit(X_train)
            X_train_s = x_scaler.transform(X_train)
            X_valid_s = x_scaler.transform(X_valid)
        else:
            X_train_s = X_train.values
            X_valid_s = X_valid.values


        # Scale dependent variables
        # TODO: ensure ys are 2d for this to work
        if y_scaler:
            y_scaler.fit(y_train)
            y_train_s = y_scaler.transform(y_train) 
            y_valid_s = y_scaler.transform(y_valid)
        else:
            y_train_s = y_train.values
            y_valid_s = y_valid.values

        return X_train_s, X_valid_s, y_train_s, y_valid_s
    
    def make_tensor_datasets(
        self,
        X_train_s,
        X_valid_s,
        y_train_s,
        y_valid_s,
        sample_weight,
    ):
        if sample_weight is not None: 
            train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train_s), torch.Tensor(y_train_s), torch.Tensor(sample_weight))
        else:
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
                drop_last = drop_last
            )
            train_loader_eval = torch.utils.data.DataLoader(
                dataset = train_dataset,
                batch_size = self.batch_size,
                shuffle = False,
                drop_last = False,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset = train_dataset,
                batch_sampler = TimeSeriesSampler(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle = True,
                    aggregate_last = aggregate_last,
                    drop_last = drop_last
                )
            )
            train_loader_eval = torch.utils.data.DataLoader(
                dataset = train_dataset,
                batch_sampler = TimeSeriesSampler(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle = False,
                    aggregate_last = aggregate_last,
                    drop_last = False
                )
            )

        valid_loader = torch.utils.data.DataLoader(
            dataset = valid_dataset,
            batch_sampler=TimeSeriesSampler(
                dataset=valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                aggregate_last = aggregate_last,
                drop_last = False
            ),
        )

        return train_loader, train_loader_eval, valid_loader
    
    def make_optimizer(self, model, optimizer_name, learning_rate, weight_decay):
        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "SGD+mom":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum = 0.9)
        else:
            # TODO: add LARS for large batch SGD training
            # TODO: add ability to pass in an optimizer class inheriting from torch.optim.Optimizer
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
        train_loader_eval,
        valid_loader,
        optimizer,
        scheduler,
        loss_func,
        sample_weight,
        sample_weight_strategy,
        #reg_turnover,
        patience,
        verbose,
    ):
        best_score = np.inf
        best_state = None 
        counter = 0

        for epoch in range(self.epochs):
            model.train()
            if sample_weight:
                for X_i, y_i, sw_i in train_loader:
                    model = self._fit_one_batch(
                        model = model,
                        X_i = X_i,
                        y_i = y_i,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        loss_func = loss_func,
                        sample_weight = sw_i,
                        sample_weight_strategy = sample_weight_strategy,
                        #reg_turnover = reg_turnover
                    )
            else:
                for X_i, y_i in train_loader:
                    model = self._fit_one_batch(
                        model = model,
                        X_i = X_i,
                        y_i = y_i,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        loss_func = loss_func,
                        sample_weight = None,
                        sample_weight_strategy = sample_weight_strategy,
                        #reg_turnover = reg_turnover
                    )
            
            train_loss = self._eval_loss(model, train_loader_eval, loss_func)
            valid_loss = self._eval_loss(model, valid_loader, loss_func)

            best_score, best_state, counter = self.update_es_stats(
                model, train_loss, valid_loss, best_score, best_state, counter, patience
            )

            if counter >= patience:
                break

            if verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}, Best Valid Loss = {best_score:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            # TODO: handle this case later
            pass

        return model

    def _fit_one_batch(
        self,
        model,
        X_i,
        y_i,
        optimizer,
        scheduler,
        loss_func,
        sample_weight,
        sample_weight_strategy,
        # reg_turnover
    ):
        optimizer.zero_grad()
        preds = model(X_i)
        if not sample_weight:
            loss = loss_func(preds, y_i)
        elif sample_weight_strategy == "native":
            loss = loss_func(preds, y_i, sample_weight)
        elif sample_weight_strategy == "reduction_none":
            loss = loss_func(preds, y_i) * sample_weight
            loss = loss.mean()

        # if reg_turnover > 0:
        #     pweight_levels = preds[1:] - preds[:-1]
        #     pweight_l1 = torch.mean(torch.abs(pweight_levels))
        #     loss = loss + reg_turnover * pweight_l1

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        return model
    
    def _eval_loss(self, model, loader, loss_func):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_i, y_i in loader:
                preds = model(X_i)
                total_loss += loss_func(preds, y_i).item()
            avg_loss = total_loss / len(loader)

        return avg_loss    
    
    def update_es_stats(self, model, train_loss, valid_loss, best_score, best_state, counter, patience):
        if valid_loss < best_score:
            best_score = valid_loss
            best_state = deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            
        return best_score, best_state, counter
    
    def _check_init_params(
        self,
        n_latent,
        fit_encoder_intercept,
        fit_head_intercept,
        encoder_activation,
        head_activation,
        loss_func,
        optimizer,
        scheduler,
        batch_size,
        learning_rate,
        weight_decay,
        reg_turnover,
        use_ts_sampler,
        aggregate_last,
        drop_last,
        epochs,
        patience,
        train_pct,
        x_scaler,
        y_scaler,
        verbose,
        random_state,
        inverse_transform_preds
    ):
        # n_latent
        if not isinstance(n_latent, numbers.Integral):
            if not isinstance(n_latent, list):
                raise TypeError("n_latent must be either an integer or a list of integers.")
            if not all(isinstance(x, numbers.Integral) for x in n_latent):
                raise TypeError("When n_latent is a list, all elements must be integers.")
            if len(n_latent) <= 1:
                raise ValueError("When n_latent is a list, it must contain more than one element.")
            if not all(x >= 1 for x in n_latent):
                raise ValueError("When n_latent is a list, all elements must be at least 1.")
        else:
            if n_latent < 1:
                raise ValueError("When n_latent is an integer, it must be at least 1.")
            
        # fit_encoder_intercept
        if not isinstance(fit_encoder_intercept, bool):
            raise TypeError("fit_encoder_intercept must be a boolean.")
        
        # fit_head_intercept
        if not isinstance(fit_head_intercept, bool):
            raise TypeError("fit_head_intercept must be a boolean.")
        
        # encoder_activation
        if not isinstance(encoder_activation, str):
            raise TypeError("encoder_activation must be a string.")
        if encoder_activation not in {"tanh", "relu", "sigmoid"}:
            raise ValueError(
                "encoder_activation must be one of 'tanh', 'relu', or 'sigmoid'."
            )
        
        # head_activation
        if not isinstance(head_activation, str):
            raise TypeError("head_activation must be a string.")
        if head_activation not in {"tanh", "relu", "sigmoid", "identity"}:
            raise ValueError(
                "head_activation must be one of 'tanh', 'relu', 'sigmoid', or 'identity'."
            )
        
        # loss_func
        if not isinstance(loss_func, nn.Module):
            raise TypeError("loss_func must inherit from nn.Module.")
        
        # optimizer
        if not isinstance(optimizer, str):
            raise TypeError("optimizer must be a string.")
        if optimizer not in {"AdamW", "SGD", "SGD+mom"}:
            raise ValueError("optimizer must be one of 'AdamW', 'SGD', or 'SGD+mom'.")
        
        # scheduler
        if scheduler is not None:
            if not isinstance(scheduler, str):
                raise TypeError("scheduler must be a string or None.")
            if scheduler not in {"OneCycleLR"}:
                raise ValueError("scheduler must be one of 'OneCycleLR' or None.")
            
        # batch_size
        if not isinstance(batch_size, numbers.Integral):
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        
        # learning_rate
        if not isinstance(learning_rate, numbers.Real):
            raise TypeError("learning_rate must be a real number.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        # weight_decay
        if not isinstance(weight_decay, numbers.Real):
            raise TypeError("weight_decay must be a real number.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        
        # reg_turnover
        if not isinstance(reg_turnover, numbers.Real):
            raise TypeError("reg_turnover must be a real number.")
        if reg_turnover < 0:
            raise ValueError("reg_turnover must be non-negative.")
        
        # use_ts_sampler
        if not isinstance(use_ts_sampler, bool):
            raise TypeError("use_ts_sampler must be a boolean.")
        
        # aggregate_last
        if not isinstance(aggregate_last, bool):
            raise TypeError("aggregate_last must be a boolean.")
        
        # drop_last
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a boolean.")
        if aggregate_last and drop_last:
            raise ValueError("aggregate_last and drop_last cannot both be True.")
        
        # epochs
        if not isinstance(epochs, numbers.Integral):
            raise TypeError("epochs must be an integer.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        
        # patience
        if not isinstance(patience, numbers.Integral):
            raise TypeError("patience must be an integer.")
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        
        # train_pct
        if not isinstance(train_pct, numbers.Real):
            raise TypeError("train_pct must be a real number.")
        if not (0 < train_pct < 1):
            raise ValueError("train_pct must be between 0 and 1.")
        
        # x_scaler
        if x_scaler is not None:
            if not isinstance(x_scaler, StandardScaler):
                raise TypeError("x_scaler must be an instance of StandardScaler or None.")
        
        # y_scaler
        if y_scaler is not None:
            if not isinstance(y_scaler, StandardScaler):
                raise TypeError("y_scaler must be an instance of StandardScaler or None.")

        # verbose
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean.")
        
        # random_state
        if not isinstance(random_state, numbers.Integral):
            if not isinstance(random_state, list):
                raise TypeError("random_state must be either an integer or a list of integers.")
            else:
                if not all(isinstance(x, numbers.Integral) for x in random_state):
                    raise TypeError("When random_state is a list, all elements must be integers.")
                if not all(x >= 0 for x in random_state):
                    raise ValueError("When random_state is a list, all elements must be non-negative.")
                if len(random_state) <= 1:
                    raise ValueError("When random_state is a list, it must contain more than one element.")
        else:     
            if random_state < 0:
                raise ValueError("When random_state is an integer, it must be non-negative.")
        
        # inverse_transform_preds
        if not isinstance(inverse_transform_preds, bool):
            raise TypeError("inverse_transform_preds must be a boolean.")
        
    def _check_fit_params(self, X, y, sample_weight):
        # TODO: X and y checks 
        # sample_weight 
        if sample_weight is not None:
            if not isinstance(sample_weight, np.ndarray):
                raise TypeError("sample_weight must be a numpy array or None.")
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be a 1D array.")
            if len(sample_weight) != len(X):
                raise ValueError("Length of sample_weight must match number of samples in X.")
            
            # Check compatibility with loss function
            sig_forward = inspect.signature(self.loss_func.forward)
            sig_constructor = inspect.signature(self.loss_func.__init__)
            if "sample_weight" not in sig_forward.parameters:
                if "reduction" not in sig_constructor.parameters:
                    raise ValueError(
                        "Sample weights are not supported by the specified loss function. The loss function must either accept a `sample_weight` tensor in its forward method or have a `reduction` parameter in its constructor."
                    )
                else:
                    reduction = sig_constructor.parameters["reduction"]
                    if reduction.default == "none":
                        return "reduction_none"
                    else:
                        raise ValueError(
                            "When the loss function does not accept a `sample_weight` tensor in its forward method but has a `reduction` parameter in its constructor, the `reduction` must be set to 'none' to support sample weights."
                        )
            else:
                return "native" 
        else:
            return None

if __name__ == "__main__":
    from macrosynergy.learning import (
        SignalOptimizer,
    )
    from macrosynergy.management.simulate import make_qdf
    import pandas as pd
    import numpy as np

    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR1", "CRY", "GROWTH", "RATES", "XR2"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2012-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2012-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR1"] = ["2012-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2012-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2012-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["RATES"] = ["2010-01-01", "2020-12-31", 0, 1, 0.5, 0.5]
    df_xcats.loc["XR2"] = ["2015-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75, seed = 42)
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

    so = SignalOptimizer(
        df=dfd,
        xcats=["CRY", "GROWTH", "RATES", "XR1", "XR2"],
        cids=cids,
        blacklist=black,
        drop_nas=True,
        n_targets=2,
    )
    X = so.X.copy(deep=True)
    y = so.y.copy(deep=True)

    mlp = MLPRegressor(
        n_latent = 2, 
        fit_encoder_intercept = False,
        fit_head_intercept = True,
        encoder_activation = "tanh",
        head_activation="identity",
        loss_func=torch.nn.MSELoss(),
        optimizer = "AdamW",
        scheduler = None, 
        batch_size = 16,
        learning_rate = 3e-4, 
        weight_decay = 1e-4,
        reg_turnover = 0,
        use_ts_sampler = True,
        aggregate_last=True,
        drop_last=False,
        epochs = 10000,
        patience = 10, 
        train_pct = 0.7,
        x_scaler = StandardScaler(with_mean=False),
        y_scaler = StandardScaler(with_mean=False),
        verbose = True, 
        random_state = [42, 43],
        inverse_transform_preds = True
    ).fit(X,y)

    print(list(mlp.models[0].parameters()))
    preds = mlp.predict(X)
    print(preds)