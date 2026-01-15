from sklearn.preprocessing import StandardScaler


from sklearn.preprocessing import StandardScaler
from copy import deepcopy


class MultiHeadReturnRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # Neural network structure hyperparameters
        n_latent,
        # Neural network training hyperparameters
        weight_decay=1e-4,
        reg_turnover=0,
        batch_size=16,
        learning_rate=3e-4,
        use_ts_sampler=True,
        fit_intercept=True,
        activation="tanh",
        epochs=10000,
        patience=1000,
        train_pct=0.7,
        # Other stuff
        verbose=False,
        random_state=42,
    ):
        # assert economic_objective in ["sharpe", "sortino", "meanvariance", "meanvarianceskew", "crra"]

        self.n_latent = n_latent
        self.weight_decay = weight_decay
        self.reg_turnover = reg_turnover
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_ts_sampler = use_ts_sampler
        self.fit_intercept = fit_intercept
        self.activation = activation
        self.epochs = epochs
        self.patience = patience
        self.train_pct = train_pct
        # self.long_only = long_only
        self.verbose = verbose
        self.random_state = random_state

        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        # Initialize model
        self.model = MultiHeadRetNet(
            n_inputs=X.shape[1],
            n_latent=self.n_latent,
            n_outputs=y.shape[1],
            fit_intercept=self.fit_intercept,
            activation=self.activation,
        )

        # Create training and validation sets
        dates = sorted(X.index.get_level_values(1).unique())
        train_dates = dates[: int(self.train_pct * len(dates))]
        test_dates = dates[int(self.train_pct * len(dates)) :]

        X_train, y_train = (
            X[X.index.get_level_values(1).isin(train_dates)],
            y[y.index.get_level_values(1).isin(train_dates)],
        )
        X_val, y_val = (
            X[X.index.get_level_values(1).isin(test_dates)],
            y[y.index.get_level_values(1).isin(test_dates)],
        )

        # Scale training and validation sets for neural network
        self.X_scaler = StandardScaler(with_mean=False).fit(X_train)
        self.y_scaler = StandardScaler(with_mean=False).fit(y_train)

        X_train_scaled = self.X_scaler.transform(X_train)
        y_train_scaled = self.y_scaler.transform(y_train)

        X_val_scaled = self.X_scaler.transform(X_val)
        y_val_scaled = self.y_scaler.transform(y_val)

        # Create dataset in torch
        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_train_scaled), torch.Tensor(y_train_scaled)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_val_scaled), torch.Tensor(y_val_scaled)
        )

        # Create dataloaders for network training
        if not self.use_ts_sampler:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )
            train_loader_eval = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=32,
                shuffle=False,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=TimeSeriesSampler(
                    dataset=train_dataset, batch_size=self.batch_size, shuffle=True
                ),
            )
            train_loader_eval = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=TimeSeriesSampler(
                    dataset=train_dataset, batch_size=self.batch_size, shuffle=False
                ),
            )

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=32,
            shuffle=False,
        )

        """ Train network """
        # Model, optimizer and loss initialization
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_func = nn.MSELoss()
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer = optimizer,
        #     max_lr = self.learning_rate,
        #     epochs = self.epochs,
        #     steps_per_epoch = len(train_dataset),
        #     pct_start = 0.25,
        #     final_div_factor = 100000,
        # )
        # Training quantities
        EPOCHS = self.epochs
        training_losses = []
        validation_losses = []

        # Early stopping
        patience = self.patience
        counter = 0
        optim_score = np.inf
        optim_state_dict = None
        epoch_stopped = None

        for epoch in range(EPOCHS):
            self.model.train()
            for idx, (X_i, y_i) in enumerate(train_loader):
                optimizer.zero_grad()
                # forward pass
                preds = self.model(X_i)
                # loss
                loss = loss_func(y_i, preds)
                if self.reg_turnover > 0:
                    # TODO: consider switching to l2
                    pweight_levels = preds[1:] - preds[:-1]
                    pweight_l1 = torch.mean(torch.abs(pweight_levels))
                    loss += self.reg_turnover * pweight_l1
                # backward pass
                loss.backward()
                optimizer.step()
                # scheduler.step()
            # Evaluation
            self.model.eval()
            with torch.no_grad():
                # Training evaluation
                train_loss = 0
                for idx, (X_i, y_i) in enumerate(train_loader_eval):
                    # forward pass
                    preds = self.model(X_i)
                    # loss
                    loss = loss_func(y_i, preds)
                    train_loss += loss
                train_loss /= len(train_loader_eval)
                # Validation evaluation
                valid_loss = 0
                for idx, (X_i, y_i) in enumerate(valid_loader):
                    # forward pass
                    preds = self.model(X_i)
                    # loss
                    loss = loss_func(y_i, preds)
                    valid_loss += loss
                valid_loss /= len(valid_loader)
            training_losses.append(train_loss)
            validation_losses.append(valid_loss)

            # Early stopping
            if valid_loss.item() < optim_score:
                optim_state_dict = deepcopy(self.model.state_dict())
                counter = 0
                optim_score = valid_loss
            else:
                counter += 1

            if counter == patience:
                break

            if self.verbose:
                if epoch % 5 == 0:
                    print(
                        f"Epoch: {epoch+1}: train_loss = {train_loss}, valid_loss = {valid_loss}"
                    )

            # scheduler.step(valid_loss)

        if optim_state_dict:
            self.model.load_state_dict(optim_state_dict)

        return self

    def predict(self, X):
        # Scale data
        X_scaled = self.X_scaler.transform(X)

        # Switch to evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Convert to torch and pass through network
            X_scaled_torch = torch.Tensor(X_scaled)
            preds = self.model(X_scaled_torch).numpy()

            # Convert back to original
            preds = self.y_scaler.inverse_transform(preds)

        return preds


class MultiHeadPortfolioAllocationEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # Neural network structure hyperparameters
        n_latent,
        economic_loss,
        # Neural network training hyperparameters
        weight_decay=1e-4,
        reg_turnover=0,
        batch_size=16,
        learning_rate=3e-4,
        use_ts_sampler=True,
        fit_intercept=True,
        activation="tanh",
        epochs=10000,
        patience=1000,
        train_pct=0.7,
        # Other stuff
        verbose=False,
        random_state=42,
    ):
        # assert economic_objective in ["sharpe", "sortino", "meanvariance", "meanvarianceskew", "crra"]

        self.n_latent = n_latent
        self.weight_decay = weight_decay
        self.economic_loss = economic_loss
        self.reg_turnover = reg_turnover
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_ts_sampler = use_ts_sampler
        self.fit_intercept = fit_intercept
        self.activation = activation
        self.epochs = epochs
        self.patience = patience
        self.train_pct = train_pct
        # self.long_only = long_only
        self.verbose = verbose
        self.random_state = random_state

        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        # Initialize model
        self.model = MultiHeadPANet(
            n_inputs=X.shape[1],
            n_latent=self.n_latent,
            n_outputs=y.shape[1],
            fit_intercept=self.fit_intercept,
            activation=self.activation,
        )

        # Create training and validation sets
        dates = sorted(X.index.get_level_values(1).unique())
        train_dates = dates[: int(self.train_pct * len(dates))]
        test_dates = dates[int(self.train_pct * len(dates)) :]

        X_train, y_train = (
            X[X.index.get_level_values(1).isin(train_dates)],
            y[y.index.get_level_values(1).isin(train_dates)],
        )
        X_val, y_val = (
            X[X.index.get_level_values(1).isin(test_dates)],
            y[y.index.get_level_values(1).isin(test_dates)],
        )

        # Scale training and validation sets for neural network
        self.X_scaler = StandardScaler(with_mean=False).fit(X_train)
        self.y_scaler = StandardScaler(with_mean=False).fit(y_train)

        X_train_scaled = self.X_scaler.transform(X_train)
        y_train_scaled = self.y_scaler.transform(y_train)

        X_val_scaled = self.X_scaler.transform(X_val)
        y_val_scaled = self.y_scaler.transform(y_val)

        # Create dataset in torch
        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_train_scaled), torch.Tensor(y_train_scaled)
        )
        valid_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_val_scaled), torch.Tensor(y_val_scaled)
        )

        # Create dataloaders for network training
        if not self.use_ts_sampler:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )
            train_loader_eval = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=32,
                shuffle=False,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=TimeSeriesSampler(
                    dataset=train_dataset, batch_size=self.batch_size, shuffle=True
                ),
            )
            train_loader_eval = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_sampler=TimeSeriesSampler(
                    dataset=train_dataset, batch_size=self.batch_size, shuffle=False
                ),
            )

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=32,
            shuffle=False,
        )

        """ Train network """
        # Model, optimizer and loss initialization
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer = optimizer,
        #     max_lr = self.learning_rate,
        #     epochs = self.epochs,
        #     steps_per_epoch = len(train_dataset),
        #     pct_start = 0.25,
        #     final_div_factor = 100000,
        # )
        # Training quantities
        EPOCHS = self.epochs
        training_losses = []
        validation_losses = []

        # Early stopping
        patience = self.patience
        counter = 0
        optim_score = np.inf
        optim_state_dict = None
        # epoch_stopped = None

        for epoch in range(EPOCHS):
            self.model.train()
            for idx, (X_i, y_i) in enumerate(train_loader):
                optimizer.zero_grad()
                # forward pass
                preds = self.model(X_i)
                # loss
                loss = self.economic_loss(y_i, preds)
                if self.reg_turnover > 0:
                    pweight_levels = preds[1:] - preds[:-1]
                    pweight_l1 = torch.mean(torch.abs(pweight_levels))
                    loss += self.reg_turnover * pweight_l1
                # backward pass
                loss.backward()
                optimizer.step()
                # scheduler.step()
            # Evaluation
            self.model.eval()
            with torch.no_grad():
                # Training evaluation
                train_loss = 0
                for idx, (X_i, y_i) in enumerate(train_loader_eval):
                    # forward pass
                    preds = self.model(X_i)
                    # loss
                    loss = self.economic_loss(y_i, preds)
                    train_loss += loss
                train_loss /= len(train_loader_eval)
                # Validation evaluation
                valid_loss = 0
                for idx, (X_i, y_i) in enumerate(valid_loader):
                    # forward pass
                    preds = self.model(X_i)
                    # loss
                    loss = self.economic_loss(y_i, preds)
                    valid_loss += loss
                valid_loss /= len(valid_loader)
            training_losses.append(train_loss)
            validation_losses.append(valid_loss)

            # Early stopping
            if valid_loss.item() < optim_score:
                optim_state_dict = deepcopy(self.model.state_dict())
                counter = 0
                optim_score = valid_loss.item()
            else:
                counter += 1

            if counter == patience:
                break

            if self.verbose:
                if epoch % 5 == 0:
                    print(
                        f"Epoch: {epoch+1}: train_loss = {train_loss}, valid_loss = {valid_loss}"
                    )

            # scheduler.step(valid_loss)

        if optim_state_dict:
            self.model.load_state_dict(optim_state_dict)

        return self

    def predict(self, X):
        # Scale data
        X_scaled = self.X_scaler.transform(X)

        # Switch to evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Convert to torch and pass through network
            X_scaled_torch = torch.Tensor(X_scaled)
            preds = self.model(X_scaled_torch).numpy()

            # Convert back to original
            # preds = self.y_scaler.inverse_transform(preds) # TODO: this might not be necessary in this setup

        return preds
