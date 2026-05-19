import unittest
import numpy as np
import pandas as pd

from parameterized import parameterized
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from macrosynergy.learning.forecasting.torch.models.mlps import MultiLayerPerceptron
from macrosynergy.learning.forecasting.nn import MLPRegressor

# Set up invalid torch_models for testing
class InvalidModelNoBase(nn.Module):
    def __init__(self, n_latent = 4):
        super().__init__()
        self.n_latent = n_latent

        self.model = nn.Sequential(
            nn.Linear(2, self.n_latent),
            nn.ReLU(),
            nn.Linear(self.n_latent, 2),
        )

    def forward(self, x):
        out = self.model(x)
        return out
    
class InvalidModelNoModule(BaseEstimator):
    def __init__(self, n_latent = 4):
        super().__init__()
        self.n_latent = n_latent

        self.model = nn.Sequential(
            nn.Linear(2, self.n_latent),
            nn.ReLU(),
            nn.Linear(self.n_latent, 2),
        )

    def forward(self, x):
        out = self.model(x)
        return out
    
class InvalidModelNoForward(nn.Module, BaseEstimator):
    def __init__(self, n_latent = 4):
        super().__init__()
        self.n_latent = n_latent

        self.model = nn.Sequential(
            nn.Linear(2, self.n_latent),
            nn.ReLU(),
            nn.Linear(self.n_latent, 2),
        )

class ValidModel(nn.Module, BaseEstimator):
    def __init__(self, n_latent = 4):
        super().__init__()
        self.n_latent = n_latent

        self.model = nn.Sequential(
            nn.Linear(2, self.n_latent),
            nn.ReLU(),
            nn.Linear(self.n_latent, 2),
        )

    def forward(self, x):
        out = self.model(x)
        return out
    
class TestMLPRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Generate data with true linear relationship
        cids = ["USD"]
        xcats = ["GROWTH", "RIR", "XR", "XR2"]

        df_cids = pd.DataFrame(index=cids, columns=["earliest", "latest"])
        df_cids.loc["USD"] = ["2019-06-01", "2020-12-31"]

        tuples = []

        for cid in cids:
            # get list of all eligible dates
            sdate = df_cids.loc[cid]["earliest"]
            edate = df_cids.loc[cid]["latest"]
            all_days = pd.date_range(sdate, edate)
            work_days = all_days[all_days.weekday < 5]
            for work_day in work_days:
                tuples.append((cid, work_day))

        n_samples = len(tuples)
        ftrs = np.random.normal(loc=0, scale=1, size=(n_samples, 2))
        labels1 = np.matmul(ftrs, [1, 2]) + np.random.normal(0, 0.5, len(ftrs))
        labels2 = np.matmul(ftrs, [-1, 0.5]) + np.random.normal(0, 0.5, len(ftrs))
        labels2 += 0.1 - 0.75 * labels1 + np.random.normal(0, 0.5, len(ftrs))
        df = pd.DataFrame(
            data=np.concatenate((np.reshape(labels1, (-1, 1)), np.reshape(labels2, (-1, 1)), ftrs), axis=1),
            index=pd.MultiIndex.from_tuples(tuples, names=["cid", "real_date"]),
            columns=xcats,
            dtype=np.float32,
        )

        self.X = df.drop(columns=["XR", "XR2"])
        self.X_numpy = self.X.values
        self.X_nan = self.X.copy()
        self.X_nan["nan_col"] = np.nan
        self.y = df[["XR", "XR2"]]
        self.y_numpy = self.y.values
        self.y_nan = self.y.copy()
        self.y_nan.iloc[0] = np.nan

        # set up invalid torch_models for testing

    def test_types_init(self):
        """ n_latent """
        # if only a single element is provided, it should be an integer greater than 0
        with self.assertRaises(TypeError):
            model = MLPRegressor(n_latent="not an int")
        with self.assertRaises(ValueError):
            model = MLPRegressor(n_latent=-1)
        with self.assertRaises(ValueError):
            model = MLPRegressor(n_latent=0)
        # if a list is provided, all elements should be integers greater than 0
        with self.assertRaises(TypeError):
            model = MLPRegressor(n_latent=[8, "not an int"])
        with self.assertRaises(ValueError):
            model = MLPRegressor(n_latent=[8, -1])
        with self.assertRaises(ValueError):
            model = MLPRegressor(n_latent=[8, 0])

        """ fit_encoder_intercept """
        # should be a boolean
        # torch_model needs to be None
        with self.assertRaises(TypeError):
            model = MLPRegressor(fit_encoder_intercept="not a bool")
        with self.assertRaises(ValueError):
            model = MLPRegressor(fit_encoder_intercept=True, torch_model="not None")

        """ fit_head_intercept """
        # should be a boolean
        # torch_model needs to be None
        with self.assertRaises(TypeError):
            model = MLPRegressor(fit_head_intercept="not a bool")
        with self.assertRaises(ValueError):
            model = MLPRegressor(fit_head_intercept=True, torch_model="not None")

        """ encoder_activation """
        # should be a string with accepted values ("tanh", "relu", "sigmoid")
        # torch_model needs to be None
        with self.assertRaises(TypeError):
            model = MLPRegressor(encoder_activation=123)
        with self.assertRaises(ValueError):
            model = MLPRegressor(encoder_activation="not an activation")
        with self.assertRaises(ValueError):
            model = MLPRegressor(encoder_activation="relu", torch_model="not None")

        """ head_activation """
        # should be a string with accepted values ("tanh", "relu", "sigmoid", "identity")
        # torch_model needs to be None
        with self.assertRaises(TypeError):
            model = MLPRegressor(head_activation=123)
        with self.assertRaises(ValueError):
            model = MLPRegressor(head_activation="not an activation")
        with self.assertRaises(ValueError):
            model = MLPRegressor(head_activation="relu", torch_model="not None")

        """ dropout_p """
        # should be a float between 0 and 0.5
        # torch_model needs to be None
        with self.assertRaises(TypeError):
            model = MLPRegressor(dropout_p="not a float")
        with self.assertRaises(ValueError):
            model = MLPRegressor(dropout_p=-0.1)
        with self.assertRaises(ValueError):
            model = MLPRegressor(dropout_p=0.5)
        with self.assertRaises(ValueError):
            model = MLPRegressor(dropout_p=0.6)
        with self.assertRaises(ValueError):
            model = MLPRegressor(dropout_p=1.234)

        with self.assertRaises(ValueError):
            model = MLPRegressor(torch_model="not None", dropout_p=0.2)

        """ torch_model """
        # should be an instance of nn.Module and BaseEstimator, with a valid forward method
        # (fit_encoder_intercept, fit_head_intercept, encoder_activation, head_activation, dropout_p) need to be None
        with self.assertRaises(TypeError):
            model = MLPRegressor(
                torch_model="not a model",
                fit_encoder_intercept=None,
                fit_head_intercept=None,
                encoder_activation=None,
                head_activation=None,
                dropout_p=None,
                n_latent=None,
            )
        with self.assertRaises(TypeError):
            model = MLPRegressor(
                torch_model=InvalidModelNoBase(),
                fit_encoder_intercept=None,
                fit_head_intercept=None,
                encoder_activation=None,
                head_activation=None,
                dropout_p=None,
                n_latent=None,
            )
        with self.assertRaises(TypeError):
            model = MLPRegressor(
                torch_model=InvalidModelNoModule(),
                fit_encoder_intercept=None,
                fit_head_intercept=None,
                encoder_activation=None,
                head_activation=None,
                dropout_p=None,
                n_latent=None,
            )
        with self.assertRaises(ValueError):
            model = MLPRegressor(
                torch_model=InvalidModelNoForward(),
                fit_encoder_intercept=None,
                fit_head_intercept=None,
                encoder_activation=None,
                head_activation=None,
                dropout_p=None,
                n_latent=None,
            )

        # Check errors are raised when incompatible parameters are provided together
        with self.assertRaises(ValueError):
            model = MLPRegressor(
                torch_model=ValidModel(),
                fit_encoder_intercept=True,
                fit_head_intercept=None,
                encoder_activation=None,
                head_activation=None,
                dropout_p=None,
                n_latent=None,
            )
        with self.assertRaises(ValueError):
            model = MLPRegressor(
                torch_model=ValidModel(),
                fit_encoder_intercept=None,
                fit_head_intercept=True,
                encoder_activation=None,
                head_activation=None,
                dropout_p=None,
                n_latent=None,
            )
        with self.assertRaises(ValueError):
            model = MLPRegressor(
                torch_model=ValidModel(),
                fit_encoder_intercept=None,
                fit_head_intercept=None,
                encoder_activation="relu",
                head_activation=None,
                dropout_p=None,
                n_latent=None,
            )
        with self.assertRaises(ValueError):
            model = MLPRegressor(
                torch_model=ValidModel(),
                fit_encoder_intercept=None,
                fit_head_intercept=None,
                encoder_activation=None,
                head_activation="relu",
                dropout_p=None,
                n_latent=None,
            )
        with self.assertRaises(ValueError):
            model = MLPRegressor(
                torch_model=ValidModel(),
                fit_encoder_intercept=None,
                fit_head_intercept=None,
                encoder_activation=None,
                head_activation=None,
                dropout_p=0.2,
                n_latent=None,
            )
        with self.assertRaises(ValueError):
            model = MLPRegressor(
                torch_model=ValidModel(),
                fit_encoder_intercept=None,
                fit_head_intercept=None,
                encoder_activation=None,
                head_activation=None,
                dropout_p=None,
                n_latent=4,
            )

        """ Loss function """
        # Should inherit from nn.Module and have a valid forward method
        with self.assertRaises(TypeError):
            model = MLPRegressor(loss_func="not a loss function")
        with self.assertRaises(ValueError):
            model = MLPRegressor(loss_func=nn.Linear(2, 2))

        """ Optimizer """
        # Should be a string with options "AdamW" or "SGD+mom"
        with self.assertRaises(TypeError):
            model = MLPRegressor(optimizer=123)

        with self.assertRaises(ValueError):
            model = MLPRegressor(optimizer="not a valid optimizer")
        
        """ Scheduler """
        # Should be a string with options "OneCycleLR"
        with self.assertRaises(TypeError):
            model = MLPRegressor(scheduler=123)

        with self.assertRaises(ValueError):
            model = MLPRegressor(scheduler="not a valid scheduler")

        """ batch_size """
        # Should be an integer greater than 0
        with self.assertRaises(TypeError):
            model = MLPRegressor(batch_size="not an int")
        with self.assertRaises(ValueError):
            model = MLPRegressor(batch_size=-1)
        with self.assertRaises(ValueError):
            model = MLPRegressor(batch_size=0)

        """ learning_rate """
        # Should be a float greater than 0
        with self.assertRaises(TypeError):
            model = MLPRegressor(learning_rate="not a float")
        with self.assertRaises(ValueError):
            model = MLPRegressor(learning_rate=-0.001)
        with self.assertRaises(ValueError):
            model = MLPRegressor(learning_rate=0)

        """ weight_decay """
        # Should be a float greater than or equal to 0
        with self.assertRaises(TypeError):
            model = MLPRegressor(weight_decay="not a float")
        with self.assertRaises(ValueError):
            model = MLPRegressor(weight_decay=-0.001)

        """ reg_turnover """
        # Should be a float greater than or equal to 0
        with self.assertRaises(TypeError):
            model = MLPRegressor(reg_turnover="not a float")
        with self.assertRaises(ValueError):
            model = MLPRegressor(reg_turnover=-0.001)

        """ use_ts_sampler """
        # Should be a boolean
        with self.assertRaises(TypeError):
            model = MLPRegressor(use_ts_sampler="not a bool")

        """ aggregate_last """
        # Should be a boolean and neither aggregate_last and drop_last can be True at the same time
        with self.assertRaises(TypeError):
            model = MLPRegressor(aggregate_last="not a bool")
        with self.assertRaises(ValueError):
            model = MLPRegressor(aggregate_last=True, drop_last=True)

        """ drop_last """
        # Should be a boolean and neither aggregate_last and drop_last can be True at the same time
        with self.assertRaises(TypeError):
            model = MLPRegressor(drop_last="not a bool")
        with self.assertRaises(ValueError):
            model = MLPRegressor(drop_last=True, aggregate_last=True)

        """ epochs """
        # Should be an integer greater than 0
        with self.assertRaises(TypeError):
            model = MLPRegressor(epochs="not an int")
        with self.assertRaises(ValueError):
            model = MLPRegressor(epochs=-1)
        with self.assertRaises(ValueError):
            model = MLPRegressor(epochs=0)

        """ patience """
        # Should be an integer greater than or equal to 0
        with self.assertRaises(TypeError):
            model = MLPRegressor(patience="not an int")
        with self.assertRaises(ValueError):
            model = MLPRegressor(patience=-1)
        with self.assertRaises(ValueError):
            model = MLPRegressor(patience=0)

        """ train_pct """
        # Should be a float between 0 and 1
        with self.assertRaises(TypeError):
            model = MLPRegressor(train_pct="not a float")
        with self.assertRaises(ValueError):
            model = MLPRegressor(train_pct=-0.1)
        with self.assertRaises(ValueError):
            model = MLPRegressor(train_pct=1.1)
        with self.assertRaises(ValueError):
            model = MLPRegressor(train_pct=1)

        """ x_scaler """
        # Instance of StandardScaler or None 
        with self.assertRaises(TypeError):
            model = MLPRegressor(x_scaler="not a scaler")

        """ y_scaler """
        # Instance of StandardScaler or None
        with self.assertRaises(TypeError):
            model = MLPRegressor(y_scaler="not a scaler")

        """ verbose """
        # Should be a boolean
        with self.assertRaises(TypeError):
            model = MLPRegressor(verbose="not a bool")

        """ random_state """
        # Should be a positive integer 
        with self.assertRaises(TypeError):
            model = MLPRegressor(random_state="not an int")
        with self.assertRaises(ValueError):
            model = MLPRegressor(random_state=-1)

        """ inverse_transform_preds """
        # Should be a boolean
        with self.assertRaises(TypeError):
            model = MLPRegressor(inverse_transform_preds="not a bool")

        """ min_samples """
        # Should be an integer greater than 0
        with self.assertRaises(TypeError):
            model = MLPRegressor(min_samples="not an int")
        with self.assertRaises(ValueError):
            model = MLPRegressor(min_samples=-1)
        with self.assertRaises(ValueError):
            model = MLPRegressor(min_samples=0)


    def test_valid_init(self):
        """  Test that valid parameters do not raise errors and are set correctly. """
        # Test default parameters
        model = MLPRegressor()
        self.assertEqual(model.n_latent, 32)
        self.assertEqual(model.fit_encoder_intercept, True)
        self.assertEqual(model.fit_head_intercept, True)
        self.assertEqual(model.encoder_activation, "relu")
        self.assertEqual(model.head_activation, "identity")
        self.assertEqual(model.dropout_p, 0)
        self.assertEqual(model.torch_model, None)
        self.assertIsInstance(model.loss_func, nn.MSELoss)
        self.assertEqual(model.optimizer, "AdamW")
        self.assertEqual(model.scheduler, None)
        self.assertEqual(model.batch_size, 32)
        self.assertEqual(model.learning_rate, 3e-4)
        self.assertEqual(model.weight_decay, 1e-4)
        self.assertEqual(model.reg_turnover, 0)
        self.assertEqual(model.use_ts_sampler, True)
        self.assertEqual(model.aggregate_last, True)
        self.assertEqual(model.drop_last, False)
        self.assertEqual(model.epochs, 10000)
        self.assertEqual(model.patience, 1000)
        self.assertEqual(model.train_pct, 0.7)
        self.assertIsInstance(model.x_scaler, StandardScaler)
        self.assertIsInstance(model.y_scaler, StandardScaler)
        self.assertEqual(model.verbose, False)
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.inverse_transform_preds, False)
        self.assertEqual(model.min_samples, 36)

        self.assertIsInstance(model, MLPRegressor)
        self.assertIsInstance(model, BaseEstimator)
        self.assertIsInstance(model, RegressorMixin)

        self.assertIsInstance(model.optimizers, list)
        self.assertIsInstance(model.optimizers[0], str)
        self.assertEqual(model.optimizers, ["AdamW"])
        self.assertIsInstance(model.random_states, list)
        self.assertIsInstance(model.random_states[0], int)
        self.assertEqual(model.random_states, [42])

        # Repeat when providing custom parameters
        model = MLPRegressor(
            n_latent = None,
            fit_encoder_intercept = None,
            fit_head_intercept = None,
            encoder_activation=None,
            head_activation=None,
            dropout_p=None,
            torch_model = ValidModel(),
            loss_func=nn.L1Loss(),
            optimizer=["AdamW", "SGD+mom"],
            scheduler="OneCycleLR",
            batch_size = 16,
            learning_rate = 1e-3,
            weight_decay = 0.01,
            reg_turnover = 0.1,
            use_ts_sampler = False,
            aggregate_last = False,
            drop_last = True,
            epochs = 10,
            patience = 2,
            train_pct = 0.8,
            x_scaler=StandardScaler(with_mean=True),
            y_scaler=None,
            verbose = True,
            random_state = [42,43],
            inverse_transform_preds = True,
            min_samples = 10,
        )
        self.assertEqual(model.n_latent, None)
        self.assertEqual(model.fit_encoder_intercept, None)
        self.assertEqual(model.fit_head_intercept, None)
        self.assertEqual(model.encoder_activation, None)
        self.assertEqual(model.head_activation, None)
        self.assertEqual(model.dropout_p, None)
        self.assertIsInstance(model.torch_model, ValidModel)
        self.assertIsInstance(model.loss_func, nn.L1Loss)
        self.assertEqual(model.optimizer, ["AdamW", "SGD+mom"])
        self.assertEqual(model.scheduler, "OneCycleLR")
        self.assertEqual(model.batch_size, 16)
        self.assertEqual(model.learning_rate, 1e-3)
        self.assertEqual(model.weight_decay, 0.01)
        self.assertEqual(model.reg_turnover, 0.1)
        self.assertEqual(model.use_ts_sampler, False)
        self.assertEqual(model.aggregate_last, False)
        self.assertEqual(model.drop_last, True)
        self.assertEqual(model.epochs, 10)
        self.assertEqual(model.patience, 2)
        self.assertEqual(model.train_pct, 0.8)
        self.assertIsInstance(model.x_scaler, StandardScaler)
        self.assertIsNone(model.y_scaler)
        self.assertEqual(model.verbose, True)
        self.assertEqual(model.random_state, [42,43])
        self.assertEqual(model.inverse_transform_preds, True)
        self.assertEqual(model.min_samples, 10)

        self.assertIsInstance(model, MLPRegressor)
        self.assertIsInstance(model, BaseEstimator)
        self.assertIsInstance(model, RegressorMixin)

        self.assertIsInstance(model.optimizers, list)
        self.assertEqual(model.optimizers, ["AdamW", "SGD+mom"])
        self.assertIsInstance(model.random_states, list)
        self.assertEqual(model.random_states, [42,43])

    def test_types_fit(self):
        """
        Test inputs of the fit method are checked for correctness.
        """
        model = MLPRegressor()
        # Test type of 'X' parameter
        self.assertRaises(TypeError, model.fit, X=1, y=self.y)
        self.assertRaises(TypeError, model.fit, X="X", y=self.y)
        self.assertRaises(TypeError, model.fit, X=self.X.values, y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X_nan, y=self.y)
        self.assertRaises(ValueError, model.fit, X=self.X.reset_index(), y=self.y)
        # Test type of 'y' parameter
        self.assertRaises(TypeError, model.fit, X=self.X, y=1)
        self.assertRaises(TypeError, model.fit, X=self.X, y="y")
        self.assertRaises(TypeError, model.fit, X=self.X, y=self.y.values)
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y.reset_index())
        # Test type of sample_weight
        self.assertRaises(TypeError, model.fit, X=self.X, y=self.y, sample_weight="weight")
        self.assertRaises(TypeError, model.fit, X=self.X, y=self.y, sample_weight=np.array(["weight"] * len(self.X)))
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y, sample_weight=np.array([1.0, 2.0]))
        self.assertRaises(TypeError, model.fit, X=self.X, y=self.y, sample_weight=np.array([1.0, "two", 3.0] * (len(self.X)//3)))
        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y, sample_weight=np.array([1.0, -2.0, 3.0] * (len(self.X)//3)))

        self.assertRaises(ValueError, model.fit, X=self.X, y=self.y, sample_weight=np.array([1.0, 2.0, 3.0] * (len(self.X)//3)))

    def test_valid_fit(self):
        """ 
        Test that valid models run as expected under different hyperparameter settings with
        stored attributes as expected. 
        """
        # Test on basic hyperparameter settings
        model1 = MLPRegressor(
            n_latent = 32,
            fit_encoder_intercept = True,
            fit_head_intercept = False,
            encoder_activation="relu",
            head_activation="tanh",
            dropout_p=0.2,
            torch_model = None,
            loss_func=nn.MSELoss(),
            optimizer="AdamW",
            scheduler="OneCycleLR",
            batch_size = 32,
            learning_rate = 3e-4,
            weight_decay = 1e-4,
            reg_turnover = 1e-4,
            use_ts_sampler = True,
            aggregate_last = True,
            drop_last = False,
            epochs = 5,
            patience = 2,
            train_pct = 0.7,
            x_scaler=StandardScaler(with_mean=False),
            y_scaler=StandardScaler(with_mean=False),
            verbose = False,
            random_state = 42,
            inverse_transform_preds = False,
            min_samples = 36,
        ).fit(self.X, self.y)

        self.assertIsInstance(model1, MLPRegressor)
        self.assertIsInstance(model1.models, list)
        self.assertIsInstance(model1.models[0], nn.Module)

        for parameter in model1.models[0].parameters():
            self.assertFalse(torch.isnan(parameter).any())
            self.assertFalse(torch.isinf(parameter).any())

        # Repeat and check the parameters are the same 
        model2 = MLPRegressor(
            n_latent = 32,
            fit_encoder_intercept = True,
            fit_head_intercept = False,
            encoder_activation="relu",
            head_activation="tanh",
            dropout_p=0.2,
            torch_model = None,
            loss_func=nn.MSELoss(),
            optimizer="AdamW",
            scheduler="OneCycleLR",
            batch_size = 32,
            learning_rate = 3e-4,
            weight_decay = 1e-4,
            reg_turnover = 1e-4,
            use_ts_sampler = True,
            aggregate_last = True,
            drop_last = False,
            epochs = 5,
            patience = 2,
            train_pct = 0.7,
            x_scaler=StandardScaler(with_mean=False),
            y_scaler=StandardScaler(with_mean=False),
            verbose = False,
            random_state = 42,
            inverse_transform_preds = False,
            min_samples = 36,
        ).fit(self.X, self.y)

        for param1, param2 in zip(model1.models[0].parameters(), model2.models[0].parameters()):
            self.assertTrue(torch.equal(param1, param2))

        # Test that different random state leads to different parameters
        model3 = MLPRegressor(
            n_latent = 32,
            fit_encoder_intercept = True,
            fit_head_intercept = False,
            encoder_activation="relu",
            head_activation="tanh",
            dropout_p=0.2,
            torch_model = None,
            loss_func=nn.MSELoss(),
            optimizer="AdamW",
            scheduler="OneCycleLR",
            batch_size = 32,
            learning_rate = 3e-4,
            weight_decay = 1e-4,
            reg_turnover = 1e-4,
            use_ts_sampler = True,
            aggregate_last = True,
            drop_last = False,
            epochs = 5,
            patience = 2,
            train_pct = 0.7,
            x_scaler=StandardScaler(with_mean=False),
            y_scaler=StandardScaler(with_mean=False),
            verbose = False,
            random_state = 43,
            inverse_transform_preds = False,
            min_samples = 36,
        ).fit(self.X, self.y)

        for param1, param3 in zip(model1.models[0].parameters(), model3.models[0].parameters()):
            self.assertFalse(torch.equal(param1, param3))

        # Test on custom torch_model as expected 
        model4 = MLPRegressor(
            torch_model=ValidModel(n_latent=16),
            n_latent = None,
            fit_encoder_intercept=None,
            fit_head_intercept=None,
            encoder_activation=None,
            head_activation=None,
            dropout_p=None,
            epochs = 5,
            patience = 2,
        ).fit(self.X, self.y)

        model5 = MLPRegressor(
            torch_model=ValidModel(n_latent=16),
            n_latent = None,
            fit_encoder_intercept=None,
            fit_head_intercept=None,
            encoder_activation=None,
            head_activation=None,
            dropout_p=None,
            epochs = 5,
            patience = 2,
        ).fit(self.X, self.y)

        model6 = MLPRegressor(
            torch_model=ValidModel(n_latent=16),
            fit_encoder_intercept=None,
            fit_head_intercept=None,
            encoder_activation=None,
            head_activation=None,
            dropout_p=None,
            n_latent=None,
            epochs = 5,
            patience = 2,
            random_state=123,
        ).fit(self.X, self.y)

        self.assertIsInstance(model4, MLPRegressor)
        self.assertIsInstance(model4.models, list)
        self.assertIsInstance(model4.models[0], nn.Module)

        for parameter in model4.models[0].parameters():
            self.assertFalse(torch.isnan(parameter).any())
            self.assertFalse(torch.isinf(parameter).any())

        for param4, param5 in zip(model4.models[0].parameters(), model5.models[0].parameters()):
            self.assertTrue(torch.equal(param4, param5))

        for param4, param6 in zip(model4.models[0].parameters(), model6.models[0].parameters()):
            self.assertFalse(torch.equal(param4, param6))

        # Now test the model runs with multiple optimizers and random states
        model7 = MLPRegressor(
            n_latent = 32,
            fit_encoder_intercept = True,
            fit_head_intercept = False,
            encoder_activation="relu",
            head_activation="tanh",
            dropout_p=0.2,
            torch_model = None,
            loss_func=nn.MSELoss(),
            epochs = 5,
            patience = 2,
            optimizer=["AdamW", "SGD+mom"],
            random_state=[42, 43],
        ).fit(self.X, self.y)

        self.assertIsInstance(model7, MLPRegressor)
        self.assertIsInstance(model7.models, list)
        self.assertEqual(len(model7.models), 4)
        for submodel in model7.models:
            self.assertIsInstance(submodel, nn.Module)
            for parameter in submodel.parameters():
                self.assertFalse(torch.isnan(parameter).any())
                self.assertFalse(torch.isinf(parameter).any())

        model8 = MLPRegressor(
            n_latent = 32,
            fit_encoder_intercept = True,
            fit_head_intercept = False,
            encoder_activation="relu",
            head_activation="tanh",
            dropout_p=0.2,
            torch_model = None,
            loss_func=nn.MSELoss(),
            epochs = 5,
            patience = 2,
            optimizer=["AdamW", "SGD+mom"],
            random_state=[42, 43],
        ).fit(self.X, self.y)

        for submodel1, submodel8 in zip(model7.models, model8.models):
            for param1, param8 in zip(submodel1.parameters(), submodel8.parameters()):
                self.assertTrue(torch.equal(param1, param8))

        # Lastly check that a custom torch model runs with multiple optimizers and random states as expected
        model9 = MLPRegressor(
            torch_model=ValidModel(n_latent=16),
            fit_encoder_intercept=None,
            fit_head_intercept=None,
            encoder_activation=None,
            head_activation=None,
            dropout_p=None,
            n_latent=None,
            epochs = 5,
            patience = 2,
            optimizer=["AdamW", "SGD+mom"],
            random_state=[42, 43],
        ).fit(self.X, self.y)

        model10 = MLPRegressor(
            torch_model=ValidModel(n_latent=16),
            fit_encoder_intercept=None,
            fit_head_intercept=None,
            encoder_activation=None,
            head_activation=None,
            dropout_p=None,
            epochs = 5,
            patience = 2,
            n_latent = None,
            optimizer=["AdamW", "SGD+mom"],
            random_state=[42, 43],
        ).fit(self.X, self.y)

        for submodel9, submodel10 in zip(model9.models, model10.models):
            for param9, param10 in zip(submodel9.parameters(), submodel10.parameters()):
                self.assertTrue(torch.equal(param9, param10))

    def test_types_predict(self):
        model = MLPRegressor().fit(self.X, self.y)
        # Test type of 'X' parameter
        self.assertRaises(TypeError, model.predict, X=1)
        self.assertRaises(TypeError, model.predict, X="X")
        self.assertRaises(TypeError, model.predict, X=self.X.values)
        self.assertRaises(ValueError, model.predict, X=self.X_nan)
        self.assertRaises(ValueError, model.predict, X=self.X.reset_index())

    def test_valid_predict(self):
        # Test that predictions are returned in the expected format
        model = MLPRegressor(epochs = 5, patience = 2)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)

        self.assertIsInstance(preds, pd.DataFrame)
        self.assertEqual(preds.shape, self.y.shape)
        self.assertFalse(preds.isna().any().any())
        self.assertFalse(np.isinf(preds.values).any())

        pd.testing.assert_index_equal(preds.index, self.y.index)
        pd.testing.assert_index_equal(preds.columns, self.y.columns)

        # Check that running fit again doesn't change the predictions if random state is the same
        model.fit(self.X, self.y)
        preds2 = model.predict(self.X)
        pd.testing.assert_frame_equal(preds, preds2)

        # Check that predictions from different seeds are different
        model_diff_seed = MLPRegressor(epochs = 5, patience = 2, random_state=43).fit(self.X, self.y)
        preds_diff_seed = model_diff_seed.predict(self.X)
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(preds, preds_diff_seed)

        # Now check that when a target has insufficient data, predictions are only
        # produced for the target with sufficient data
        # suppose the second column only has 10 samples, which is less than the
        # min_samples of 36, while the first column has all samples
        y_insufficient = self.y.copy()
        y_insufficient.iloc[10:, 1] = np.nan
        model_insufficient = MLPRegressor(epochs = 5, patience = 2, min_samples=36).fit(self.X, y_insufficient)
        preds_insufficient = model_insufficient.predict(self.X)
        self.assertIsInstance(preds_insufficient, pd.DataFrame)
        self.assertEqual(preds_insufficient.shape[1], 1)
        self.assertEqual(preds_insufficient.shape[0], self.y.shape[0])
        self.assertIn("XR", preds_insufficient.columns)
        self.assertNotIn("XR2", preds_insufficient.columns)

    def test_valid_create_train_test_splits(self):
        # Test that there is no leakage between train and validation sets in the time split and that the split is done according to the specified train_pct
        model = MLPRegressor()
        X_tr, X_va, y_tr, y_va = model.create_train_valid_splits(self.X, self.y, train_pct=0.6)

        dates = sorted(self.X.index.get_level_values(1).unique())
        cut = int(0.6 * len(dates))
        train_dates = set(dates[:cut])
        val_dates = set(dates[cut:])

        self.assertEqual(set(X_tr.index.get_level_values("real_date").unique()), train_dates)
        self.assertEqual(set(X_va.index.get_level_values("real_date").unique()), val_dates)
        self.assertEqual(set(y_tr.index.get_level_values("real_date").unique()), train_dates)
        self.assertEqual(set(y_va.index.get_level_values("real_date").unique()), val_dates)

    def test_valid_scale_data(self):
        # Test that the data is scaled correctly when both scalers are provided
        model = MLPRegressor()
        X_tr, X_va, y_tr, y_va = model.create_train_valid_splits(self.X, self.y, train_pct=0.6)
        X_tr_scaled, X_va_scaled, y_tr_scaled, y_va_scaled = model.scale_data(X_tr, X_va, y_tr, y_va, x_scaler=StandardScaler(with_mean=True), y_scaler=StandardScaler(with_mean=True))

        self.assertIsInstance(X_tr_scaled, np.ndarray)
        self.assertIsInstance(X_va_scaled, np.ndarray)
        self.assertIsInstance(y_tr_scaled, np.ndarray)
        self.assertIsInstance(y_va_scaled, np.ndarray)

        self.assertAlmostEqual(X_tr_scaled.mean(), 0, places=5)
        self.assertAlmostEqual(y_tr_scaled.mean(), 0, places=5)
        self.assertAlmostEqual(X_tr_scaled.std(), 1, places=5)
        self.assertAlmostEqual(y_tr_scaled.std(), 1, places=5)

        # Repeat for the case where the means are not removed (a common use case)
        X_tr_scaled, X_va_scaled, y_tr_scaled, y_va_scaled = model.scale_data(X_tr, X_va, y_tr, y_va, x_scaler=StandardScaler(with_mean=False), y_scaler=StandardScaler(with_mean=False))
        self.assertIsInstance(X_tr_scaled, np.ndarray)
        self.assertIsInstance(X_va_scaled, np.ndarray)
        self.assertIsInstance(y_tr_scaled, np.ndarray)
        self.assertIsInstance(y_va_scaled, np.ndarray)

        # Repeat for the case where X is scaled but y is not
        X_tr_scaled, X_va_scaled, y_tr_scaled, y_va_scaled = model.scale_data(X_tr, X_va, y_tr, y_va, x_scaler=StandardScaler(with_mean=False), y_scaler=None)
        self.assertIsInstance(X_tr_scaled, np.ndarray)
        self.assertIsInstance(X_va_scaled, np.ndarray)
        self.assertIsInstance(y_tr_scaled, np.ndarray)
        self.assertIsInstance(y_va_scaled, np.ndarray)

        self.assertAlmostEqual(y_tr_scaled.mean(), y_tr.values.mean(), places=5)
        self.assertAlmostEqual(y_va_scaled.mean(), y_va.values.mean(), places=5)
        self.assertAlmostEqual(y_tr_scaled.std(), y_tr.values.std(), places=5)
        self.assertAlmostEqual(y_va_scaled.std(), y_va.values.std(), places=5)

    def test_valid_fit_one_batch(self):
        """
        Pass data through one training step and check that the model parameters have updated
        """
        torch_model = MultiLayerPerceptron(
            n_inputs=self.X.shape[1],
            n_latent=8,
            n_outputs=self.y.shape[1],
        )
        optimizer = torch.optim.Adam(torch_model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=10)
        before = {k: v.detach().clone() for k, v in torch_model.state_dict().items()}

        sklearn_model = MLPRegressor()
        X_tr, X_va, y_tr, y_va = sklearn_model.create_train_valid_splits(self.X, self.y, train_pct=0.6)
        X_tr_scaled, X_va_scaled, y_tr_scaled, y_va_scaled = sklearn_model.scale_data(X_tr, X_va, y_tr, y_va, x_scaler=StandardScaler(with_mean=True), y_scaler=StandardScaler(with_mean=True))

        torch_model = sklearn_model._fit_one_batch(
            torch_model,
            torch.Tensor(X_tr_scaled[:8]),
            torch.Tensor(y_tr_scaled[:8]),
            optimizer=optimizer,
            scheduler = None,
            loss_func = nn.MSELoss(),
            sample_weight = None,
            sample_weight_strategy = None,
        )
        after = {k: v.detach().clone() for k, v in torch_model.state_dict().items()}
        changed = any(not torch.equal(before[k], after[k]) for k in before.keys())
        self.assertTrue(changed)

        # Repeat with a scheduler 
        torch_model = sklearn_model._fit_one_batch(
            torch_model,
            torch.Tensor(X_tr_scaled[8:16]),
            torch.Tensor(y_tr_scaled[8:16]),
            optimizer=optimizer,
            scheduler = scheduler,
            loss_func = nn.MSELoss(),
            sample_weight = None,
            sample_weight_strategy = None,
        )
        after2 = {k: v.detach().clone() for k, v in torch_model.state_dict().items()}
        changed2 = any(not torch.equal(after[k], after2[k]) for k in after.keys())
        self.assertTrue(changed2)
