import unittest
import itertools
import numpy as np
import pandas as pd

from parameterized import parameterized
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from macrosynergy.learning.forecasting.torch.models.mlps import MultiLayerPerceptron
from macrosynergy.learning.forecasting.nn import MLPRegressor
import macrosynergy.learning as msl

# Set up invalid torch_models for testing
class InvalidModelNoBase(nn.Module):
    def __init__(self, n_latent = 4):
        super().__init__()
        self.n_latent = n_latent

        self.encoder = nn.Sequential(
            nn.Linear(2, self.n_latent),
            nn.ReLU(),
        )

        self.head = nn.Linear(self.n_latent, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.head(out)
        
        return out
    
class InvalidModelNoModule(BaseEstimator):
    def __init__(self, n_latent = 4):
        super().__init__()
        self.n_latent = n_latent

        self.encoder = nn.Sequential(
            nn.Linear(2, self.n_latent),
            nn.ReLU(),
        )
        self.head = nn.Linear(self.n_latent, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.head(out)
        return out
    
class InvalidModelNoForward(nn.Module, BaseEstimator):
    def __init__(self, n_latent = 4):
        super().__init__()
        self.n_latent = n_latent

        self.encoder = nn.Sequential(
            nn.Linear(2, self.n_latent),
            nn.ReLU(),
        )
        self.head = nn.Linear(self.n_latent, 2)

class ValidModel(nn.Module, BaseEstimator):
    def __init__(self, n_latent = 4):
        super().__init__()
        self.n_latent = n_latent


        self.encoder = nn.Sequential(
            nn.Linear(2, self.n_latent),
            nn.ReLU(),
        )
        self.head = nn.Linear(self.n_latent, 2)

    def forward(self, x):
        out = self.encoder(x)
        out = self.head(out)
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
            model = MLPRegressor(dropout_p=1.234)

        with self.assertRaises(ValueError):
            model = MLPRegressor(torch_model="not None", dropout_p=0.2)

        """signal_modifier"""
        # should be a callable
        with self.assertRaises(TypeError):
            model = MLPRegressor(signal_modifier="not a callable")
        # should be a Module object
        with self.assertRaises(TypeError):
            model = MLPRegressor(signal_modifier=123)

        """ normalization """
        # should be a string
        with self.assertRaises(TypeError):
            model = MLPRegressor(normalization=True)
        with self.assertRaises(TypeError):
            model = MLPRegressor(normalization=3)
        # should be one of batch or layer
        with self.assertRaises(ValueError):
            model = MLPRegressor(normalization="not an option")

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
                normalization=None
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
                normalization=None
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
                normalization=None
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
                normalization=None
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
                normalization=None
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
                normalization=None
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
                normalization=None
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
                normalization=None
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
                normalization=None
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
                normalization=None
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

        """ train_splitter """
        # Should be a float between 0 and 1 or a scikit-learn splitter
        with self.assertRaises(TypeError):
            model = MLPRegressor(train_splitter="not a float")
        with self.assertRaises(ValueError):
            model = MLPRegressor(train_splitter=-0.1)
        with self.assertRaises(ValueError):
            model = MLPRegressor(train_splitter=1.1)
        with self.assertRaises(ValueError):
            model = MLPRegressor(train_splitter=1)

        """ x_scaler """
        # Instance of StandardScaler or None 
        with self.assertRaises(TypeError):
            model = MLPRegressor(x_scaler="not a scaler")

        """ y_scaler """
        # Instance of StandardScaler or None
        with self.assertRaises(TypeError):
            model = MLPRegressor(y_scaler="not a scaler")

        """ refit """
        # Should be a boolean
        with self.assertRaises(TypeError):
            model = MLPRegressor(refit="not a bool")

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
        self.assertEqual(model.normalization, None)
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
        self.assertEqual(model.train_splitter, 0.7)
        self.assertIsInstance(model.x_scaler, StandardScaler)
        self.assertIsInstance(model.y_scaler, StandardScaler)
        self.assertEqual(model.refit, False)
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
            normalization=None,
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
            train_splitter = 0.8,
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
        self.assertEqual(model.normalization, None)
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
        self.assertEqual(model.train_splitter, 0.8)
        self.assertIsInstance(model.x_scaler, StandardScaler)
        self.assertIsNone(model.y_scaler)
        self.assertEqual(model.refit, False)
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
        # self.assertRaises(TypeError, model.fit, X=self.X, y=self.y, sample_weight="weight")
        # self.assertRaises(TypeError, model.fit, X=self.X, y=self.y, sample_weight=np.array(["weight"] * len(self.X)))
        # self.assertRaises(ValueError, model.fit, X=self.X, y=self.y, sample_weight=np.array([1.0, 2.0]))
        # self.assertRaises(TypeError, model.fit, X=self.X, y=self.y, sample_weight=np.array([1.0, "two", 3.0] * (len(self.X)//3)))
        # self.assertRaises(ValueError, model.fit, X=self.X, y=self.y, sample_weight=np.array([1.0, -2.0, 3.0] * (len(self.X)//3)))

        # self.assertRaises(ValueError, model.fit, X=self.X, y=self.y, sample_weight=np.array([1.0, 2.0, 3.0] * (len(self.X)//3)))

    @parameterized.expand(itertools.product([None, nn.Softmax(dim=-1), msl.LongShortModule()], ["batch", "layer", None], [True, False], [0.7, msl.RollingKFoldPanelSplit(5)]))
    def test_valid_fit(self, signal_modifier, normalization, refit, train_splitter):
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
            signal_modifier = signal_modifier,
            normalization = normalization,
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
            train_splitter = train_splitter,
            x_scaler=StandardScaler(with_mean=False),
            y_scaler=StandardScaler(with_mean=False),
            refit = refit,
            verbose = False,
            random_state = 42,
            inverse_transform_preds = False,
            min_samples = 36,
        ).fit(self.X, self.y)

        self.assertIsInstance(model1, MLPRegressor)
        self.assertIsInstance(model1.models, list)
        # Without refitting, each stored model is a list of validated models
        modules1 = model1.models[0] if isinstance(model1.models[0], list) else [model1.models[0]]
        for module in modules1:
            self.assertIsInstance(module, nn.Module)
            for parameter in module.parameters():
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
            signal_modifier = signal_modifier,
            normalization = normalization,
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
            train_splitter = train_splitter,
            x_scaler=StandardScaler(with_mean=False),
            y_scaler=StandardScaler(with_mean=False),
            refit = refit,
            verbose = False,
            random_state = 42,
            inverse_transform_preds = False,
            min_samples = 36,
        ).fit(self.X, self.y)

        modules2 = model2.models[0] if isinstance(model2.models[0], list) else [model2.models[0]]
        for module1, module2 in zip(modules1, modules2):
            for param1, param2 in zip(module1.parameters(), module2.parameters()):
                self.assertTrue(torch.equal(param1, param2))

        # Test that different random state leads to different parameters
        model3 = MLPRegressor(
            n_latent = 32,
            fit_encoder_intercept = True,
            fit_head_intercept = False,
            encoder_activation="relu",
            head_activation="tanh",
            dropout_p=0.2,
            signal_modifier = signal_modifier,
            normalization = normalization,
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
            train_splitter = train_splitter,
            x_scaler=StandardScaler(with_mean=False),
            y_scaler=StandardScaler(with_mean=False),
            refit = refit,
            verbose = False,
            random_state = 43,
            inverse_transform_preds = False,
            min_samples = 36,
        ).fit(self.X, self.y)

        modules3 = model3.models[0] if isinstance(model3.models[0], list) else [model3.models[0]]
        for module1, module3 in zip(modules1, modules3):
            for param1, param3 in zip(module1.parameters(), module3.parameters()):
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
            normalization = None,
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
            normalization = None,
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
            normalization = None,
            random_state=123,
        ).fit(self.X, self.y)

        self.assertIsInstance(model4, MLPRegressor)
        self.assertIsInstance(model4.models, list)
        # Without refitting, each stored model is a list of validated models
        modules4 = model4.models[0] if isinstance(model4.models[0], list) else [model4.models[0]]
        modules5 = model5.models[0] if isinstance(model5.models[0], list) else [model5.models[0]]
        modules6 = model6.models[0] if isinstance(model6.models[0], list) else [model6.models[0]]

        for module in modules4:
            self.assertIsInstance(module, nn.Module)
            for parameter in module.parameters():
                self.assertFalse(torch.isnan(parameter).any())
                self.assertFalse(torch.isinf(parameter).any())

        for module4, module5 in zip(modules4, modules5):
            for param4, param5 in zip(module4.parameters(), module5.parameters()):
                self.assertTrue(torch.equal(param4, param5))

        for module4, module6 in zip(modules4, modules6):
            for param4, param6 in zip(module4.parameters(), module6.parameters()):
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
            submodules = submodel if isinstance(submodel, list) else [submodel]
            for module in submodules:
                self.assertIsInstance(module, nn.Module)
                for parameter in module.parameters():
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
            submodules1 = submodel1 if isinstance(submodel1, list) else [submodel1]
            submodules8 = submodel8 if isinstance(submodel8, list) else [submodel8]
            for module1, module8 in zip(submodules1, submodules8):
                for param1, param8 in zip(module1.parameters(), module8.parameters()):
                    self.assertTrue(torch.equal(param1, param8))

        # Check that a custom torch model runs with multiple optimizers and random states as expected
        model9 = MLPRegressor(
            torch_model=ValidModel(n_latent=16),
            fit_encoder_intercept=None,
            fit_head_intercept=None,
            encoder_activation=None,
            head_activation=None,
            dropout_p=None,
            n_latent=None,
            normalization=None,
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
            normalization=None,
            optimizer=["AdamW", "SGD+mom"],
            random_state=[42, 43],
        ).fit(self.X, self.y)

        for submodel9, submodel10 in zip(model9.models, model10.models):
            submodules9 = submodel9 if isinstance(submodel9, list) else [submodel9]
            submodules10 = submodel10 if isinstance(submodel10, list) else [submodel10]
            for module9, module10 in zip(submodules9, submodules10):
                for param9, param10 in zip(module9.parameters(), module10.parameters()):
                    self.assertTrue(torch.equal(param9, param10))

    def test_valid_fit_refit(self):
        """
        Test that refitting produces different results.
        """
        model_refit = MLPRegressor(epochs = 5, patience = 2, refit=True).fit(self.X, self.y)
        model_norefit = MLPRegressor(epochs = 5, patience = 2, refit=False).fit(self.X, self.y)
        # Without refitting, each stored model is a list of validated models
        self.assertIsInstance(model_norefit.models[0], list)
        params_refit = [p.clone() for p in model_refit.models[0].parameters()]
        for validated_model in model_norefit.models[0]:
            params_norefit = [p.clone() for p in validated_model.parameters()]
            for p_norefit, p_refit in zip(params_norefit, params_refit):
                self.assertFalse(torch.equal(p_norefit, p_refit))

    def test_valid_fit_dollar_neutral(self):
        """
        Test that the dollar neutral constraint is applied correctly during training.
        """
        model = MLPRegressor(epochs = 5, patience = 2, signal_modifier = msl.LongShortModule(dollar_neutral = True)).fit(self.X.iloc[:, :1], pd.concat((self.X.iloc[:, :1], self.y), axis=1))
        # Without refitting, each stored model is a list of validated models
        self.assertIsInstance(model.models[0], list)
        for idx, validated_model in enumerate(model.models[0]):
            for param in validated_model.parameters():
                self.assertFalse(torch.isnan(param).any())
                self.assertFalse(torch.isinf(param).any())

            # Get forward pass outputs for the training data, scaled as in predict
            validated_model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(model.x_scalers[idx].transform(self.X.iloc[:, :1]), dtype=torch.float32)
                outputs = validated_model(X_tensor)

            # Check that the sum of the outputs across all samples is approximately zero (dollar neutral constraint)
            output_mean = outputs.sum(dim = 1)
            self.assertTrue(torch.allclose(output_mean, torch.zeros_like(output_mean), atol=1e-4))

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

    def test_valid_create_train_test_splits_(self):
        # Test that there is no leakage between train and validation sets in the time split and that the split is done according to the specified train_splitter
        model = MLPRegressor()
        X_tr, X_va, y_tr, y_va = model.create_train_valid_splits_(self.X, self.y, splitter=0.6)

        dates = sorted(self.X.index.get_level_values(1).unique())
        cut = int(0.6 * len(dates))
        train_dates = set(dates[:cut])
        val_dates = set(dates[cut:])

        self.assertEqual(set(X_tr[0].index.get_level_values("real_date").unique()), train_dates)
        self.assertEqual(set(X_va[0].index.get_level_values("real_date").unique()), val_dates)
        self.assertEqual(set(y_tr[0].index.get_level_values("real_date").unique()), train_dates)
        self.assertEqual(set(y_va[0].index.get_level_values("real_date").unique()), val_dates)

    def test_valid_scale_data_(self):
        # Test that the data is scaled correctly when both scalers are provided
        model = MLPRegressor()
        X_trs, X_vas, y_trs, y_vas = model.create_train_valid_splits_(self.X, self.y, splitter=0.6)
        X_tr, X_va, y_tr, y_va = X_trs[0], X_vas[0], y_trs[0], y_vas[0]
        X_trs_s, y_trs_s, X_vas_s, y_vas_s, _, _ = model.scale_data_(X_trains=X_trs, y_trains=y_trs, X_valids=X_vas, y_valids=y_vas, x_scaler=StandardScaler(with_mean=True), y_scaler=StandardScaler(with_mean=True))
        X_tr_scaled, y_tr_scaled, X_va_scaled, y_va_scaled = X_trs_s[0], y_trs_s[0], X_vas_s[0], y_vas_s[0]

        self.assertIsInstance(X_tr_scaled, np.ndarray)
        self.assertIsInstance(X_va_scaled, np.ndarray)
        self.assertIsInstance(y_tr_scaled, np.ndarray)
        self.assertIsInstance(y_va_scaled, np.ndarray)

        self.assertAlmostEqual(X_tr_scaled.mean(), 0, places=5)
        self.assertAlmostEqual(y_tr_scaled.mean(), 0, places=5)
        self.assertAlmostEqual(X_tr_scaled.std(), 1, places=5)
        self.assertAlmostEqual(y_tr_scaled.std(), 1, places=5)

        # Repeat for the case where the means are not removed (a common use case)
        X_trs_s, y_trs_s, X_vas_s, y_vas_s, _, _ = model.scale_data_(X_trains=X_trs, y_trains=y_trs, X_valids=X_vas, y_valids=y_vas, x_scaler=StandardScaler(with_mean=False), y_scaler=StandardScaler(with_mean=False))
        X_tr_scaled, y_tr_scaled, X_va_scaled, y_va_scaled = X_trs_s[0], y_trs_s[0], X_vas_s[0], y_vas_s[0]
        self.assertIsInstance(X_tr_scaled, np.ndarray)
        self.assertIsInstance(X_va_scaled, np.ndarray)
        self.assertIsInstance(y_tr_scaled, np.ndarray)
        self.assertIsInstance(y_va_scaled, np.ndarray)

        # Repeat for the case where X is scaled but y is not
        X_trs_s, y_trs_s, X_vas_s, y_vas_s, _, _ = model.scale_data_(X_trains=X_trs, y_trains=y_trs, X_valids=X_vas, y_valids=y_vas, x_scaler=StandardScaler(with_mean=False), y_scaler=None)
        X_tr_scaled, y_tr_scaled, X_va_scaled, y_va_scaled = X_trs_s[0], y_trs_s[0], X_vas_s[0], y_vas_s[0]
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
        X_trs, X_vas, y_trs, y_vas = sklearn_model.create_train_valid_splits_(self.X, self.y, splitter=0.6)
        X_trs_s, y_trs_s, X_vas_s, y_vas_s, _, _ = sklearn_model.scale_data_(X_trains=X_trs, y_trains=y_trs, X_valids=X_vas, y_valids=y_vas, x_scaler=StandardScaler(with_mean=True), y_scaler=StandardScaler(with_mean=True))
        X_tr_scaled, y_tr_scaled = X_trs_s[0], y_trs_s[0]

        torch_model = sklearn_model._fit_one_batch(
            torch_model,
            torch.Tensor(X_tr_scaled[:8]),
            torch.Tensor(y_tr_scaled[:8]),
            optimizer=optimizer,
            scheduler = None,
            loss_func = nn.MSELoss(),
            reg_turnover = 1
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
            reg_turnover = 1
        )
        after2 = {k: v.detach().clone() for k, v in torch_model.state_dict().items()}
        changed2 = any(not torch.equal(after[k], after2[k]) for k in after.keys())
        self.assertTrue(changed2)
