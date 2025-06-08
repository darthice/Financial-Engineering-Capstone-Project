import pandas as pd
import numpy as np
import yfinance as yf
# Loading the technical indicators (features)
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score, accuracy_score

# Loading ML models
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Loading DL models
from tensorflow import keras
from keras.layers import Dense, LSTM, Conv1D, Flatten
from keras.models import Sequential, load_model

from sklearn.metrics import accuracy_score







# Class for storing the data
class DataStore():
    """
       The DataStore class retrieves the assets (tickers) prices from Yahoo Finance. It calculates and stores
       prices and features for tickers  
    """
    def __init__(self, tickers=[], start_date: str='2010-01-01', end_date: str='2025-04-20', period: str='1d', caps=dict()):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        # retrieve price and return data
        self.data = self.get_data()
        # calculate features and target variable
        self.features_target = self.getfeatures()
        self.caps = caps

    def get_data(self):
        df = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            interval=self.period
        )['Close'] 

        df = df.dropna()
        returns = np.log(df).diff()
        returns = returns.dropna()
        return df, returns
    
    def getfeatures(self, future_shift: int = 5):
        """ 
            The getfeatures method calculates features for ML modeling purposes
            Arg:
                - future_shift (int): how many periods to predict towards the furture  
            --------------
            Returns the features in dataframe format.
        """
        prices = self.data[0]
        returns = self.data[1]

        # Initializing the standard scaler
        # scaler = StandardScaler()

        feature_functions = {
            "daily_return": lambda ticker: np.log(prices[ticker]).diff(),
            "weekly_return": lambda ticker: returns[ticker].rolling(5).apply(lambda a: 100 * (np.prod(1+a/100)-1)),
            "monthly_return": lambda ticker: returns[ticker].rolling(30).apply(lambda a: 100 * (np.prod(1+a/100)-1)),
            "sma_chg": lambda ticker: SMAIndicator(prices[ticker], window=50).sma_indicator().diff(),
            "ema_chg": lambda ticker: EMAIndicator(prices[ticker], window=50).ema_indicator().diff(),
            "rsi_chg": lambda ticker: RSIIndicator(prices[ticker], window=14).rsi().diff(),
            "macd_chg": lambda ticker: MACD(prices[ticker], window_slow=26, window_fast=12, window_sign=9).macd().diff(),
            "target": lambda ticker: np.log(prices[ticker]).diff().shift(-future_shift)
        }
        features = pd.DataFrame()
        feat_tuples = []

        # Calculate each feature for every ticker
        for ticker in self.tickers:
            for feature in feature_functions.keys():
                features[(ticker, feature)] = feature_functions[feature](ticker)
                feat_tuples.append((ticker, feature))

        # Set column multi-index and drop rows with any missing values
        features.columns = pd.MultiIndex.from_tuples(feat_tuples) 
        features = features.dropna()

        return features



# Class for forecasting the future price - DesicionTreeRegressor
class DesicionTreeRegressorPredictor():
    def __init__(self, ticker: str, start_date: str, end_date: str, features_target: pd.DataFrame):
        """ 
            DesicionTreeRegressor model makes a price forecast for a ticker by 'foracast_shift' period

            Args:
                - ticker (str): ticker of the asset for which the model predicts the future price
                - start_date (str): start date of dataset
                - end_date (str): end date of dataset
                - features_target (pd.DataFrame): dataframe that contains the features and the target columns
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date 
        # Initialize Decision Tree model
        self.predictor = DecisionTreeRegressor(random_state=42)
        self.data = features_target
        # Extract features and target columns for this ticker
        self.features = features_target[self.ticker].iloc[:, :-1]
        self.target = features_target[self.ticker]["target"]
        self.X_train = self.split()[0]
        self.X_test = self.split()[1]
        self.y_train = self.split()[2]
        self.y_test = self.split()[3]

    def split(self, test_size: float = 0.1):
        """ 
            The function splits the dataset into training and testing sets
            Args:
                test_size (float): the size of dataset for testing (by default = 10%)
            ---------    
            Returns the splitted dataset (training and testing features and target)
        """
        X_df = self.features
        y_df = self.target

        # Splitting the dataset into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_df, test_size=test_size, random_state=42, shuffle=False
        )    

        return X_train, X_test, y_train, y_test

    def make_forecast(self):
        """ 
        The function fits the model on training set
        -----------
        Return: the forecast of price one step ahead, accuracy score on training periond, accuracy score on testing periond 
        """
        # Fitting the model
        self.predictor.fit(self.X_train, self.y_train)
        # Prediction for testing period
        y_pred = self.predictor.predict(self.X_test)
        # Accuracy scores
        acc_score_train = round(self.predictor.score(self.X_train, self.y_train)*100, 2)
        r2_score_train = r2_score(self.y_train, self.predictor.predict(self.X_train))
        r2_score_test = r2_score(self.y_test, y_pred)

        # Return forecast (last prediction) and metrics
        return y_pred[-1], acc_score_train, r2_score_train, r2_score_test



class LinearRegressionPredictor():
    def __init__(self, ticker: str, start_date: str, end_date: str, features_target: pd.DataFrame):
        """ 
            LinearRegression model makes a price forecast for a ticker by 'foracast_shift' period

            Args:
                - ticker (str): ticker of the asset for which the model predicts the future price
                - start_date (str): start date of dataset
                - end_date (str): end date of dataset
                - features_target (pd.DataFrame): dataframe that contains the features and the target columns
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date 
        self.predictor = LinearRegression()
        self.data = features_target
        self.features = features_target[self.ticker].iloc[:, :-1]
        self.target = features_target[self.ticker]["target"]
        self.X_train = self.split()[0]
        self.X_test = self.split()[1]
        self.y_train = self.split()[2]
        self.y_test = self.split()[3]

    def split(self, test_size: float = 0.1):
        """ 
            The function splits the dataset into training and testing sets
            Args:
                test_size (float): the size of dataset for testing (by default = 10%)
            ---------    
            Returns the splitted dataset (training and testing features and target)
        """
        X_df = self.features
        y_df = self.target

        # Splitting the dataset into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_df, test_size=test_size, random_state=42, shuffle=False
        )    

        return X_train, X_test, y_train, y_test

    def make_forecast(self):
        """ 
        The function fits the model on training set
        -----------
        Return: the forecast of price one step ahead, accuracy score on training periond, accuracy score on testing periond 
        """
        # Fitting the model
        self.predictor.fit(self.X_train, self.y_train)
        # Prediction for testing period
        y_pred = self.predictor.predict(self.X_test)
        # Accuracy scores
        acc_score_train = round(self.predictor.score(self.X_train, self.y_train)*100, 2)
        r2_score_train = r2_score(self.y_train, self.predictor.predict(self.X_train))
        r2_score_test = r2_score(self.y_test, y_pred)

        # Return forecast (last prediction) and metrics
        return y_pred[-1], acc_score_train, r2_score_train, r2_score_test




class SVRPredictor():
    def __init__(self, ticker: str, start_date: str, end_date: str, features_target: pd.DataFrame):
        """ 
            SVR model makes a price forecast for a ticker by 'foracast_shift' period

            Args:
                - ticker (str): ticker of the asset for which the model predicts the future price
                - start_date (str): start date of dataset
                - end_date (str): end date of dataset
                - features_target (pd.DataFrame): dataframe that contains the features and the target columns
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date 
        self.predictor = SVR()
        self.data = features_target
        self.features = features_target[self.ticker].iloc[:, :-1]
        self.target = features_target[self.ticker]["target"]
        self.X_train = self.split()[0]
        self.X_test = self.split()[1]
        self.y_train = self.split()[2]
        self.y_test = self.split()[3]

    def split(self, test_size: float = 0.1):
        """ 
            The function splits the dataset into training and testing sets
            Args:
                test_size (float): the size of dataset for testing (by default = 10%)
            ---------    
            Returns the splitted dataset (training and testing features and target)
        """
        X_df = self.features
        y_df = self.target

        # Splitting the dataset into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_df, test_size=test_size, random_state=42, shuffle=False
        )    

        return X_train, X_test, y_train, y_test

    def make_forecast(self):
        """ 
        The function fits the model on training set
        -----------
        Return: the forecast of price one step ahead, accuracy score on training periond, accuracy score on testing periond 
        """
        # Fitting the model
        self.predictor.fit(self.X_train, self.y_train)
        # Prediction for testing period
        y_pred = self.predictor.predict(self.X_test)
        # Accuracy scores
        acc_score_train = round(self.predictor.score(self.X_train, self.y_train)*100, 2)
        r2_score_train = r2_score(self.y_train, self.predictor.predict(self.X_train))
        r2_score_test = r2_score(self.y_test, y_pred)

        # Return forecast (last prediction) and metrics
        return y_pred[-1], acc_score_train, r2_score_train, r2_score_test


class LSTMPredictor:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        features_target: pd.DataFrame,
        test_size: float = 0.1,
        random_state: int = 42,
        epochs: int = 2,
        batch_size: int = 32
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date 
        self.data = features_target

        # split once
        X_train, X_test, y_train, y_test = self._split(
            features_target[self.ticker], test_size, random_state
        )

        # scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # reshape for LSTM (samples, time_steps, features)
        self.X_train = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
        self.X_test  = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
        self.y_train = y_train
        self.y_test  = y_test

        # build model
        self.model = Sequential([
            LSTM(50, input_shape=(1, X_train_s.shape[1])),
            Dense(1)
        ])
        self.model.compile(optimizer="adam", loss="mse")

        self.epochs = epochs
        self.batch_size = batch_size

    def _split(self, df: pd.DataFrame, test_size: float, random_state: int):
        X = df.drop(columns="target").values
        y = df["target"].values
        return train_test_split(
            X, y,
            test_size=test_size,
            shuffle=False,
            random_state=random_state
        )

    def make_forecast(self):
        # train
        self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )

        # predict
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test  = self.model.predict(self.X_test)

        # metrics
        r2_tr = r2_score(self.y_train, y_pred_train)
        r2_te = r2_score(self.y_test,  y_pred_test)
        last_pred = float(y_pred_test[-1])
        # Return forecast (last prediction) and metrics
        return (
            last_pred,            # your next-step forecast
            round(r2_tr * 100, 2),# R² on train in %
            r2_tr,                # raw R² train
            r2_te                 # raw R² test
        )


class CNNPredictor:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        features_target: pd.DataFrame,
        test_size: float = 0.1,
        random_state: int = 42,
        epochs: int = 2,
        batch_size: int = 32
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date 
        self.data = features_target

        # split once
        X_train, X_test, y_train, y_test = self._split(
            features_target[self.ticker], test_size, random_state
        )

        # scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # reshape for Conv1D (samples, time_steps, features)
        self.X_train = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
        self.X_test  = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
        self.y_train = y_train
        self.y_test  = y_test

        # build model
        self.model = Sequential([
            Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(1, X_train_s.shape[1])),
            Flatten(),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer="adam", loss="mse")

        self.epochs = epochs
        self.batch_size = batch_size

    def _split(self, df: pd.DataFrame, test_size: float, random_state: int):
        X = df.drop(columns="target").values
        y = df["target"].values
        return train_test_split(
            X, y,
            test_size=test_size,
            shuffle=False,
            random_state=random_state
        )

    def make_forecast(self):
        # train
        self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )

        # predict
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test  = self.model.predict(self.X_test)

        # metrics
        r2_tr = r2_score(self.y_train, y_pred_train)
        r2_te = r2_score(self.y_test,  y_pred_test)
        last_pred = float(y_pred_test[-1])
        # Return forecast (last prediction) and metrics
        return (
            last_pred,            # your next-step forecast
            round(r2_tr * 100, 2),# R² on train in %
            r2_tr,                # raw R² train
            r2_te                 # raw R² test
        )





class TransformerPredictor:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        features_target: pd.DataFrame,
        test_size: float = 0.1,
        random_state: int = 42,
        epochs: int = 2,
        batch_size: int = 32
    ):
        from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Flatten
        from tensorflow.keras.models import Model

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = features_target

        X_train, X_test, y_train, y_test = self._split(
            features_target[self.ticker], test_size, random_state
        )

        # Standardize features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Transformer block
        input_layer = Input(shape=(X_train_s.shape[1], 1))
        x = MultiHeadAttention(num_heads=2, key_dim=8)(input_layer, input_layer)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(1)(x)
        self.model = Model(inputs=input_layer, outputs=x)
        self.model.compile(optimizer="adam", loss="mse")

        # reshape for transformer (samples, timesteps, features)
        self.X_train = X_train_s.reshape((X_train_s.shape[0], X_train_s.shape[1], 1))
        self.X_test = X_test_s.reshape((X_test_s.shape[0], X_test_s.shape[1], 1))
        self.y_train = y_train
        self.y_test = y_test

        self.epochs = epochs
        self.batch_size = batch_size

    def _split(self, df: pd.DataFrame, test_size: float, random_state: int):
        X = df.drop(columns="target").values
        y = df["target"].values
        return train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)

    def make_forecast(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        r2_tr = r2_score(self.y_train, y_pred_train)
        r2_te = r2_score(self.y_test, y_pred_test)
        last_pred = float(y_pred_test[-1])
        # Return forecast (last prediction) and metrics
        return (last_pred, round(r2_tr * 100, 2), r2_tr, r2_te)



class XGBoostPredictor:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        features_target: pd.DataFrame,
        test_size: float = 0.1,
        random_state: int = 42,

    ):
        import xgboost as xgb

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = features_target

        X_train, X_test, y_train, y_test = self._split(
            features_target[self.ticker], test_size, random_state
        )

        # Standardize features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        self.model = xgb.XGBRegressor(random_state=random_state, n_estimators=100)

    def _split(self, df: pd.DataFrame, test_size: float, random_state: int):
        X = df.drop(columns="target").values
        y = df["target"].values
        return train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)

    def make_forecast(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        r2_tr = r2_score(self.y_train, y_pred_train)
        r2_te = r2_score(self.y_test, y_pred_test)
        last_pred = float(y_pred_test[-1])
        # Return forecast (last prediction) and metrics
        return (last_pred, round(r2_tr * 100, 2), r2_tr, r2_te)



class MLPPredictor:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        features_target: pd.DataFrame,
        test_size: float = 0.1,
        random_state: int = 42,
        epochs: int = 2,
        batch_size: int = 32
    ):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = features_target

        X_train, X_test, y_train, y_test = self._split(
            features_target[self.ticker], test_size, random_state
        )
        # Standardize features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        self.X_train = X_train_s
        self.X_test = X_test_s
        self.y_train = y_train
        self.y_test = y_test

        self.model = Sequential([
            Dense(64, activation="relu", input_shape=(X_train_s.shape[1],)),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        self.model.compile(optimizer="adam", loss="mse")
        self.epochs = epochs
        self.batch_size = batch_size

    def _split(self, df: pd.DataFrame, test_size: float, random_state: int):
        X = df.drop(columns="target").values
        y = df["target"].values
        return train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)

    def make_forecast(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        r2_tr = r2_score(self.y_train, y_pred_train)
        r2_te = r2_score(self.y_test, y_pred_test)
        last_pred = float(y_pred_test[-1])
        # Return forecast (last prediction) and metrics
        return (last_pred, round(r2_tr * 100, 2), r2_tr, r2_te)


class RidgeRegressionPredictor:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        features_target: pd.DataFrame,
        test_size: float = 0.1,
        random_state: int = 42,

    ):
        from sklearn.linear_model import Ridge

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = features_target
        # Train-test split
        X_train, X_test, y_train, y_test = self._split(
            features_target[self.ticker], test_size, random_state
        )
        # Standardize features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        # Initialize Ridge regression model
        self.model = Ridge(alpha=1.0)

    def _split(self, df: pd.DataFrame, test_size: float, random_state: int):
        X = df.drop(columns="target").values
        y = df["target"].values
        return train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)

    def make_forecast(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        r2_tr = r2_score(self.y_train, y_pred_train)
        r2_te = r2_score(self.y_test, y_pred_test)
        last_pred = float(y_pred_test[-1])
        # Return forecast (last prediction) and metrics
        return (last_pred, round(r2_tr * 100, 2), r2_tr, r2_te)





class WeightedAverageEnsemble:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        features_target: pd.DataFrame,
        test_size: float = 0.1,
        random_state: int = 42,
        epochs: int = 1, 
        batch_size: int = 32  
    ):
        # Ensemble of: Ridge, XGBoost, MLP
        from sklearn.linear_model import Ridge
        import xgboost as xgb
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = features_target
        # Train-test split
        X_train, X_test, y_train, y_test = self._split(
            features_target[self.ticker], test_size, random_state
        )
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        # Model initialization
        self.ridge = Ridge(alpha=1.0)
        self.xgb = xgb.XGBRegressor(random_state=random_state, n_estimators=100)
        self.mlp = Sequential([
            Dense(64, activation="relu", input_shape=(self.X_train.shape[1],)),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        self.mlp.compile(optimizer="adam", loss="mse")

        # Weights for ensemble: [ridge, xgb, mlp]
        self.weights = [0.3, 0.3, 0.4]
        self.epochs = epochs
        self.batch_size = batch_size

    def _split(self, df: pd.DataFrame, test_size: float, random_state: int):
        X = df.drop(columns="target").values
        y = df["target"].values
        return train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)

    def make_forecast(self):
        # Train all models
        self.ridge.fit(self.X_train, self.y_train)
        self.xgb.fit(self.X_train, self.y_train)
        self.mlp.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        # Generate predictions for train and test sets
        pred_train = [
            self.ridge.predict(self.X_train),
            self.xgb.predict(self.X_train),
            self.mlp.predict(self.X_train).flatten()
        ]
        pred_test = [
            self.ridge.predict(self.X_test),
            self.xgb.predict(self.X_test),
            self.mlp.predict(self.X_test).flatten()
        ]
        # Weighted average of model predictions
        y_pred_train = sum(w * p for w, p in zip(self.weights, pred_train))
        y_pred_test = sum(w * p for w, p in zip(self.weights, pred_test))
        
        # Compute performance metrics
        r2_tr = r2_score(self.y_train, y_pred_train)
        r2_te = r2_score(self.y_test, y_pred_test)
        last_pred = float(y_pred_test[-1])
        # Return forecast (last prediction) and metrics
        return (last_pred, round(r2_tr * 100, 2), r2_tr, r2_te)
