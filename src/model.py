import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def train_random_forest(x_train, y_train) -> RandomForestRegressor:
    model = RandomForestRegressor(random_state=42)
    model.fit(x_train, y_train)

    return model


def test_random_forest(model: RandomForestRegressor, x_test, y_test):
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)

    return mae


def top_features_random_forest(model: RandomForestRegressor, cols: list, top: int) -> list:
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[-top:][::-1]

    top_features = pd.DataFrame()
    top_features['feature'] = [cols[i] for i in sorted_idx]
    top_features['importance'] = importances[sorted_idx]

    return top_features



def train_gradient_boosting(x_train, y_train) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(random_state=42)
    model.fit(x_train, y_train)
    return model


def test_gradient_boosting(model: GradientBoostingRegressor, x_test, y_test):
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae


def top_features_gradient_boosting(model: GradientBoostingRegressor, cols: list, top: int) -> list:
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[-top:][::-1]

    top_features = pd.DataFrame()
    top_features['feature'] = [cols[i] for i in sorted_idx]
    top_features['importance'] = importances[sorted_idx]

    return top_features
