from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge

def get_regression_model(model_name: str):
    """Return (base_model, param_grid) for regression model selection."""
    model_name = model_name.strip().lower()

    # --- LightGBM ---
    if model_name == "lgb":
        base_model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10,
            random_state=42,
            force_row_wise=True
        )
        param_grid = {
            'n_estimators': [100],
            'max_depth': [3],
            'learning_rate': [0.1],
            'subsample': [1],
            'colsample_bytree': [1],
            'num_leaves': [7],
            'min_data_in_leaf': [5, 10]
        }

    # --- XGBoost ---
    elif model_name == "xgboost":
        base_model = XGBRegressor(objective='reg:squarederror', random_state=42)
        param_grid = {
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'subsample': [0.8, 1.0]
        }

    # --- CatBoost ---
    elif model_name == "catboost":
        base_model = CatBoostRegressor(verbose=0, random_state=42)
        param_grid = {
            'depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1],
            'iterations': [200, 400]
        }

    # --- Random Forest ---
    elif model_name == "randomforest":
        base_model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5]
        }

    # --- Gradient Boosting ---
    elif model_name == "gradientboosting":
        base_model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }

    # --- Extra Trees ---
    elif model_name == "extratrees":
        base_model = ExtraTreesRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None]
        }

    # --- Linear Models ---
    elif model_name == "linearregression":
        base_model = LinearRegression()
        param_grid = {}

    elif model_name == "ridge":
        base_model = Ridge()
        param_grid = {'alpha': [0.1, 1.0, 10.0]}

    elif model_name == "lasso":
        base_model = Lasso()
        param_grid = {'alpha': [0.01, 0.1, 1.0]}

    elif model_name == "elasticnet":
        base_model = ElasticNet()
        param_grid = {'alpha': [0.1, 1.0], 'l1_ratio': [0.3, 0.5, 0.7]}

    elif model_name == "bayesianridge":
        base_model = BayesianRidge()
        param_grid = {}

    # --- SVM ---
    elif model_name == "svr":
        base_model = SVR()
        param_grid = {'kernel': ['rbf', 'linear'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}

    # --- Neural Network ---
    elif model_name == "mlp":
        base_model = MLPRegressor(max_iter=500, random_state=42)
        param_grid = {
            'hidden_layer_sizes': [(64, 32), (128, 64)],
            'learning_rate_init': [0.001, 0.01]
        }

    # --- Decision Tree ---
    elif model_name == "decisiontree":
        base_model = DecisionTreeRegressor(random_state=42)
        param_grid = {
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]
        }

    # --- Kernel Ridge ---
    elif model_name == "kernelridge":
        base_model = KernelRidge()
        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf']
        }

    # --- Gaussian Process ---
    elif model_name == "gaussianprocess":
        base_model = GaussianProcessRegressor(random_state=42)
        param_grid = {}

    else:
        raise ValueError(f"❌ Unsupported model name: {model_name}")

    return base_model, param_grid
