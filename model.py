import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def train_test_split_data(X, y, test_size=0.2, random_state=42) -> tuple:
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_regression(X_train, y_train):
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=200, max_depth=15, 
                        min_samples_split=5, min_samples_leaf=2, random_state=42):
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features='sqrt',
        n_jobs=-1,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train, n_estimators=200, learning_rate=0.1,
                            max_depth=5, min_samples_split=5, min_samples_leaf=2,
                            subsample=0.8, random_state=42):
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, n_estimators=200, learning_rate=0.1,
                  max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42):
    
    if not XGBOOST_AVAILABLE:
        return None
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=random_state,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

def train_extra_trees(X_train, y_train, n_estimators=200, max_depth=15,
                      min_samples_split=5, min_samples_leaf=2, random_state=42):
    
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features='sqrt',
        n_jobs=-1,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train, alpha=1.0):
    
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, X_train=None, y_train=None, cv_folds=5) -> dict:
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    result = {
        'R² Score': round(r2, 4),
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2)
    }
    
    if X_train is not None and y_train is not None:
        try:
            X_full = np.vstack([X_train, X_test]) if hasattr(X_train, 'values') is False else pd.concat([X_train, X_test])
            y_full = np.concatenate([y_train, y_test]) if hasattr(y_train, 'values') is False else pd.concat([y_train, y_test])
            
            cv_scores = cross_val_score(model, X_full, y_full, cv=cv_folds, scoring='r2')
            result['CV R² (mean)'] = round(cv_scores.mean(), 4)
            result['CV R² (std)'] = round(cv_scores.std(), 4)
        except Exception:
            pass
    
    return result

def train_all_models(X_train, y_train) -> dict:
    
    models = {
        'Linear Regression': train_linear_regression(X_train, y_train),
        'Ridge Regression': train_ridge_regression(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'Extra Trees': train_extra_trees(X_train, y_train),
        'Gradient Boosting': train_gradient_boosting(X_train, y_train),
    }
    
    if XGBOOST_AVAILABLE:
        xgb_model = train_xgboost(X_train, y_train)
        if xgb_model is not None:
            models['XGBoost'] = xgb_model
    
    return models

def evaluate_all_models(models: dict, X_test, y_test, X_train=None, y_train=None) -> pd.DataFrame:
    
    results = []
    
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, X_train, y_train)
        metrics['Model'] = model_name
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    base_cols = ['Model', 'R² Score', 'MAE', 'RMSE']
    cv_cols = [col for col in ['CV R² (mean)', 'CV R² (std)'] if col in results_df.columns]
    results_df = results_df[base_cols + cv_cols]
    
    results_df = results_df.sort_values('R² Score', ascending=False).reset_index(drop=True)
    
    return results_df

def select_best_model(models: dict, results_df: pd.DataFrame) -> tuple:
    
    best_idx = results_df['R² Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_r2_score = results_df.loc[best_idx, 'R² Score']
    best_model = models[best_model_name]
    
    return best_model_name, best_model, best_r2_score

def predict_yield(model, crop_encoder, season_encoder, crop, season, rainfall, temperature, area) -> float:
    
    crop_encoded = crop_encoder.transform([crop])[0]
    season_encoded = season_encoder.transform([season])[0]
    
    features = np.array([[crop_encoded, season_encoded, rainfall, temperature, area]])
    
    prediction = model.predict(features)[0]
    
    return round(prediction, 2)

def train_and_get_best_model(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    
    models = train_all_models(X_train, y_train)
    
    results_df = evaluate_all_models(models, X_test, y_test, X_train, y_train)
    
    best_model_name, best_model, _ = select_best_model(models, results_df)
    
    return best_model, best_model_name, results_df, models
