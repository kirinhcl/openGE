"""Evaluation metrics for crop trait prediction."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R-squared value
    """
    return r2_score(y_true, y_pred)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient and p-value.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Correlation coefficient and p-value
    """
    from scipy.stats import pearsonr
    return pearsonr(y_true, y_pred)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with all metrics
    """
    return {
        "rmse": calculate_rmse(y_true, y_pred),
        "r2": calculate_r2(y_true, y_pred),
        "mae": calculate_mae(y_true, y_pred),
        "pearson_r": calculate_pearson_correlation(y_true, y_pred)[0],
    }
