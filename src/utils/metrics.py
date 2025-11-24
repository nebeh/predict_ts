"""
Модуль для расчета метрик качества прогнозирования
"""
import numpy as np
from typing import Dict


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error"""
    # Исключаем очень маленькие значения, чтобы избежать деления на ноль и огромных ошибок
    # Используем порог 1% от среднего значения для фильтрации
    threshold = np.abs(y_true).mean() * 0.01
    mask = np.abs(y_true) > threshold
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Исключаем очень маленькие значения, чтобы избежать огромных ошибок
    threshold = np.abs(denominator).mean() * 0.01
    mask = denominator > threshold
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, final_day_only: bool = True) -> Dict[str, float]:
    """
    Вычисляет все метрики для прогноза
    
    Args:
        y_true: реальные значения
        y_pred: предсказанные значения
        final_day_only: если True, вычисляет метрики только для последнего дня (для кумулятивной суммы)
        
    Returns:
        словарь с метриками
    """
    if final_day_only:
        # Вычисляем метрики только для последнего дня (30-й день)
        # Для кумулятивной суммы нас интересует итоговое значение на 30-й день
        # Преобразуем в numpy массивы если нужно
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Берем последнее значение
        y_true_final = y_true[-1] if y_true.ndim > 0 and len(y_true) > 0 else float(y_true)
        y_pred_final = y_pred[-1] if y_pred.ndim > 0 and len(y_pred) > 0 else float(y_pred)
        
        # Для скалярных значений вычисляем метрики напрямую
        mae_val = np.abs(y_true_final - y_pred_final)
        rmse_val = np.sqrt((y_true_final - y_pred_final) ** 2)
        
        # MAPE и sMAPE для последнего дня
        if np.abs(y_true_final) > 1e-8:
            mape_val = np.abs((y_true_final - y_pred_final) / y_true_final) * 100
        else:
            mape_val = np.nan
        
        denominator = (np.abs(y_true_final) + np.abs(y_pred_final)) / 2
        if denominator > 1e-8:
            smape_val = np.abs(y_true_final - y_pred_final) / denominator * 100
        else:
            smape_val = np.nan
        
        return {
            'MAE': float(mae_val),
            'RMSE': float(rmse_val),
            'MAPE': float(mape_val) if not np.isnan(mape_val) else np.nan,
            'sMAPE': float(smape_val) if not np.isnan(smape_val) else np.nan
        }
    else:
        # Старый способ: вычисляем метрики для всех дней
        return {
            'MAE': mae(y_true, y_pred),
            'RMSE': rmse(y_true, y_pred),
            'MAPE': mape(y_true, y_pred),
            'sMAPE': smape(y_true, y_pred)
        }

