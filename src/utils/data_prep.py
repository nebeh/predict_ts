"""
Модуль для подготовки данных для временных рядов
"""
import numpy as np
from typing import Tuple


def to_cumulative(data: np.ndarray, start_value: float = 0.0) -> np.ndarray:
    """
    Преобразует данные в кумулятивную сумму
    
    Args:
        data: массив данных
        start_value: начальное значение для кумулятивной суммы (по умолчанию 0)
        
    Returns:
        кумулятивная сумма
    """
    return start_value + np.cumsum(data)


def prepare_train_test(data: np.ndarray, forecast_horizon: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Разделяет данные на обучающую и тестовую выборки
    
    Args:
        data: полный массив данных
        forecast_horizon: горизонт прогнозирования
        
    Returns:
        train_data, test_data
    """
    split_idx = len(data) - forecast_horizon
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data


def prepare_sequences(data: np.ndarray, lookback: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подготавливает последовательности для обучения моделей
    
    Args:
        data: временной ряд
        lookback: количество точек истории для предсказания
        forecast_horizon: горизонт прогнозирования
        
    Returns:
        X (история), y (целевые значения)
    """
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + forecast_horizon])
    return np.array(X), np.array(y)

