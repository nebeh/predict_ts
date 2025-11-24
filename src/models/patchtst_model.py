"""
Модель PatchTST
"""
import numpy as np
from typing import Tuple, Optional
from .base_model import BaseForecastModel

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PatchTSTModel(BaseForecastModel):
    """Упрощенная реализация PatchTST"""
    
    def __init__(self, seq_len: int = 96, pred_len: int = 30, d_model: int = 128):
        super().__init__("PatchTST")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.data = None
        if not TORCH_AVAILABLE:
            print("Предупреждение: PyTorch не установлен. Будет использован упрощенный метод.")
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Нормализация данных"""
        if self.scaler_mean is None:
            self.scaler_mean = np.mean(data)
            self.scaler_std = np.std(data) + 1e-8
        return (data - self.scaler_mean) / self.scaler_std
    
    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Денормализация данных"""
        return data * self.scaler_std + self.scaler_mean
    
    def fit(self, data: np.ndarray) -> None:
        """Обучает простую модель на основе последних значений"""
        self.data = data.copy()
        # Нормализуем данные
        normalized_data = self._normalize(data)
        
        # Простая модель: используем среднее приращение
        # В реальной реализации здесь была бы полная архитектура PatchTST
        self.is_fitted = True
    
    def predict(self, horizon: int, return_conf_int: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Делает прогноз в авторегрессивном режиме (seq2seq)
        Каждый следующий день предсказывается на основе предыдущих предсказаний
        В реальной реализации здесь была бы полная модель PatchTST
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        # ВАЖНО: Работаем только с train_data (self.data), НЕ используем test_data!
        # Вычисляем среднее приращение из обучающих данных
        if len(self.data) > 1:
            diffs = np.diff(self.data[-min(30, len(self.data)):])
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
        else:
            mean_diff = 0
            std_diff = 0
        
        # Авторегрессивный прогноз: каждый следующий день опирается на предыдущие предсказания
        predictions = []
        current = self.data[-1]  # Начинаем с последнего значения из train_data
        
        for step in range(horizon):
            # Предсказываем следующий день на основе текущего значения
            # Используем среднее приращение из обучающих данных
            current += mean_diff
            predictions.append(current)
            # ВАЖНО: Для следующего шага используем СВОЕ предсказание, а не реальные данные!
        
        predictions = np.array(predictions)
        
        # Доверительные интервалы расширяются со временем
        if return_conf_int:
            lower = np.maximum(predictions - 1.96 * std_diff * np.sqrt(np.arange(1, horizon + 1)), 0)
            upper = predictions + 1.96 * std_diff * np.sqrt(np.arange(1, horizon + 1))
        else:
            lower, upper = None, None
        
        return predictions, lower, upper

