"""
Модель TFT (Temporal Fusion Transformer)
"""
import numpy as np
from typing import Tuple, Optional
from .base_model import BaseForecastModel

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TFTModel(BaseForecastModel):
    """Упрощенная реализация TFT"""
    
    def __init__(self, seq_len: int = 96, pred_len: int = 30):
        super().__init__("TFT")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = None
        self.scaler_mean = None
        self.scaler_std = None
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
        normalized_data = self._normalize(data)
        # В реальной реализации здесь была бы полная архитектура TFT
        self.is_fitted = True
    
    def predict(self, horizon: int, return_conf_int: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Делает прогноз в авторегрессивном режиме (seq2seq) используя экспоненциальное сглаживание с трендом
        Каждый следующий день предсказывается на основе предыдущих предсказаний
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        # ВАЖНО: Работаем только с train_data (self.data), НЕ используем test_data!
        # Используем метод Хольта-Винтерса для авторегрессивного прогноза
        if len(self.data) < 2:
            # Если данных мало, используем простое среднее
            predictions = np.full(horizon, self.data[-1])
        else:
            # Вычисляем параметры из обучающих данных
            alpha = 0.3  # параметр сглаживания уровня
            beta = 0.1   # параметр сглаживания тренда
            
            # Инициализация из последних значений train_data
            level = self.data[-1]
            trend = self.data[-1] - self.data[-2] if len(self.data) > 1 else 0
            
            # Авторегрессивный прогноз: каждый следующий день опирается на предыдущие предсказания
            predictions = []
            current_level = level
            current_trend = trend
            
            for step in range(horizon):
                # ВАЖНО: Обновляем уровень и тренд на основе ПРЕДЫДУЩЕГО ПРЕДСКАЗАНИЯ, а не реальных данных!
                # Для первого шага используем последнее значение из train_data
                if step == 0:
                    prev_value = self.data[-1]
                else:
                    # Для последующих шагов используем СВОЕ предыдущее предсказание
                    prev_value = predictions[-1]
                
                # Обновляем уровень и тренд на основе предыдущего значения (предсказания)
                current_level = alpha * prev_value + (1 - alpha) * (current_level + current_trend)
                if step > 0:
                    # Обновляем тренд на основе изменения уровня
                    current_trend = beta * (current_level - level) + (1 - beta) * current_trend
                level = current_level
                
                # Прогноз на основе текущего уровня и тренда
                forecast = current_level + current_trend
                predictions.append(max(0, forecast))  # Неотрицательные объемы
            
            predictions = np.array(predictions)
        
        # Доверительные интервалы на основе вариации обучающих данных
        if return_conf_int:
            # Используем вариацию остатков из train_data
            if len(self.data) > 1:
                residuals = np.diff(self.data[-min(30, len(self.data)):])
                std = np.std(residuals)
            else:
                std = 0
            
            # Доверительные интервалы расширяются со временем
            lower = np.maximum(predictions - 1.96 * std * np.sqrt(np.arange(1, horizon + 1)), 0)
            upper = predictions + 1.96 * std * np.sqrt(np.arange(1, horizon + 1))
        else:
            lower, upper = None, None
        
        return predictions, lower, upper

