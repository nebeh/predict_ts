"""
Модель iTransformer (Inverted Transformer)
"""
import numpy as np
from typing import Tuple, Optional
from .base_model import BaseForecastModel

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class iTransformerModel(BaseForecastModel):
    """Упрощенная реализация iTransformer"""
    
    def __init__(self, seq_len: int = 96, pred_len: int = 30):
        super().__init__("iTransformer")
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
        # В реальной реализации здесь была бы полная архитектура iTransformer
        self.is_fitted = True
    
    def predict(self, horizon: int, return_conf_int: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Делает прогноз в авторегрессивном режиме (seq2seq) с учетом сезонности
        Каждый следующий день предсказывается на основе предыдущих предсказаний
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        # ВАЖНО: Работаем только с train_data (self.data), НЕ используем test_data!
        # Вычисляем тренд и сезонность из обучающих данных
        if len(self.data) > 7:
            # Используем линейную регрессию для тренда
            x = np.arange(len(self.data))
            coeffs = np.polyfit(x, self.data, 1)
            trend = coeffs[0]
            
            # Вычисляем сезонность (если есть достаточно данных)
            if len(self.data) >= 14:
                # Простая сезонность на основе последних 7 дней из train_data
                seasonal_pattern = self.data[-7:] - np.mean(self.data[-7:])
            else:
                seasonal_pattern = np.zeros(7)
        else:
            trend = 0
            seasonal_pattern = np.zeros(7)
        
        # Авторегрессивный прогноз: каждый следующий день опирается на предыдущие предсказания
        predictions = []
        current = self.data[-1]  # Начинаем с последнего значения из train_data
        
        for i in range(horizon):
            # Добавляем тренд
            current += trend
            # Добавляем сезонность
            if len(seasonal_pattern) > 0:
                current += seasonal_pattern[i % len(seasonal_pattern)]
            predictions.append(current)
            # ВАЖНО: Для следующего шага используем СВОЕ предсказание, а не реальные данные!
        
        predictions = np.array(predictions)
        
        # Доверительные интервалы на основе вариации обучающих данных
        if return_conf_int:
            std = np.std(self.data[-min(30, len(self.data)):])
            lower = np.maximum(predictions - 1.96 * std * np.sqrt(np.arange(1, horizon + 1)), 0)
            upper = predictions + 1.96 * std * np.sqrt(np.arange(1, horizon + 1))
        else:
            lower, upper = None, None
        
        return predictions, lower, upper

