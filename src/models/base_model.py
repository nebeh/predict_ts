"""
Базовый класс для моделей прогнозирования
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseForecastModel(ABC):
    """Базовый класс для всех моделей прогнозирования"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Обучает модель на данных
        
        Args:
            data: временной ряд для обучения
        """
        pass
    
    @abstractmethod
    def predict(self, horizon: int, return_conf_int: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Делает прогноз на заданный горизонт
        
        Args:
            horizon: горизонт прогнозирования
            return_conf_int: возвращать ли доверительные интервалы
            
        Returns:
            predictions, lower_bound, upper_bound
        """
        pass

