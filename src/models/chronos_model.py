"""
Модель Amazon Chronos
"""
import numpy as np
from typing import Tuple, Optional
from .base_model import BaseForecastModel

try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False


class ChronosModel(BaseForecastModel):
    """Обертка для Amazon Chronos"""
    
    def __init__(self, model_name: str = "chronos-tiny"):
        super().__init__("Chronos")
        self.model_name = model_name
        self.pipeline = None
        self.last_value = None
        self.data = None
        if not CHRONOS_AVAILABLE:
            print("Предупреждение: chronos не установлен. Будет использован упрощенный метод.")
    
    def fit(self, data: np.ndarray) -> None:
        """Chronos не требует обучения, использует предобученную модель"""
        try:
            # Правильные имена моделей Chronos на HuggingFace
            # Доступные модели: amazon/chronos-t5-tiny, amazon/chronos-bolt-tiny, amazon/chronos-2 и др.
            model_variants = []
            
            # Если указан "chronos-tiny", пробуем правильные варианты
            if self.model_name == "chronos-tiny":
                model_variants = [
                    "amazon/chronos-t5-tiny",  # Правильное имя
                    "amazon/chronos-bolt-tiny",
                    "amazon/chronos-tiny",  # На случай если есть
                ]
            else:
                # Для других имен пробуем стандартные варианты
                model_variants = [
                    f"amazon/{self.model_name}",
                    self.model_name,
                ]
            
            self.pipeline = None
            last_error = None
            for model_path in model_variants:
                try:
                    # Пробуем загрузить модель
                    self.pipeline = ChronosPipeline.from_pretrained(model_path, device_map="cpu")
                    print(f"Chronos успешно загружен: {model_path}")
                    break
                except Exception as e:
                    last_error = e
                    continue
            
            if self.pipeline is None:
                raise Exception(f"Не удалось загрузить ни один вариант модели. Последняя ошибка: {last_error}")
                
        except Exception as e:
            print(f"Не удалось загрузить Chronos: {e}. Будет использован упрощенный метод.")
            self.pipeline = None
        self.data = data.copy()
        self.last_value = data[-1]
        self.is_fitted = True
    
    def predict(self, horizon: int, return_conf_int: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Делает прогноз в авторегрессивном режиме (seq2seq)
        Каждый следующий день предсказывается на основе предыдущих предсказаний
        
        Args:
            horizon: горизонт прогнозирования
            return_conf_int: возвращать ли доверительные интервалы
            
        Returns:
            predictions, lower_bound, upper_bound
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        # ВАЖНО: Работаем только с train_data (self.data), НЕ используем test_data!
        # Начинаем с контекста из обучающих данных
        context_window = min(100, len(self.data))
        current_context = self.data[-context_window:].copy().astype(np.float32)
        predictions = []
        lower_bounds = []
        upper_bounds = []
        
        # Fallback режим
        if self.pipeline is None:
            # Простая авторегрессия: используем среднее приращение
            if len(self.data) > 1:
                mean_daily = np.mean(self.data[-30:])
                mean_diff = np.mean(np.diff(self.data[-30:])) if len(self.data) > 1 else 0
            else:
                mean_daily = self.data[-1] if len(self.data) > 0 else 0
                mean_diff = 0
            
            current_value = self.data[-1]
            std_daily = np.std(self.data[-30:]) if len(self.data) > 1 else 0
            
            for step in range(horizon):
                # Предсказываем следующий день на основе текущего значения
                current_value = current_value + mean_diff
                predictions.append(current_value)
                if return_conf_int:
                    # Доверительные интервалы расширяются со временем
                    lower_bounds.append(max(0, current_value - 1.96 * std_daily * np.sqrt(step + 1)))
                    upper_bounds.append(current_value + 1.96 * std_daily * np.sqrt(step + 1))
            
            predictions = np.array(predictions)
            lower = np.array(lower_bounds) if return_conf_int else None
            upper = np.array(upper_bounds) if return_conf_int else None
            return predictions, lower, upper
        
        # Режим с Chronos: авторегрессивное предсказание
        try:
            import torch
            
            for step in range(horizon):
                # Преобразуем текущий контекст в тензор
                context_tensor = torch.tensor(current_context, dtype=torch.float32)
                if context_tensor.ndim == 1:
                    context_tensor = context_tensor.unsqueeze(0)  # Добавляем batch dimension
                
                # Предсказываем ОДИН шаг вперед
                forecast = self.pipeline.predict(
                    inputs=context_tensor,
                    prediction_length=1,  # Только один шаг!
                    num_samples=100
                )
                
                # Преобразуем в numpy
                if hasattr(forecast, 'cpu'):
                    forecast = forecast.cpu().numpy()
                elif hasattr(forecast, 'numpy'):
                    forecast = forecast.numpy()
                elif hasattr(forecast, 'values'):
                    forecast = forecast.values
                
                # Убираем batch dimension если есть
                if forecast.ndim == 3:
                    forecast = forecast[0]  # [batch, num_samples, 1] -> [num_samples, 1]
                
                # Получаем предсказание для следующего дня
                if forecast.ndim > 1:
                    pred_value = np.mean(forecast, axis=0)[0]  # Среднее по сэмплам, первый (и единственный) шаг
                    if return_conf_int:
                        lower_val = np.percentile(forecast, 5, axis=0)[0]
                        upper_val = np.percentile(forecast, 95, axis=0)[0]
                    else:
                        lower_val, upper_val = None, None
                else:
                    pred_value = forecast[0] if len(forecast) > 0 else current_context[-1]
                    lower_val, upper_val = None, None
                
                # Убеждаемся, что прогноз неотрицателен
                pred_value = max(0, pred_value)
                
                # Добавляем предсказание к результатам
                predictions.append(pred_value)
                if return_conf_int and lower_val is not None:
                    lower_bounds.append(max(0, lower_val))
                    upper_bounds.append(max(0, upper_val))
                
                # ВАЖНО: Обновляем контекст, добавляя СВОЕ предсказание (не реальные данные!)
                # Это и есть авторегрессивный режим: следующий шаг опирается на предыдущие предсказания
                current_context = np.append(current_context, pred_value)
                # Ограничиваем размер контекста
                if len(current_context) > context_window:
                    current_context = current_context[-context_window:]
            
            predictions = np.array(predictions)
            lower = np.array(lower_bounds) if return_conf_int and len(lower_bounds) > 0 else None
            upper = np.array(upper_bounds) if return_conf_int and len(upper_bounds) > 0 else None
            
        except Exception as e:
            # Fallback на простую авторегрессию если Chronos не работает
            print(f"Ошибка при использовании Chronos: {e}. Используется упрощенный метод.")
            if len(self.data) > 1:
                mean_daily = np.mean(self.data[-30:])
                mean_diff = np.mean(np.diff(self.data[-30:])) if len(self.data) > 1 else 0
            else:
                mean_daily = self.data[-1] if len(self.data) > 0 else 0
                mean_diff = 0
            
            current_value = self.data[-1]
            predictions = []
            for step in range(horizon):
                current_value = current_value + mean_diff
                predictions.append(current_value)
            
            predictions = np.array(predictions)
            if return_conf_int:
                std_daily = np.std(self.data[-30:]) if len(self.data) > 1 else 0
                lower = np.maximum(predictions - 1.96 * std_daily * np.sqrt(np.arange(1, horizon + 1)), 0)
                upper = predictions + 1.96 * std_daily * np.sqrt(np.arange(1, horizon + 1))
            else:
                lower, upper = None, None
        
        return predictions, lower, upper

