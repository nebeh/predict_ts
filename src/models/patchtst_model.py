"""
Модель PatchTST с предобученными весами из HuggingFace
"""
import numpy as np
from typing import Tuple, Optional
from .base_model import BaseForecastModel

try:
    from transformers import PatchTSTForPrediction, PatchTSTConfig
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PatchTSTModel(BaseForecastModel):
    """Реализация PatchTST с предобученными весами из HuggingFace"""
    
    def __init__(self, model_name: str = "ibm-research/patchtst-etth1-pretrain", seq_len: int = 96, pred_len: int = 30):
        super().__init__("PatchTST")
        self.model_name = model_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.data = None
        if not TRANSFORMERS_AVAILABLE:
            print("Предупреждение: transformers не установлен. Будет использован упрощенный метод.")
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
        """Загружает предобученную модель PatchTST из HuggingFace"""
        self.data = data.copy()
        # Нормализуем данные
        normalized_data = self._normalize(data)
        
        # Пытаемся загрузить предобученную модель
        if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            try:
                print(f"Загрузка предобученной модели PatchTST: {self.model_name}")
                # Пробуем загрузить предобученную модель
                try:
                    self.model = PatchTSTForPrediction.from_pretrained(self.model_name)
                    self.model.eval()  # Переводим в режим оценки
                    print(f"✓ Модель PatchTST успешно загружена из HuggingFace")
                except Exception as e:
                    print(f"⚠ Не удалось загрузить предобученную модель {self.model_name}: {e}")
                    print("  Будет использован упрощенный метод")
                    self.model = None
            except Exception as e:
                print(f"⚠ Ошибка при загрузке PatchTST: {e}")
                print("  Будет использован упрощенный метод")
                self.model = None
        else:
            self.model = None
        
        self.is_fitted = True
    
    def predict(self, horizon: int, return_conf_int: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Делает прогноз в авторегрессивном режиме (seq2seq)
        Каждый следующий день предсказывается на основе предыдущих предсказаний
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        # Если модель загружена, используем её
        if self.model is not None and TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
            try:
                # Нормализуем данные
                normalized_data = self._normalize(self.data)
                
                # Подготавливаем входные данные для модели
                # PatchTST ожидает тензор формы (batch_size, sequence_length, num_features)
                # Для унивариантного ряда: (1, seq_len, 1)
                context_len = min(self.seq_len, len(normalized_data))
                context = normalized_data[-context_len:]
                
                # Авторегрессивный прогноз: каждый следующий день опирается на предыдущие предсказания
                predictions = []
                lower_bounds = []
                upper_bounds = []
                current_context = context.copy()
                
                with torch.no_grad():
                    for step in range(horizon):
                        # Подготавливаем тензор для модели
                        # PatchTST ожидает (batch_size, sequence_length, num_features)
                        input_tensor = torch.tensor(current_context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                        
                        # Получаем прогноз от модели
                        outputs = self.model(past_values=input_tensor, prediction_length=1)
                        pred_value = outputs.prediction_mean[0, 0, 0].item()
                        
                        # Денормализуем
                        pred_value = self._denormalize(np.array([pred_value]))[0]
                        predictions.append(max(0, pred_value))  # Неотрицательные объемы
                        
                        # Доверительные интервалы (если доступны)
                        if return_conf_int and hasattr(outputs, 'prediction_distribution'):
                            try:
                                # Пытаемся получить квантили из распределения
                                lower_val = outputs.prediction_distribution.icdf(torch.tensor(0.025)).item()
                                upper_val = outputs.prediction_distribution.icdf(torch.tensor(0.975)).item()
                                lower_val = self._denormalize(np.array([lower_val]))[0]
                                upper_val = self._denormalize(np.array([upper_val]))[0]
                                lower_bounds.append(max(0, lower_val))
                                upper_bounds.append(upper_val)
                            except:
                                # Если не удалось получить распределение, используем эмпирические интервалы
                                std_val = np.std(self.data[-min(30, len(self.data)):])
                                lower_bounds.append(max(0, pred_value - 1.96 * std_val))
                                upper_bounds.append(pred_value + 1.96 * std_val)
                        
                        # Обновляем контекст для следующего шага, используя СВОЕ предсказание
                        # Добавляем предсказание в контекст (нормализованное)
                        pred_normalized = self._normalize(np.array([pred_value]))[0]
                        current_context = np.append(current_context[1:], pred_normalized)
                
                predictions = np.array(predictions)
                
                if return_conf_int:
                    if len(lower_bounds) == horizon:
                        lower = np.array(lower_bounds)
                        upper = np.array(upper_bounds)
                    else:
                        # Fallback: эмпирические интервалы
                        std_val = np.std(self.data[-min(30, len(self.data)):])
                        lower = np.maximum(predictions - 1.96 * std_val * np.sqrt(np.arange(1, horizon + 1)), 0)
                        upper = predictions + 1.96 * std_val * np.sqrt(np.arange(1, horizon + 1))
                else:
                    lower, upper = None, None
                
                return predictions, lower, upper
                
            except Exception as e:
                print(f"⚠ Ошибка при использовании предобученной модели PatchTST: {e}")
                print("  Переключаюсь на упрощенный метод")
                # Fallback на упрощенный метод
        
        # Fallback: упрощенный метод (если модель не загружена или произошла ошибка)
        # ВАЖНО: Работаем только с train_data (self.data), НЕ используем test_data!
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

