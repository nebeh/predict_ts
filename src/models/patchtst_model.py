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
        self.required_context_length = None  # Будет установлено при загрузке модели
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
                    # Загружаем конфигурацию модели
                    try:
                        config = PatchTSTConfig.from_pretrained(self.model_name)
                    except:
                        # Если не удалось загрузить конфигурацию, создаем новую
                        config = PatchTSTConfig()
                    
                    # Получаем требуемую длину контекста из конфигурации модели
                    if hasattr(config, 'context_length') and config.context_length is not None:
                        self.required_context_length = config.context_length
                    else:
                        # По умолчанию используем 512 (стандартная длина для PatchTST)
                        self.required_context_length = 512
                        config.context_length = 512
                    
                    # ВАЖНО: Обновляем конфигурацию для univariate данных (1 канал)
                    # Предобученная модель может быть настроена на multivariate (7 каналов)
                    config.num_input_channels = 1  # Устанавливаем 1 канал для univariate
                    config.prediction_length = 1  # Для авторегрессивного режима
                    
                    # Пробуем загрузить модель с обновленной конфигурацией
                    try:
                        self.model = PatchTSTForPrediction.from_pretrained(
                            self.model_name,
                            config=config,
                            ignore_mismatched_sizes=True  # Игнорируем несоответствие размеров
                        )
                    except Exception as e1:
                        # Если не удалось загрузить с ignore_mismatched_sizes, создаем новую модель
                        print(f"  ⚠ Не удалось загрузить с ignore_mismatched_sizes: {e1}")
                        print(f"  Создаю новую модель с правильной конфигурацией...")
                        self.model = PatchTSTForPrediction(config=config)
                        # Пробуем загрузить веса только для совместимых слоев
                        try:
                            pretrained_model = PatchTSTForPrediction.from_pretrained(self.model_name)
                            # Копируем веса только для совместимых слоев (исключая input embedding)
                            state_dict = pretrained_model.state_dict()
                            model_dict = self.model.state_dict()
                            # Фильтруем веса, исключая слои, связанные с num_input_channels
                            filtered_dict = {k: v for k, v in state_dict.items() 
                                           if k in model_dict and 'value_embedding' not in k 
                                           and model_dict[k].shape == v.shape}
                            model_dict.update(filtered_dict)
                            self.model.load_state_dict(model_dict, strict=False)
                            print(f"  ✓ Загружены совместимые веса из предобученной модели")
                        except Exception as e2:
                            print(f"  ⚠ Не удалось загрузить веса: {e2}")
                            print(f"  Модель будет использовать случайную инициализацию")
                    
                    self.model.eval()  # Переводим в режим оценки
                    print(f"✓ Модель PatchTST успешно загружена/создана")
                    print(f"  Требуемая длина контекста: {self.required_context_length}")
                    print(f"  Количество каналов: {config.num_input_channels}")
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
                # Используем требуемую длину контекста из конфигурации модели
                if self.required_context_length is not None:
                    context_len = self.required_context_length
                else:
                    # Fallback: используем seq_len или длину данных
                    context_len = min(self.seq_len, len(normalized_data))
                
                # Берем последние context_len элементов или дополняем первым значением
                if len(normalized_data) >= context_len:
                    context = normalized_data[-context_len:]
                else:
                    # Если данных меньше требуемой длины, дополняем первым значением
                    padding = np.full(context_len - len(normalized_data), normalized_data[0] if len(normalized_data) > 0 else 0.0)
                    context = np.concatenate([padding, normalized_data])
                
                # Авторегрессивный прогноз: каждый следующий день опирается на предыдущие предсказания
                predictions = []
                lower_bounds = []
                upper_bounds = []
                current_context = context.copy()
                
                with torch.no_grad():
                    for step in range(horizon):
                        # Подготавливаем тензор для модели
                        # PatchTST ожидает (batch_size, sequence_length, num_features)
                        # Для univariate: (1, sequence_length, 1)
                        input_tensor = torch.tensor(current_context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                        
                        # Убеждаемся, что форма тензора правильная: (batch, seq_len, num_channels)
                        # num_channels должно быть 1 для univariate
                        assert input_tensor.shape == (1, len(current_context), 1), \
                            f"Неверная форма тензора: {input_tensor.shape}, ожидается (1, {len(current_context)}, 1)"
                        
                        # Получаем прогноз от модели
                        # PatchTSTForPrediction использует только past_values в forward()
                        # prediction_length задается в конфигурации модели
                        outputs = self.model(past_values=input_tensor)
                        
                        # Получаем прогноз из outputs
                        # PatchTSTForPrediction возвращает объект с prediction_outputs
                        try:
                            if hasattr(outputs, 'prediction_outputs'):
                                # prediction_outputs имеет форму (batch, prediction_length, num_features)
                                pred_tensor = outputs.prediction_outputs
                                if pred_tensor.dim() == 3:
                                    pred_value = pred_tensor[0, 0, 0].item()
                                elif pred_tensor.dim() == 2:
                                    pred_value = pred_tensor[0, 0].item()
                                else:
                                    pred_value = pred_tensor[0].item()
                            elif hasattr(outputs, 'prediction_mean'):
                                # Альтернативный вариант
                                pred_tensor = outputs.prediction_mean
                                if pred_tensor.dim() == 3:
                                    pred_value = pred_tensor[0, 0, 0].item()
                                elif pred_tensor.dim() == 2:
                                    pred_value = pred_tensor[0, 0].item()
                                else:
                                    pred_value = pred_tensor[0].item()
                            elif isinstance(outputs, tuple) and len(outputs) > 0:
                                # Если outputs - это кортеж, берем первый элемент
                                pred_tensor = outputs[0]
                                if pred_tensor.dim() >= 3:
                                    pred_value = pred_tensor[0, 0, 0].item()
                                elif pred_tensor.dim() == 2:
                                    pred_value = pred_tensor[0, 0].item()
                                else:
                                    pred_value = pred_tensor[0].item()
                            elif hasattr(outputs, 'last_hidden_state'):
                                # Используем last_hidden_state как fallback
                                pred_tensor = outputs.last_hidden_state
                                if pred_tensor.dim() >= 2:
                                    # Берем последний элемент и применяем простую проекцию
                                    pred_value = pred_tensor[0, -1, 0].item()
                                else:
                                    pred_value = pred_tensor[0].item()
                            else:
                                # Если ничего не подошло, пробуем получить доступ к тензору напрямую
                                raise ValueError(f"Не удалось извлечь прогноз. Тип outputs: {type(outputs)}")
                        except Exception as e:
                            # Если не удалось извлечь прогноз, переходим на fallback
                            raise ValueError(f"Ошибка при извлечении прогноза: {e}")
                        
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
                        # Сохраняем требуемую длину контекста
                        current_context = np.append(current_context[1:], pred_normalized)
                        # Убеждаемся, что длина контекста соответствует требуемой
                        if len(current_context) > context_len:
                            current_context = current_context[-context_len:]
                        elif len(current_context) < context_len:
                            # Дополняем первым значением, если нужно
                            padding = np.full(context_len - len(current_context), current_context[0])
                            current_context = np.concatenate([padding, current_context])
                
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

