"""
Модель Google TimesFM
"""
import numpy as np
from typing import Tuple, Optional
from .base_model import BaseForecastModel

try:
    import timesfm
    from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
    TIMESFM_AVAILABLE = True
except ImportError:
    TIMESFM_AVAILABLE = False
    TimesFm = None
    TimesFmHparams = None
    TimesFmCheckpoint = None


class TimesFMModel(BaseForecastModel):
    """Обертка для Google TimesFM"""
    
    def __init__(self):
        super().__init__("TimesFM")
        self.model = None
        self.last_value = None
        self.data = None
        self.use_fallback = not TIMESFM_AVAILABLE
        if self.use_fallback:
            print("Предупреждение: timesfm не установлен или несовместим с текущей версией Python.")
            print("  Будет использован упрощенный метод на основе экстраполяции тренда.")
            print("  Для полной функциональности TimesFM требуется Python 3.10-3.11 и зависимости:")
            print("  jax==0.4.26, jaxlib==0.4.26, paxml==1.4.0, praxis==1.4.0")
    
    def fit(self, data: np.ndarray) -> None:
        """TimesFM не требует обучения, использует предобученную модель"""
        if self.use_fallback:
            self.model = None
        else:
            try:
                # Правильный способ инициализации TimesFM через TimesFmHparams и TimesFmCheckpoint
                hparams = TimesFmHparams(
                    context_len=512,
                    horizon_len=128,
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=20,
                    model_dims=1280,
                    backend="cpu",
                )
                # Используем checkpoint с указанием репозитория на HuggingFace
                # Пробуем разные версии
                checkpoint = None
                for version in ["jax", "pytorch"]:
                    try:
                        checkpoint = TimesFmCheckpoint(
                            version=version,
                            huggingface_repo_id="google/timesfm-1.0-200m"
                        )
                        self.model = TimesFm(hparams=hparams, checkpoint=checkpoint)
                        print(f"TimesFM успешно загружен с версией: {version}")
                        break
                    except Exception as e:
                        continue
                
                if self.model is None:
                    raise Exception("Не удалось загрузить TimesFM ни с одной версией")
            except Exception as e:
                print(f"Не удалось загрузить TimesFM: {e}. Будет использован упрощенный метод.")
                self.model = None
                self.use_fallback = True
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
        context_window = min(512, len(self.data))
        current_context = self.data[-context_window:].copy().astype(np.float32)
        predictions = []
        lower_bounds = []
        upper_bounds = []
        
        if self.model is None or self.use_fallback:
            # Улучшенный fallback: авторегрессивная линейная регрессия
            if len(self.data) > 1:
                lookback = min(60, len(self.data))
                x = np.arange(lookback)
                y = self.data[-lookback:]
                coeffs = np.polyfit(x, y, 1)
                trend = coeffs[0]
                current_value = self.data[-1]
                
                # Вычисляем стандартное отклонение остатков
                if len(self.data) > 2:
                    residuals = y - np.polyval(coeffs, x)
                    std_residual = np.std(residuals)
                else:
                    std_residual = 0
            else:
                current_value = self.last_value
                trend = 0
                std_residual = 0
            
            # Авторегрессивный прогноз
            for step in range(horizon):
                # Предсказываем следующий день на основе текущего значения и тренда
                current_value = current_value + trend
                predictions.append(max(0, current_value))
                if return_conf_int:
                    # Доверительные интервалы расширяются со временем
                    lower_bounds.append(max(0, current_value - 1.96 * std_residual * np.sqrt(step + 1)))
                    upper_bounds.append(current_value + 1.96 * std_residual * np.sqrt(step + 1))
            
            predictions = np.array(predictions)
            lower = np.array(lower_bounds) if return_conf_int else None
            upper = np.array(upper_bounds) if return_conf_int else None
            return predictions, lower, upper
        
        # Режим с TimesFM: авторегрессивное предсказание
        try:
            for step in range(horizon):
                # Предсказываем ОДИН шаг вперед
                forecast = self.model.forecast(current_context, 1)
                
                # Преобразуем в numpy если нужно
                if hasattr(forecast, 'numpy'):
                    forecast = forecast.numpy()
                elif hasattr(forecast, 'cpu'):
                    forecast = forecast.cpu().numpy()
                
                # Получаем предсказание для следующего дня
                if forecast.ndim > 1:
                    pred_value = np.mean(forecast, axis=0)[0] if forecast.shape[0] > 1 else forecast[0, 0]
                    if return_conf_int:
                        lower_val = np.percentile(forecast, 5, axis=0)[0] if forecast.shape[0] > 1 else forecast[0, 0]
                        upper_val = np.percentile(forecast, 95, axis=0)[0] if forecast.shape[0] > 1 else forecast[0, 0]
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
            # Fallback на простую авторегрессию
            print(f"Ошибка при использовании TimesFM: {e}. Используется упрощенный метод.")
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
                predictions.append(max(0, current_value))
            
            predictions = np.array(predictions)
            if return_conf_int:
                std_daily = np.std(self.data[-30:]) if len(self.data) > 1 else 0
                lower = np.maximum(predictions - 1.96 * std_daily * np.sqrt(np.arange(1, horizon + 1)), 0)
                upper = predictions + 1.96 * std_daily * np.sqrt(np.arange(1, horizon + 1))
            else:
                lower, upper = None, None
        
        return predictions, lower, upper

