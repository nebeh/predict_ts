"""
Модель Google TimesFM
"""
import numpy as np
from typing import Tuple, Optional
from .base_model import BaseForecastModel

try:
    import timesfm
    # Проверяем доступность нового API (как в рабочем скрипте timesFM.py)
    TIMESFM_NEW_API = hasattr(timesfm, 'TimesFM_2p5_200M_torch') and hasattr(timesfm, 'ForecastConfig')
    TIMESFM_AVAILABLE = True
except ImportError:
    TIMESFM_AVAILABLE = False
    TIMESFM_NEW_API = False
    timesfm = None


class TimesFMModel(BaseForecastModel):
    """Обертка для Google TimesFM"""
    
    def __init__(self):
        super().__init__("TimesFM")
        self.model = None
        self.last_value = None
        self.data = None
        self.use_fallback = not TIMESFM_AVAILABLE
        self.use_new_api = False
        if self.use_fallback:
            print("Предупреждение: timesfm не установлен или несовместим с текущей версией Python.")
            print("  Будет использован упрощенный метод на основе экстраполяции тренда.")
    
    def fit(self, data: np.ndarray) -> None:
        """TimesFM не требует обучения, использует предобученную модель"""
        if self.use_fallback:
            self.model = None
        else:
            # Пробуем новый API (TimesFM 2.5) - используем тот же подход, что в timesFM.py
            loaded = False
            if TIMESFM_NEW_API:
                try:
                    # Используем прямой доступ через timesfm, как в рабочем скрипте
                    self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                        "google/timesfm-2.5-200m-pytorch",
                        torch_compile=True
                    )
                    
                    # Компилируем модель с конфигурацией (как в timesFM.py)
                    self.model.compile(
                        timesfm.ForecastConfig(
                            max_context=1024,
                            max_horizon=256,
                            normalize_inputs=True,
                            use_continuous_quantile_head=True,
                            force_flip_invariance=True,
                            infer_is_positive=True,
                            fix_quantile_crossing=True,
                        )
                    )
                    print("✓ TimesFM 2.5 успешно загружен из google/timesfm-2.5-200m-pytorch")
                    self.use_new_api = True
                    loaded = True
                except Exception as e:
                    # Если новый API не сработал, пробуем старый
                    print(f"Не удалось загрузить TimesFM 2.5: {e}")
                    print("Пробую старый API (TimesFM 1.0)...")
            
            # Если новый API не сработал или недоступен, пробуем старый API (TimesFM 1.0)
            if not loaded:
                try:
                    # Импортируем старый API
                    from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint
                    
                    hparams = TimesFmHparams(
                        context_len=512,
                        horizon_len=128,
                        input_patch_len=32,
                        output_patch_len=128,
                        num_layers=20,
                        model_dims=1280,
                        backend="cpu",
                    )
                    
                    # Пробуем загрузить старую модель
                    checkpoint = TimesFmCheckpoint(
                        huggingface_repo_id="google/timesfm-1.0-200m"
                    )
                    self.model = TimesFm(hparams=hparams, checkpoint=checkpoint)
                    print("✓ TimesFM 1.0 успешно загружен")
                    self.use_new_api = False
                    loaded = True
                except Exception as e:
                    # Если не удалось загрузить ни одну версию, используем fallback
                    print(f"Не удалось загрузить TimesFM: {e}. Будет использован упрощенный метод.")
                    self.model = None
                    self.use_fallback = True
                    self.use_new_api = False
                    
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
            if hasattr(self, 'use_new_api') and self.use_new_api:
                # Новый API (TimesFM 2.5)
                # Используем весь доступный контекст для прогноза
                point_forecast, quantile_forecast = self.model.forecast(
                    horizon=horizon,
                    inputs=[current_context]
                )
                
                # Преобразуем в numpy если нужно
                if hasattr(point_forecast, 'numpy'):
                    point_forecast = point_forecast.numpy()
                elif hasattr(point_forecast, 'cpu'):
                    point_forecast = point_forecast.cpu().numpy()
                
                if hasattr(quantile_forecast, 'numpy'):
                    quantile_forecast = quantile_forecast.numpy()
                elif hasattr(quantile_forecast, 'cpu'):
                    quantile_forecast = quantile_forecast.cpu().numpy()
                
                # point_forecast имеет форму (1, horizon) - один временной ряд, horizon шагов
                # quantile_forecast имеет форму (1, horizon, num_quantiles)
                if point_forecast.ndim > 1:
                    predictions = point_forecast[0]  # Берем первый (и единственный) временной ряд
                else:
                    predictions = point_forecast
                
                if return_conf_int and quantile_forecast is not None:
                    # Используем квантили для доверительных интервалов
                    if quantile_forecast.ndim == 3:
                        # quantile_forecast: (1, horizon, num_quantiles)
                        quantiles = quantile_forecast[0]  # (horizon, num_quantiles)
                        # Обычно квантили: [0.1, 0.5, 0.9] или подобные
                        # Берем нижний и верхний квантили
                        if quantiles.shape[1] >= 3:
                            lower = quantiles[:, 0]  # Первый квантиль (например, 0.1)
                            upper = quantiles[:, -1]  # Последний квантиль (например, 0.9)
                        elif quantiles.shape[1] == 2:
                            lower = quantiles[:, 0]
                            upper = quantiles[:, 1]
                        else:
                            # Если только один квантиль, используем его для обоих
                            lower = quantiles[:, 0] - 0.1 * np.abs(quantiles[:, 0])
                            upper = quantiles[:, 0] + 0.1 * np.abs(quantiles[:, 0])
                    else:
                        # Fallback: используем стандартное отклонение
                        std_pred = np.std(predictions) if len(predictions) > 1 else predictions[0] * 0.1
                        lower = predictions - 1.96 * std_pred
                        upper = predictions + 1.96 * std_pred
                else:
                    lower, upper = None, None
                
                predictions = np.maximum(predictions, 0)  # Убеждаемся, что прогноз неотрицателен
                if lower is not None:
                    lower = np.maximum(lower, 0)
                if upper is not None:
                    upper = np.maximum(upper, 0)
                    
            else:
                # Старый API (TimesFM 1.0) - авторегрессивное предсказание
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

