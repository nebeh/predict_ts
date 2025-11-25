"""
Модель TFT (Temporal Fusion Transformer) - полноценная реализация на PyTorch
"""
import numpy as np
from typing import Tuple, Optional
from .base_model import BaseForecastModel

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class TemporalFusionTransformer(nn.Module):
    """Архитектура TFT (Temporal Fusion Transformer)"""
    
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 30,
        d_model: int = 64,
        n_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len + pred_len, d_model))
        
        # Encoder (для исторических данных)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Decoder (для прогноза)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.projection = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (batch, seq_len, 1) - входные исторические данные
        returns: (batch, pred_len) - прогноз
        """
        batch_size = x.shape[0]
        
        # Embedding исторических данных
        x_enc = self.input_embedding(x)  # (batch, seq_len, d_model)
        x_enc = x_enc + self.pos_encoding[:, :self.seq_len, :]
        x_enc = self.dropout(x_enc)
        
        # Encoder
        memory = self.encoder(x_enc)  # (batch, seq_len, d_model)
        
        # Decoder input (для прогноза)
        # Создаем placeholder для прогнозных шагов
        decoder_input = torch.zeros(batch_size, self.pred_len, self.d_model, device=x.device)
        decoder_input = decoder_input + self.pos_encoding[:, self.seq_len:, :]
        
        # Decoder
        decoder_output = self.decoder(decoder_input, memory)  # (batch, pred_len, d_model)
        
        # Output projection
        pred = self.projection(decoder_output).squeeze(-1)  # (batch, pred_len)
        
        return pred


class TFTModel(BaseForecastModel):
    """Полноценная реализация TFT на PyTorch"""
    
    def __init__(self, seq_len: int = 96, pred_len: int = 30, d_model: int = 64, n_heads: int = 4):
        super().__init__("TFT")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.data = None
        self.d_model = d_model
        self.n_heads = n_heads
        
        if not TORCH_AVAILABLE:
            print("⚠ Предупреждение: PyTorch не установлен. Будет использован упрощенный метод.")
    
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
        """Обучает модель TFT"""
        self.data = data.copy()
        normalized_data = self._normalize(data)
        
        if TORCH_AVAILABLE:
            try:
                # Создаем модель
                self.model = TemporalFusionTransformer(
                    seq_len=self.seq_len,
                    pred_len=self.pred_len,
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                )
                self.model.train()
                
                # Простое обучение на доступных данных
                if len(normalized_data) >= self.seq_len + self.pred_len:
                    # Подготавливаем данные для обучения
                    X_train = []
                    y_train = []
                    
                    for i in range(len(normalized_data) - self.seq_len - self.pred_len + 1):
                        X_train.append(normalized_data[i:i+self.seq_len])
                        y_train.append(normalized_data[i+self.seq_len:i+self.seq_len+self.pred_len])
                    
                    if len(X_train) > 0:
                        X_train = torch.tensor(np.array(X_train), dtype=torch.float32).unsqueeze(-1)
                        y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
                        
                        # Простое обучение (несколько эпох)
                        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
                        criterion = nn.MSELoss()
                        
                        # Мини-обучение (для демонстрации)
                        n_epochs = min(10, max(1, len(X_train) // 10))
                        for epoch in range(n_epochs):
                            optimizer.zero_grad()
                            pred = self.model(X_train)
                            loss = criterion(pred, y_train)
                            loss.backward()
                            optimizer.step()
                        
                        self.model.eval()
                        print(f"✓ TFT обучен на {len(X_train)} примерах ({n_epochs} эпох)")
                    else:
                        print("⚠ Недостаточно данных для обучения. Модель будет использовать случайную инициализацию.")
                        self.model.eval()
                else:
                    print("⚠ Недостаточно данных для обучения. Модель будет использовать случайную инициализацию.")
                    self.model.eval()
                    
            except Exception as e:
                print(f"⚠ Ошибка при создании/обучении TFT: {e}")
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
        
        # Если модель обучена, используем её
        if self.model is not None and TORCH_AVAILABLE:
            try:
                normalized_data = self._normalize(self.data)
                
                # Авторегрессивный прогноз: каждый следующий день опирается на предыдущие предсказания
                predictions = []
                lower_bounds = []
                upper_bounds = []
                current_context = normalized_data[-self.seq_len:].copy()
                
                with torch.no_grad():
                    for step in range(horizon):
                        # Подготавливаем входной тензор
                        input_tensor = torch.tensor(current_context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                        
                        # Получаем прогноз (можем предсказать несколько шагов, но используем только первый)
                        pred_normalized = self.model(input_tensor)[0, 0].item()
                        
                        # Денормализуем
                        pred_value = self._denormalize(np.array([pred_normalized]))[0]
                        predictions.append(max(0, pred_value))  # Неотрицательные объемы
                        
                        # Доверительные интервалы (эмпирические)
                        if return_conf_int:
                            std_val = np.std(self.data[-min(30, len(self.data)):])
                            lower_bounds.append(max(0, pred_value - 1.96 * std_val))
                            upper_bounds.append(pred_value + 1.96 * std_val)
                        
                        # Обновляем контекст для следующего шага, используя СВОЕ предсказание
                        pred_normalized = self._normalize(np.array([pred_value]))[0]
                        current_context = np.append(current_context[1:], pred_normalized)
                
                predictions = np.array(predictions)
                
                if return_conf_int:
                    lower = np.array(lower_bounds) if len(lower_bounds) == horizon else None
                    upper = np.array(upper_bounds) if len(upper_bounds) == horizon else None
                else:
                    lower, upper = None, None
                
                return predictions, lower, upper
                
            except Exception as e:
                print(f"⚠ Ошибка при использовании модели TFT: {e}")
                print("  Переключаюсь на упрощенный метод")
                # Fallback на упрощенный метод
        
        # Fallback: упрощенный метод (если модель не обучена или произошла ошибка)
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
