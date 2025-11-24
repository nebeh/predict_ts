"""
Модули для моделей прогнозирования временных рядов
"""
from .base_model import BaseForecastModel
from .chronos_model import ChronosModel
from .timesfm_model import TimesFMModel
from .patchtst_model import PatchTSTModel
from .itransformer_model import iTransformerModel
from .tft_model import TFTModel

__all__ = [
    'BaseForecastModel',
    'ChronosModel',
    'TimesFMModel',
    'PatchTSTModel',
    'iTransformerModel',
    'TFTModel'
]

