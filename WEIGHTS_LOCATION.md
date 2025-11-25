# Расположение весов моделей

## Модели с предобученными весами

### 1. Amazon Chronos
- **Репозиторий**: `amazon/chronos-t5-tiny`
- **Локальный путь**: `~/.cache/huggingface/hub/models--amazon--chronos-t5-tiny/`
- **Размер**: ~64 MB
- **Основные файлы**:
  - `model.safetensors` (32 MB) - веса модели
  - `config.json` - конфигурация модели
  - `generation_config.json` - конфигурация генерации

### 2. Google TimesFM
- **Репозиторий (новый API)**: `google/timesfm-2.5-200m-pytorch` ✅ **Используется по умолчанию**
- **Локальный путь**: `~/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch/`
- **Размер**: ~400-500 MB
- **Основные файлы**:
  - `model.safetensors` - веса модели в формате safetensors
  - `config.json` - конфигурация модели
- **API**: Используется новый API TimesFM 2.5 с PyTorch backend
  - `timesfm.TimesFM_2p5_200M_torch.from_pretrained()`
  - `model.compile()` с `ForecastConfig`
  - `model.forecast()` возвращает point_forecast и quantile_forecast

- **Репозиторий (старый API)**: `google/timesfm-1.0-200m` (fallback)
- **Локальный путь**: `~/.cache/huggingface/hub/models--google--timesfm-1.0-200m/`
- **Размер**: ~1553 MB (1.5 GB)
- **Основные файлы**:
  - `checkpoints/checkpoint_1100000/` - JAX checkpoint модели
  - Другие файлы конфигурации
- **Важно**: 
  - Старый репозиторий доступен только с **JAX checkpoint**, файл `torch_model.ckpt` отсутствует
  - Если новый API недоступен, код автоматически пробует старый API
  - Если загрузка не удается, модель автоматически переключается на упрощенный метод (fallback)

### 3. PatchTST
- **Репозиторий**: `ibm-research/patchtst-etth1-pretrain`
- **Локальный путь**: `~/.cache/huggingface/hub/models--ibm-research--patchtst-etth1-pretrain/`
- **Размер**: ~50-100 MB (зависит от версии)
- **Основные файлы**:
  - `pytorch_model.bin` или `model.safetensors` - веса модели
  - `config.json` - конфигурация модели
- **Библиотека**: `transformers` (HuggingFace)
- **Статус**: ✅ Используются предобученные веса из HuggingFace через библиотеку `transformers`

## Модели, обучаемые на данных (трансформерные архитектуры)

Следующие модели являются **полноценными трансформерными архитектурами** и обучаются на данных:

### 4. iTransformer (Inverted Transformer)
- **Архитектура**: Inverted Transformer (переменные как токены, временные точки как features)
- **Репозиторий**: https://github.com/thuml/iTransformer
- **Статус**: ✅ Используется полноценная нейросетевая архитектура на PyTorch
- **Обучение**: Модель обучается на данных при вызове `fit()`
- **Веса**: Веса генерируются при обучении, предобученных весов нет
- **Библиотека**: PyTorch (нативная реализация)
- **Fallback**: Если PyTorch недоступен, используется упрощенный метод (экстраполяция с сезонностью)

### 5. TFT (Temporal Fusion Transformer)
- **Архитектура**: Temporal Fusion Transformer (Encoder-Decoder с attention)
- **Репозиторий**: https://github.com/google-research/google-research/tree/master/tft
- **Статус**: ✅ Используется полноценная нейросетевая архитектура на PyTorch
- **Обучение**: Модель обучается на данных при вызове `fit()`
- **Веса**: Веса генерируются при обучении, предобученных весов нет
- **Библиотека**: PyTorch (нативная реализация)
- **Fallback**: Если PyTorch недоступен, используется упрощенный метод (метод Хольта-Винтерса)

**Примечание**: Эти модели теперь используют полноценные нейросетевые архитектуры, которые обучаются на данных. Если PyTorch недоступен, модели автоматически переключаются на упрощенные методы.

## Настройка пути для хранения весов

По умолчанию веса хранятся в: `~/.cache/huggingface/hub/`

Можно изменить путь через переменные окружения:

```bash
export HF_HOME=/path/to/custom/cache
export TRANSFORMERS_CACHE=/path/to/custom/cache
export HUGGINGFACE_HUB_CACHE=/path/to/custom/cache
```

## Проверка весов

Проверить наличие весов можно командой:

```bash
# Chronos
ls -lh ~/.cache/huggingface/hub/models--amazon--chronos-t5-tiny/snapshots/*/

# TimesFM
ls -lh ~/.cache/huggingface/hub/models--google--timesfm-1.0-200m/snapshots/*/

# PatchTST
ls -lh ~/.cache/huggingface/hub/models--ibm-research--patchtst-etth1-pretrain/snapshots/*/
```

## Загрузка весов

### Предобученные модели (автоматическая загрузка):

Веса автоматически загружаются при первом использовании модели через:
- **Chronos**: `ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny")`
- **TimesFM (новый API)**: `TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")`
- **TimesFM (старый API)**: `TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m")`
- **PatchTST**: `PatchTSTForPrediction.from_pretrained("ibm-research/patchtst-etth1-pretrain")` (через библиотеку `transformers`)

После первой загрузки веса кэшируются локально и не загружаются повторно.

### Модели, обучаемые на данных:

iTransformer и TFT автоматически обучаются на данных при вызове метода `fit()`:

```python
# Пример использования
from src.models import iTransformerModel, TFTModel

# iTransformer
model = iTransformerModel(seq_len=96, pred_len=30)
model.fit(train_data)  # Модель обучается здесь
predictions = model.predict(horizon=30)

# TFT
model = TFTModel(seq_len=96, pred_len=30)
model.fit(train_data)  # Модель обучается здесь
predictions = model.predict(horizon=30)
```

**Обучение**: Модели обучаются на доступных данных (мини-обучение для демонстрации). Для продакшн-использования рекомендуется более длительное обучение с валидацией.

**Сохранение весов** (опционально):
```python
# Сохранение весов после обучения
torch.save(model.model.state_dict(), 'weights/itransformer.pth')
torch.save(model.model.state_dict(), 'weights/tft.pth')

# Загрузка весов
model.model.load_state_dict(torch.load('weights/itransformer.pth'))
```

## Общий размер всех весов

### Предобученные модели (загружены с HuggingFace):
- Chronos: ~64 MB
- TimesFM (2.5): ~400-500 MB
- TimesFM (1.0, fallback): ~1553 MB
- PatchTST: ~50-100 MB
- **Итого предобученных**: ~1.0-1.7 GB (в зависимости от используемой версии TimesFM)

### Модели, обучаемые на данных:
- iTransformer: веса генерируются при обучении (полноценная нейросетевая архитектура)
- TFT: веса генерируются при обучении (полноценная нейросетевая архитектура)



