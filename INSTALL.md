# Установка зависимостей

**Важно:** TimesFM требует Python 3.10 или 3.11 и имеет строгие зависимости версий.
Для работы с моделями прогнозирования временных рядов необходимо установить следующие библиотеки 

## Базовые зависимости

```bash
micromamba install -y numpy matplotlib pandas scipy pyyaml
```

## Зависимости для моделей

### 1. Amazon Chronos
```bash
pip install chronos-forecasting
```

### 2. Google TimesFM

Если у вас Python 3.10-3.11:
```bash
pip install timesfm
```

Если у вас другая версия Python (например, 3.14), TimesFM будет использовать упрощенный метод прогнозирования.

**Зависимости TimesFM:**
- jax==0.4.26
- jaxlib==0.4.26  
- paxml==1.4.0
- praxis==1.4.0
- utilsforecast==0.1.10
- huggingface-hub[cli]==0.23.0
- numpy==1.26.4
- pandas==2.1.4
- scikit-learn==1.5.1

### 3. PatchTST
PatchTST использует предобученные веса из HuggingFace через библиотеку `transformers`:
```bash
pip install transformers
```

Также требуется PyTorch:
```bash
micromamba install -y pytorch -c pytorch
# или
pip install torch
```

**Примечание**: PatchTST автоматически загружает предобученные веса из `ibm-research/patchtst-etth1-pretrain` при первом использовании.

### 4. iTransformer, TFT
Эти модели требуют PyTorch (в текущей реализации используются упрощенные методы):
```bash
micromamba install -y pytorch -c pytorch
# или
pip install torch
```

## Примечания

- Если какая-то библиотека не установлена, модели будут использовать упрощенные методы прогнозирования (экстраполяция на основе тренда)
- Для полной функциональности рекомендуется установить все зависимости
- Chronos и TimesFM - это большие предобученные модели, их загрузка может занять время при первом использовании

