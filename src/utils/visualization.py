"""
Модуль для визуализации результатов прогнозирования
"""
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import matplotlib.dates as mdates


def plot_single_method(
    historical_data: np.ndarray,
    true_future: np.ndarray,
    predicted: np.ndarray,
    method_name: str,
    confidence_lower: Optional[np.ndarray] = None,
    confidence_upper: Optional[np.ndarray] = None,
    start_date: Optional[datetime] = None,
    ylabel: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Создает график для одного метода: история + прогноз с реальными данными
    
    Args:
        historical_data: исторические данные
        true_future: реальные будущие данные
        predicted: предсказанные данные
        method_name: название метода
        confidence_lower: нижняя граница доверительного интервала
        confidence_upper: верхняя граница доверительного интервала
        start_date: начальная дата для оси X
        
    Returns:
        figure, axes
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    n_hist = len(historical_data)
    n_future = len(true_future)
    
    if start_date is None:
        dates_hist = np.arange(n_hist)
        dates_future = np.arange(n_hist, n_hist + n_future)
    else:
        dates_hist = [start_date + timedelta(days=i) for i in range(n_hist)]
        dates_future = [start_date + timedelta(days=i) for i in range(n_hist, n_hist + n_future)]
    
    # Исторические данные
    ax.plot(dates_hist, historical_data, 'b-', label='История', linewidth=2, alpha=0.7)
    
    # Реальные будущие данные
    ax.plot(dates_future, true_future, 'g-', label='Реальные данные', linewidth=2, marker='o', markersize=4)
    
    # Предсказания
    ax.plot(dates_future, predicted, 'r--', label='Предсказание', linewidth=2, marker='s', markersize=4)
    
    # Доверительные интервалы
    if confidence_lower is not None and confidence_upper is not None:
        ax.fill_between(
            dates_future, 
            confidence_lower, 
            confidence_upper, 
            alpha=0.3, 
            color='red', 
            label='Доверительный интервал'
        )
    
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel(ylabel if ylabel is not None else 'Кумулятивная сумма продаж', fontsize=12)
    ax.set_title(f'{method_name} - История и прогноз', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if start_date is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (n_hist + n_future) // 10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig, ax


def plot_prediction_comparison(
    true_future: np.ndarray,
    predicted: np.ndarray,
    method_name: str,
    metrics: Dict[str, float],
    confidence_lower: Optional[np.ndarray] = None,
    confidence_upper: Optional[np.ndarray] = None,
    start_date: Optional[datetime] = None,
    offset: int = 0,
    ylabel: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Создает график сравнения только прогнозов (без истории)
    
    Args:
        true_future: реальные будущие данные
        predicted: предсказанные данные
        method_name: название метода
        metrics: словарь с метриками
        confidence_lower: нижняя граница доверительного интервала
        confidence_upper: верхняя граница доверительного интервала
        start_date: начальная дата для оси X
        offset: смещение для дат (день начала прогноза)
        
    Returns:
        figure, axes
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_future = len(true_future)
    
    if start_date is None:
        dates = np.arange(offset, offset + n_future)
    else:
        dates = [start_date + timedelta(days=i) for i in range(offset, offset + n_future)]
    
    # Реальные данные
    ax.plot(dates, true_future, 'g-', label='Реальные данные', linewidth=2.5, marker='o', markersize=6)
    
    # Предсказания
    ax.plot(dates, predicted, 'r--', label='Предсказание', linewidth=2.5, marker='s', markersize=6)
    
    # Доверительные интервалы
    if confidence_lower is not None and confidence_upper is not None:
        ax.fill_between(
            dates, 
            confidence_lower, 
            confidence_upper, 
            alpha=0.3, 
            color='red', 
            label='Доверительный интервал'
        )
    
    # Добавляем метрики в текст (только для последнего дня)
    metrics_text = 'Метрики (30-й день):\n' + '\n'.join([f'{k}: {v:.2f}' for k, v in metrics.items()])
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel(ylabel if ylabel is not None else 'Кумулятивная сумма продаж', fontsize=12)
    ax.set_title(f'{method_name} - Сравнение прогнозов (метрики на последний день)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if start_date is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, n_future // 10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig, ax


def plot_all_methods_comparison(
    true_future: np.ndarray,
    predictions: Dict[str, np.ndarray],
    start_date: Optional[datetime] = None,
    offset: int = 0,
    ylabel: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Создает итоговый график со всеми методами
    
    Args:
        true_future: реальные будущие данные
        predictions: словарь {метод: предсказания}
        start_date: начальная дата для оси X
        offset: смещение для дат (день начала прогноза)
        
    Returns:
        figure, axes
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    n_future = len(true_future)
    
    if start_date is None:
        dates = np.arange(offset, offset + n_future)
    else:
        dates = [start_date + timedelta(days=i) for i in range(offset, offset + n_future)]
    
    # Реальные данные
    ax.plot(dates, true_future, 'k-', label='Реальные данные', linewidth=3, marker='o', markersize=8)
    
    # Цвета для разных методов (расширенный список для 5+ методов)
    colors = ['red', 'blue', 'orange', 'purple', 'brown', 'green', 'pink', 'gray', 'olive', 'cyan']
    markers = ['s', '^', 'v', 'D', 'p', '*', 'X', 'h', 'H', '+']
    
    # Проверяем, что все предсказания имеют правильную длину
    valid_predictions = {}
    averaged_prediction = None
    for method_name, predicted in predictions.items():
        if predicted is not None and len(predicted) == n_future:
            if method_name == 'Усредненное':
                averaged_prediction = predicted
            else:
                valid_predictions[method_name] = predicted
        else:
            print(f"Предупреждение: {method_name} имеет неверную длину прогноза ({len(predicted) if predicted is not None else 0} вместо {n_future})")
    
    print(f"Отображаются {len(valid_predictions)} методов на итоговом графике: {list(valid_predictions.keys())}")
    
    # Сначала рисуем обычные методы
    for idx, (method_name, predicted) in enumerate(valid_predictions.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(dates, predicted, '--', label=method_name, 
                linewidth=2, marker=marker, markersize=5, color=color, alpha=0.8)
    
    # Затем рисуем усредненное предсказание (отличается от реальных данных)
    if averaged_prediction is not None:
        ax.plot(dates, averaged_prediction, '-.', label='Усредненное', 
                linewidth=3, marker='D', markersize=7, color='darkblue', alpha=0.9, zorder=10)
    
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel(ylabel if ylabel is not None else 'Кумулятивная сумма продаж', fontsize=12)
    ax.set_title(f'Сравнение всех методов прогнозирования ({len(valid_predictions)} методов)', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if start_date is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, n_future // 10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig, ax

