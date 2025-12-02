"""
Задание 3: Анализ датасета Diabetes - ШАБЛОН
Цель: Анализ данных диабета - регрессионная задача

ЗАДАЧИ:
1. Загрузить данные в load_data()
2. Вычислить статистику целевой переменной в target_analysis()
3. Вычислить статистику признаков в feature_statistics()
4. Создать гистограмму и KDE график целевой переменной в visualize_target()
5. Создать гистограммы всех признаков в visualize_features()
6. Создать scatter plots признаков vs целевой переменной в scatter_features_vs_target()
7. Вычислить и визуализировать корреляции в correlation_analysis()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')

# Настройки для отображения графиков в VS Code
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)

# Попробовать использовать стиль, который точно существует
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')  # Использовать стиль по умолчанию, если seaborn недоступен


def load_data():
    """Загрузить датасет Diabetes и конвертировать в DataFrame"""
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    return df


def target_analysis(df):
    """Анализ целевой переменной (прогрессия болезни)"""
    target = df['target']
    print("\n" + "="*50)
    print("АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (target)")
    print("="*50)
    print(f"Среднее значение:      {target.mean():.2f}")
    print(f"Медиана:               {target.median():.2f}")
    print(f"Стандартное отклонение: {target.std():.2f}")
    print(f"Минимум:               {target.min():.2f}")
    print(f"Максимум:              {target.max():.2f}")
    print(f"Размах:                {target.max() - target.min():.2f}")
    print(f"25-й процентиль:       {target.quantile(0.25):.2f}")
    print(f"50-й процентиль:       {target.quantile(0.50):.2f}")
    print(f"75-й процентиль:       {target.quantile(0.75):.2f}")


def feature_statistics(df):
    """Вычислить статистику по признакам"""
    features = df.columns[:-1]
    print("\n" + "="*50)
    print("СТАТИСТИКА ПРИЗНАКОВ")
    print("="*50)
    stats = df[features].describe().T[['mean', 'std', 'min', 'max']]
    stats.columns = ['Среднее', 'Стд. отклонение', 'Минимум', 'Максимум']
    print(stats.round(4))


def visualize_target(df):
    """Визуализировать распределение целевой переменной"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Гистограмма
    ax1.hist(df['target'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(df['target'].mean(), color='red', linestyle='--', label=f"Среднее: {df['target'].mean():.1f}")
    ax1.set_title('Гистограмма целевой переменной')
    ax1.set_xlabel('Прогрессия болезни')
    ax1.set_ylabel('Частота')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # KDE
    sns.kdeplot(df['target'], ax=ax2, color='orange', fill=True)
    ax2.axvline(df['target'].mean(), color='red', linestyle='--', label=f"Среднее: {df['target'].mean():.1f}")
    ax2.set_title('KDE целевой переменной')
    ax2.set_xlabel('Прогрессия болезни')
    ax2.set_ylabel('Плотность')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('03_diabetes_target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✅ Сохранен файл: 03_diabetes_target_distribution.png")


def visualize_features(df):
    """Визуализировать распределение признаков"""
    features = df.columns[:-1]
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols  # 10 признаков → 5 строк
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()
    
    colors = plt.cm.tab10.colors  # 10 цветов
    
    for i, feature in enumerate(features):
        axes[i].hist(df[feature], bins=15, alpha=0.7, color=colors[i % len(colors)], edgecolor='black')
        axes[i].axvline(df[feature].mean(), color='red', linestyle='--', 
                        label=f"Среднее: {df[feature].mean():.3f}")
        axes[i].set_title(f'Распределение: {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Частота')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Скрыть пустые оси (если есть)
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('03_diabetes_features_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✅ Сохранен файл: 03_diabetes_features_distribution.png")


def scatter_features_vs_target(df):
    """Scatter plots признаков относительно целевой переменной"""
    features = df.columns[:-1]
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()
    
    colors = plt.cm.Set2.colors
    
    for i, feature in enumerate(features):
        # Scatter plot
        axes[i].scatter(df[feature], df['target'], alpha=0.6, color=colors[i % len(colors)], s=30)
        
        # Линия тренда (линейная регрессия) с обработкой NaN значений
        x_clean = np.nan_to_num(df[feature])
        y_clean = np.nan_to_num(df['target'])
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2,
                     label=f"Корреляция: {df[feature].corr(df['target']):.3f}")
        
        axes[i].set_title(f'{feature} vs Target')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Прогрессия болезни')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('03_diabetes_features_vs_target.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✅ Сохранен файл: 03_diabetes_features_vs_target.png")


def correlation_analysis(df):
    """Анализ корреляций"""
    correlations = df.corr()['target'].drop('target').sort_values(ascending=False)
    
    print("\n" + "="*50)
    print("КОРРЕЛЯЦИЯ ПРИЗНАКОВ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    print("="*50)
    for feature, corr in correlations.items():
        print(f"{feature:<10}: {corr:>8.4f}")  # Увеличена ширина для названий признаков
    
    # Горизонтальная столбчатая диаграмма
    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'steelblue' for x in correlations.values]
    bars = plt.barh(correlations.index, correlations.values, color=colors, alpha=0.8, edgecolor='black')
    
    # Добавить значения на бары
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (0.01 if width >= 0 else -0.03), 
                 bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', 
                 va='center', ha='left' if width >= 0 else 'right', fontweight='bold')
    
    plt.axvline(0, color='black', linewidth=0.8)
    plt.axvline(0.3, color='orange', linestyle='--', alpha=0.6)
    plt.axvline(-0.3, color='orange', linestyle='--', alpha=0.6)
    plt.xlabel('Коэффициент корреляции Пирсона')
    plt.title('Корреляция признаков с целевой переменной')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('03_diabetes_correlation_bars.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✅ Сохранен файл: 03_diabetes_correlation_bars.png")


def main():
    """Главная функция"""
    print("=" * 60)
    print("ЗАДАНИЕ 3: EXPLORATORY DATA ANALYSIS - DIABETES DATASET")
    print("=" * 60)

    df = load_data()
    print(f"\nДатасет загружен. Размер: {df.shape}")
    print("\nПервые 5 строк:")
    print(df.head())

    target_analysis(df)
    feature_statistics(df)
    visualize_target(df)
    visualize_features(df)
    scatter_features_vs_target(df)
    correlation_analysis(df)

    print("\n" + "=" * 60)
    print("Анализ завершен!")
    print("Созданы файлы визуализаций:")
    print(" - 03_diabetes_target_distribution.png")
    print(" - 03_diabetes_features_distribution.png")
    print(" - 03_diabetes_features_vs_target.png")
    print(" - 03_diabetes_correlation_bars.png")
    print("=" * 60)


if __name__ == "__main__":
    main()