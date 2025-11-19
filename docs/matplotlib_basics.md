# Основы matplotlib и seaborn для построения графиков

В примерах ниже предполагается, что у вас есть `pandas.DataFrame` с именем `df`.

---

## Гистограмма (histogram) с matplotlib

```python
import matplotlib.pyplot as plt

plt.hist(df["column_name"], bins=30, 
    color='skyblue',          # цвет столбцов
    edgecolor='black',        # цвет границ
    linewidth=1,              # толщина границ
    alpha=0.7,                # прозрачность
    density=False)            # нормализация

plt.xlabel("column_name", fontsize=12, color='navy')
plt.ylabel("Frequency", fontsize=12, color='navy')
plt.title("Histogram of column_name", fontsize=14, color='darkblue', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.show()
```


---

## Линейный график (line plot) с matplotlib

```python

import matplotlib.pyplot as plt

plt.plot(df["x"], df["y"], 
         color='red',              # цвет линии
         linewidth=2,              # толщина линии
         marker='o',               # маркеры точек
         markersize=4,             # размер маркеров
         markerfacecolor='blue',   # цвет маркеров
         markeredgecolor='black',  # цвет границ маркеров
         markeredgewidth=1,        # толщина границ маркеров
         linestyle='-',            # стиль линии (-, --, :, -.)
         alpha=0.8)                # прозрачность

plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.title("Line plot of y over x", fontsize=14)
plt.grid(True, color='gray', alpha=0.3, linestyle=':')
plt.show()
```
---

## Гистограмма с seaborn

```python

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=df, x="column_name", bins=30,
             color='green',                             # цвет заливки
             edgecolor='white',                         # цвет границ
             linewidth=1.5,                             # толщина границ
             alpha=0.6,                                 # прозрачность
             kde=True,                                  # ядерная оценка плотности
             kde_kws={'color': 'red', 'linewidth': 2})  # параметры KDE

plt.xlabel("column_name", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Histogram of column_name (seaborn)", fontsize=14)
plt.show()
```

---

## График рассеяния (scatter plot) с seaborn

```python

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=df, x="x", y="y", hue="category",
                palette='viridis',  # цветовая палитра
                size=5,             # размер точек
                sizes=(20, 200),    # диапазон размеров (если size передает переменную)
                alpha=0.7,          # прозрачность
                edgecolor='black',  # цвет границ точек
                linewidth=0.5,      # толщина границ
                marker='o')         # форма маркера

plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.title("Scatter plot of y over x, colored by category", fontsize=14)
plt.legend(frameon=True, framealpha=0.8, edgecolor='black')
plt.show()
```

---

## Boxplot с seaborn

```python

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=df, x="category", y="value",
            palette='Set3',                                                         # цветовая палитра
            linewidth=1.5,                                                          # толщина линий
            fliersize=3,                                                            # размер выбросов
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 4},
            whiskerprops={'color': 'black', 'linewidth': 1},
            capprops={'color': 'black', 'linewidth': 1},
            medianprops={'color': 'red', 'linewidth': 2})

plt.xlabel("category", fontsize=12)
plt.ylabel("value", fontsize=12)
plt.title("Boxplot of value by category", fontsize=14)
plt.xticks(rotation=45)
plt.show()
```

---

## Тепловая карта корреляции (heatmap) с seaborn

```python

import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f",
            cmap='coolwarm',                                    # цветовая карта
            center=0,                                           # центр цветовой шкалы
            square=True,                                        # квадратные ячейки
            linewidths=0.5,                                     # толщина линий между ячейками
            linecolor='white',                                  # цвет линий между ячейками
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},   # параметры цветовой шкалы
            annot_kws={'size': 10, 'color': 'black'})           # параметры аннотаций
            
plt.title("Correlation Heatmap", fontsize=14, pad=20)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

---

## Круговая столбчатая диаграмма (Polar bar chart)

```python

import matplotlib.pyplot as plt
import numpy as np

categories = ['North', 'N-E', 'East', 'S-E', 'South', 'S-W', 'West', 'N-W']
values = [7, 5, 8, 6, 9, 4, 7, 5]
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
bars = ax.bar(angles, values, width=0.5, 
              alpha=0.7,                                                    # прозрачность
              color=plt.cm.viridis(np.linspace(0, 1, len(categories))),
              edgecolor='black',                                            # цвет границ
              linewidth=1,                                                  # толщина границ
              linestyle='-')                                                # стиль границ

ax.set_xticks(angles)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, max(values) + 2)
ax.set_title('Polar Bar Chart - Directional Data', pad=30, fontsize=14)
ax.grid(True, color='gray', alpha=0.3)

for angle, value, bar in zip(angles, values, bars):
    ax.text(angle, value + 0.5, str(value), 
            ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

plt.show()
```

---

## График потоков (streamplot)

```python

import matplotlib.pyplot as plt
import numpy as np

Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2

fig, ax = plt.subplots(figsize=(10, 8))
speed = np.sqrt(U**2 + V**2)
strm = ax.streamplot(X, Y, U, V, 
                     density=2,              # плотность линий
                     color=speed,            # цвет по скорости
                     linewidth=1.5,          # толщина линий
                     cmap='plasma',          # цветовая карта
                     arrowsize=1.5,          # размер стрелок
                     arrowstyle='->',        # стиль стрелок
                     minlength=0.1,          # минимальная длина линии
                     maxlength=4.0)          # максимальная длина линии

ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Streamplot - Vector Field Flow', fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.colorbar(strm.lines, ax=ax, label='Flow Speed', shrink=0.8)
plt.show()
```

---

## Круговая диаграмма (PieChart)

```python

import matplotlib.pyplot as plt

categories = ['Category A', 'Category B', 'Category C', 'Category D']
sizes = [25, 35, 20, 20]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=categories, colors=colors, 
        autopct='%1.1f%%',                                                    # формат процентов
        startangle=90,                                                        # начальный угол
        shadow=True,                                                          # тень
        explode=(0, 0.1, 0, 0),                                               # выдвижение секторов
        textprops={'fontsize': 12, 'color': 'darkblue'},                      # параметры текста
        wedgeprops={'edgecolor': 'black', 'linewidth': 2, 'linestyle': '-'})  # параметры секторов

plt.title('Pie Chart - Distribution of Categories', fontsize=14, pad=20)
plt.axis('equal')
plt.show()
'''