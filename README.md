Проект — консольный ассистент по построению графиков в Python (matplotlib + seaborn).

Пользователь пишет запрос, например:

- "Построй круговую диаграмму по массиву months.revenue"
- "Даны массивы Y и X, построй график y от x"

Ассистент:

1. Определяет, что именно нужно (тип графика / теория / смешанное).
2. Ищет подходящие фрагменты в локальной базе документации (RAG).
3. Генерирует понятный ответ и готовый пример кода `python` с использованием matplotlib/seaborn.

Используется:

- внешняя LLM через OpenRouter API,
- RAG по локальным `.md` файлам с документацией,
- 2 LLM-агента (маршрутизатор + ассистент),
- Консольный интерфейс.

---

## Основные возможности

- Ответы на вопросы по построению графиков: гистограммы, линейные графики, scatter, boxplot, heatmap и т.д.
- Генерация готового к запуску кода на Python с matplotlib/seaborn.
- Объяснение шагов построения графика простым языком.
- RAG (Retrieval-Augmented Generation):
  - ассистент опирается на локальные `.md` файлы с документацией и примерами.
- Мультиагентная архитектура:
  - Агент 1 - маршрутизатор: классифицирует запрос и формирует поисковый запрос для RAG.
  - Агент 2 - ассистент по matplotlib: собирает контекст, обращается к LLM и формирует ответ.

---

## Установка и запуск:

```bash
git clone https://github.com/Kamyshanskii/Graph_creator.git
cd Graph_creator

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env

# Получите API-ключ на https://openrouter.ai/settings/keys и присвойте его в OPENROUTER_API_KEY в файле .env

python -m src.build_index 
python -m src.app_cli
```

