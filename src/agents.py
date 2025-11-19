import json
import os
from typing import Dict, Any

import requests
from dotenv import load_dotenv

from src.rag import RAGRetriever

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "deepseek/deepseek-r1-distill-llama-70b:free",
)

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "Не найден OPENROUTER_API_KEY в .env. Добавь строку:\n"
        "OPENROUTER_API_KEY=твой_ключ_от_OpenRouter"
    )


def call_openrouter(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": "Bearer {}".format(OPENROUTER_API_KEY),
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Matplotlib-RAG-Assistant",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

    if resp.status_code != 200:
        return (
            "Ошибка при обращении к OpenRouter API "
            "(status_code = {}, body = {})."
        ).format(resp.status_code, resp.text)

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return "OpenRouter API вернул пустой ответ: {}".format(data)

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        return "OpenRouter API не вернул content в message: {}".format(data)

    return content


def _extract_json_block(text: str) -> str:
    """Достаём JSON-объект из ответа модели (если она добавила лишний текст)."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def route_question(question: str) -> Dict[str, str]:
    """
    Агент 1: маршрутизатор.
    Возвращает словарь:
      {
        "category": "plot" | "theory" | "mixed",
        "search_query": "..."
      }
    """
    system_prompt = (
        "Ты маршрутизатор вопросов по matplotlib. "
        "Твоя задача — понять, что нужно пользователю и выдать JSON.\n"
        "Возможные категории:\n"
        "  - 'plot'   — нужно построить график / пример кода\n"
        "  - 'theory' — чистая теория по matplotlib (параметры, функции)\n"
        "  - 'mixed'  — и теория, и код\n\n"
        "Ответь ЧИСТЫМ JSON без пояснений, вида:\n"
        '{\"category\": \"plot\", \"search_query\": \"линейный график plt.plot\"}'
    )

    user_prompt = (
        "Вопрос пользователя:\n{}\n\n"
        "Подбери category и search_query (1 короткая фраза на русском "
        "или английском, по которой удобно искать в документации).".format(question)
    )

    raw = call_openrouter(system_prompt, user_prompt, temperature=0.1)

    try:
        data = json.loads(_extract_json_block(raw))
        category = data.get("category", "plot")
        if category not in ("plot", "theory", "mixed"):
            category = "plot"
        search_query = data.get("search_query") or question
    except Exception:
        category = "plot"
        search_query = question

    return {"category": category, "search_query": search_query}


_retriever = None


def _get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


def answer_question(question: str) -> Dict[str, Any]:
    """
    Агент 2: помощник по matplotlib.
    Использует:
      - результат маршрутизатора (route_question)
      - контекст из RAG (RAGRetriever)
      - LLM через OpenRouter для генерации ответа / кода.
    """
    routing = route_question(question)
    retriever = _get_retriever()

    context = retriever.make_context(routing["search_query"], top_k=4)

    system_prompt = (
        "Ты опытный преподаватель Python и библиотеки matplotlib.\n"
        "У тебя есть фрагменты документации (context) и вопрос пользователя.\n\n"
        "Требования к ответу:\n"
        "1) Если категория 'plot' или 'mixed' — обязательно дай завершённый "
        "пример кода на Python с matplotlib (и seaborn при необходимости) "
        "в блоке ```python ...```.\n"
        "2) Объясняй шаги простым языком.\n"
        "3) Старайся опираться на context, не выдумывать из головы.\n"
        "4) Если чего-то нет в context, можешь дополнить общими знаниями, "
        "но не противоречь context.\n"
        "5) Ответ на русском."
    )

    user_prompt = (
        "Категория: {cat}\n\n"
        "Вопрос пользователя:\n{q}\n\n"
        "Фрагменты документации (context):\n{ctx}\n\n"
        "Сформируй подробный, но по делу ответ.".format(
            cat=routing["category"],
            q=question,
            ctx=context,
        )
    )

    answer_text = call_openrouter(system_prompt, user_prompt, temperature=0.25)

    return {
        "routing": routing,
        "context": context,
        "answer": answer_text,
    }
