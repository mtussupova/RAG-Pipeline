"""
Configuration for the Naive RAG pipeline (Stage 2).

In командном проекте здесь удобно хранить «оптимальные»
параметры, которые нашли одногруппники при экспериментах:
- k (количество документов из векторного поиска),
- параметры LLM (temperature, max_tokens, модель и т.п.).

При необходимости этот модуль можно заменить чтением
из .yaml/.json файла, чтобы не хардкодить значения.
"""

from dataclasses import dataclass


@dataclass
class RagConfig:
    # число документов, которое возвращает retriever
    k: int = 5

    # параметры LLM — значения-заглушки, чтобы была структура
    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.1
    max_tokens: int = 512


def load_config() -> RagConfig:
    """
    Вернуть конфигурацию для Naive RAG.

    Одногруппники после подбора гиперпараметров могут
    просто изменить значения по умолчанию или добавить
    логику чтения из файла.
    """
    return RagConfig()

