"""
Naive RAG pipeline wiring and simple CLI.

Задача этого модуля — ИНТЕГРАЦИЯ:
- взять готовое векторное хранилище и retriever (их делают одногруппники),
- взять LLM/chain (тоже может быть оформлено ими),
- собрать из этого единый RAG-процесс и удобный интерфейс.
"""

from typing import Any, Dict

from rag_config import RagConfig


def _build_retriever(config: RagConfig):
    """
    Создать/получить retriever из векторного хранилища.

    Предполагается, что одногруппники реализуют, например, модуль
    `vector_store.py` с функцией `build_vector_store()`, которая
    возвращает объект LangChain VectorStore (FAISS/Chroma/...).
    Тогда здесь можно будет сделать:

        from vector_store import build_vector_store
        vector_store = build_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={\"k\": config.k})

    Сейчас оставляем минимальную заглушку, чтобы структура была понятна.
    """
    try:
        from vector_store import build_vector_store  # type: ignore
    except ImportError as exc:  # pragma: no cover - до интеграции
        raise RuntimeError(
            "Модуль `vector_store` с функцией `build_vector_store()` "
            "должен быть реализован одногруппниками."
        ) from exc

    vector_store = build_vector_store()
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": config.k})
    except AttributeError as exc:  # pragma: no cover - защита от некорректной реализации
        raise RuntimeError(
            "Ожидается, что объект vector_store поддерживает метод "
            "`as_retriever(search_kwargs={\"k\": ...})` (как в LangChain)."
        ) from exc

    return retriever


def _build_llm(config: RagConfig):
    """
    Создать LLM/chain для ответа с использованием контекста.

    Здесь предполагается использование LangChain (или аналога).
    Конкретная реализация может отличаться — главное, чтобы на выходе
    получилась цепочка, у которой можно вызвать `invoke()` или `__call__()`.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - до установки зависимостей
        raise RuntimeError(
            "Не удалось импортировать `langchain_openai.ChatOpenAI`. "
            "Установите зависимости (например, `pip install langchain-openai`)."
        ) from exc

    # Параметры модели берём из RagConfig — сюда одногруппники подставят
    # оптимальные значения после экспериментов.
    llm = ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return llm


def init_rag_pipeline(config: RagConfig):
    """
    Собрать Naive RAG пайплайн из:
    - retriever (векторное хранилище + k),
    - LLM,
    - RAG-цепочки (RetrievalQA или create_retrieval_chain).
    """
    retriever = _build_retriever(config)

    # Здесь один из типовых вариантов через LangChain.
    try:
        from langchain.chains import RetrievalQA  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Не удалось импортировать `langchain.chains.RetrievalQA`. "
            "Установите зависимости LangChain."
        ) from exc

    llm = _build_llm(config)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    return rag_chain


def run_cli_session(rag_chain: Any, config: RagConfig) -> None:
    """
    Простой CLI для демонстрации работы Naive RAG.

    - показывает использующиеся параметры (k, модель),
    - даёт возможность интерактивно задавать вопросы,
    - по желанию может выводить источники (метаданные документов).
    """
    print("=== Naive RAG (Stage 2: интеграция) ===")
    print(f"Модель LLM: {config.model_name}")
    print(f"k (число документов в контексте): {config.k}")
    print("Введите вопрос (или пустую строку / 'exit' для выхода).")
    print()

    while True:
        query = input(">>> ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            print("Завершение сессии RAG.")
            break

        try:
            result: Dict[str, Any] = rag_chain.invoke({"query": query})  # type: ignore[arg-type]
        except TypeError:
            # На случай, если реализован другой интерфейс LangChain:
            result = rag_chain({"query": query})  # type: ignore[assignment]

        # В зависимости от конкретной реализации цепочки ключи могут отличаться.
        answer = result.get("result") or result.get("answer") or str(result)
        print("\nОтвет:")
        print(answer)

        sources = result.get("source_documents") or result.get("sources")
        if sources:
            print("\nИсточники:")
            for idx, doc in enumerate(sources, start=1):
                meta = getattr(doc, "metadata", {}) or {}
                source = meta.get("source") or meta.get("title") or f"doc_{idx}"
                print(f"- {idx}. {source}")

        print()

