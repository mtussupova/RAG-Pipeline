"""
Entry point for the Naive RAG pipeline (Stage 2: integration).

This script is responsible for:
- loading optimal RAG parameters,
- initializing the RAG pipeline using components prepared by teammates
  (vector store, retriever, LLM configuration),
- providing a simple CLI to interact with the system.
"""

from rag_config import load_config
from pipeline import init_rag_pipeline, run_cli_session


def main() -> None:
    """
    Initialize the Naive RAG pipeline with optimal parameters
    and start an interactive CLI loop.
    """
    config = load_config()
    rag_chain = init_rag_pipeline(config)
    run_cli_session(rag_chain, config)


if __name__ == "__main__":
    main()

