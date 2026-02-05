# RAG Pipeline - Data Extraction & Validation

## Step 1: PDF Data Extraction with Quality Validation

### Overview

`step1-retrieval-docling.py` extracts text from PDF documents using Docling and validates extraction quality using a Visual LLM (GPT-4o) as a judge.

### Requirements

```bash
pip install docling pymupdf openai python-dotenv
```

### Environment Setup

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

### Configuration

Edit these variables in `step1-retrieval-docling.py`:

```python
PDF_PATH = "kaztelecom.pdf"      # Path to PDF file
PAGES_TO_EXTRACT = [2, 3]        # Pages to process (1-indexed)
OUTPUT_DIR = Path("extraction_output")  # Output directory
```

### Usage

```bash
python step1-retrieval-docling.py
```

### Output

The script creates `extraction_output/` directory with:

| File | Description |
|------|-------------|
| `page_N.png` | Screenshot of page N |
| `page_N_extracted.md` | Extracted text/markdown from page N |
| `validation_results.json` | LLM evaluation scores |

### Validation Criteria

The Visual LLM evaluates extraction quality on a 1-5 scale:

- **Structure**: Headers (H1, H2, H3), lists preservation
- **Tables**: Structure, cell readability, alignment
- **Formatting**: Bold, italic, special characters
- **Completeness**: All text extracted, no missing blocks

### Output Format

`validation_results.json` structure:

```json
[
  {
    "page_num": 2,
    "extraction_method": "docling",
    "evaluation": {
      "structure_score": 4,
      "tables_score": 1,
      "formatting_score": 3,
      "completeness_score": 5,
      "overall_score": 3.25,
      "comments": "..."
    }
  }
]
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `create_page_screenshot(pdf_path, page_num, dpi)` | Renders PDF page to PNG |
| `extract_text_with_docling(pdf_path, pages)` | Extracts text using Docling |
| `validate_extraction_with_llm(screenshot, text, client)` | Sends to GPT-4o for evaluation |

### Programmatic Usage

```python
from step1_retrieval_docling import (
    create_page_screenshot,
    extract_text_with_docling,
    validate_extraction_with_llm,
    encode_image_to_base64
)
from openai import OpenAI

# Extract text
texts = extract_text_with_docling("document.pdf", [1, 2])

# Create screenshot
screenshot = create_page_screenshot("document.pdf", 1)

# Validate
client = OpenAI()
result = validate_extraction_with_llm(
    encode_image_to_base64(screenshot),
    texts[1],
    client
)
```

---

## Step 2: Chunking Coherence Evaluation

### Overview

`step2_chunking_coherence_evaluation.py` takes the extracted text from Step 1 and evaluates various text chunking strategies. The goal is to find the strategy that produces the most semantically coherent chunks, which is crucial for retrieval quality in a RAG system.

### Requirements

Activate your virtual environment (`source venv/bin/activate`) and run:
```bash
pip install langchain-text-splitters sentence-transformers langchain langchain_experimental langchain_community numpy matplotlib seaborn pandas scikit-learn "urllib3<2.0"
```

### How it Works

The script performs the following actions:
1.  **Loads Data**: Reads the `.md` files from the `extraction_output/` directory.
2.  **Applies Chunking Strategies**: It splits the text using different methods:
    - Fixed-size chunking (by token count)
    - Sentence-based chunking (by number of sentences)
    - Recursive chunking
    - Semantic chunking
3.  **Calculates Coherence**: For each chunk, it calculates a "Semantic Coherence Score" by embedding the sentences within the chunk and measuring their cosine similarity to the chunk's average embedding (centroid).
4.  **Generates Results**: It analyzes the scores for all strategies and determines the best one.

### Usage

Ensure you have run Step 1 first to generate the `extraction_output` directory. Then, run the script from your terminal:
```bash
python step2_chunking_coherence_evaluation.py
```

### Output

The script creates the `evaluation_results/` directory with the following files:

| File | Description |
|------|-------------|
| `coherence_results.csv` | A CSV table summarizing the performance of each chunking method (Avg Coherence, Avg Variance, Num Chunks). |
| `avg_coherence_chart.png` | A bar chart visualizing the average coherence score for each strategy. |
| `coherence_distribution_boxplot.png` | A box plot showing the distribution of coherence scores for each strategy. |
| `recommendation.txt` | A text file with a recommendation for the best-performing chunking strategy based on the analysis. |
