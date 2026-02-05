# %% [markdown]
# # Step 2: Chunking Coherence Evaluation
# This script implements and evaluates different chunking strategies based on their semantic coherence.

# %% [markdown]
# ### 1. Install Dependencies
# Make sure you have the required libraries installed. You can run the following command in your terminal:
# `pip install langchain-text-splitters sentence-transformers langchain langchain_experimental langchain_community numpy matplotlib seaborn pandas scikit-learn "urllib3<2.0"`

# %%
# ### 2. Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import sent_tokenize
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

# Download the 'punkt' tokenizer
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# %% [markdown]
# ### 3. Load Data

# %%
def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return ""

text_files = [
    'extraction_output/page_2_extracted.md',
    'extraction_output/page_3_extracted.md'
]

full_text = ""
for text_path in text_files:
    content = read_file(text_path)
    if content:
        full_text += content + "\\n\\n"

if full_text.strip():
    print(f"Successfully loaded text. Total length: {len(full_text)} characters.")
else:
    print("No text loaded. Using dummy text for demonstration.")
    full_text = "This is a sample text for demonstration purposes. It talks about artificial intelligence and natural language processing. The goal is to see how different chunking strategies perform. A coherent chunk should contain sentences that are semantically related. We will test fixed-size, sentence-based, and semantic chunking. This text will be split into multiple chunks and then evaluated." * 10

# %% [markdown]
# ### 4. Initialize Models

# %%
print("Initializing embedding models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Models initialized.")

# %% [markdown]
# ### 5. Define Coherence Metric

# %%
def calculate_coherence_score(chunk_text: str) -> float:
    try:
        sentences = sent_tokenize(chunk_text)
    except Exception:
        return 0.0
        
    if len(sentences) <= 1:
        return 1.0
    
    embeddings = embedding_model.encode(sentences)
    centroid = np.mean(embeddings, axis=0).reshape(1, -1)
    similarities = cosine_similarity(embeddings, centroid)
    return np.mean(similarities)

# %% [markdown]
# ### 6. Define Chunking Strategies

# %%
def get_fixed_size_chunks(text: str, chunk_size: int, chunk_overlap: int = 20) -> List[str]:
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def get_sentence_chunks(text: str, n_sentences: int) -> List[str]:
    sentences = sent_tokenize(text)
    return [" ".join(sentences[i:i + n_sentences]) for i in range(0, len(sentences), n_sentences)]

def get_recursive_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def get_semantic_chunks(text: str) -> List[str]:
    splitter = SemanticChunker(hf_embeddings)
    return [doc.page_content for doc in splitter.create_documents([text])]

# %% [markdown]
# ### 7. Run Experiments

# %%
results = []
experiments = {
    "Fixed-200": lambda t: get_fixed_size_chunks(t, 200),
    "Fixed-500": lambda t: get_fixed_size_chunks(t, 500),
    "Fixed-1000": lambda t: get_fixed_size_chunks(t, 1000),
    "Sentence-3": lambda t: get_sentence_chunks(t, 3),
    "Sentence-5": lambda t: get_sentence_chunks(t, 5),
    "Sentence-10": lambda t: get_sentence_chunks(t, 10),
    "Recursive-500": lambda t: get_recursive_chunks(t, 500),
    "Semantic": get_semantic_chunks
}

print("Running experiments...")
for name, chunk_func in experiments.items():
    print(f"Processing: {name}...")
    try:
        chunks = chunk_func(full_text)
        if not chunks:
            print(f"  -> No chunks generated.")
            continue
        scores = [calculate_coherence_score(chunk) for chunk in chunks]
        results.append({
            "Chunking Method": name,
            "Avg Coherence": np.mean(scores),
            "Avg Variance": np.var(scores),
            "Num Chunks": len(chunks),
            "Scores": scores
        })
        print(f"  -> Found {len(chunks)} chunks. Avg Coherence: {np.mean(scores):.4f}")
    except Exception as e:
        print(f"  -> Error: {e}")

# %% [markdown]
# ### 8. Analyze and Visualize Results

# %%
# Create output directory
output_dir = "evaluation_results"
os.makedirs(output_dir, exist_ok=True)

if results:
    df_results = pd.DataFrame(results)
    
    # Save Table to CSV
    csv_path = os.path.join(output_dir, "coherence_results.csv")
    df_results.drop(columns=['Scores']).to_csv(csv_path, index=False)
    print(f"\nResults table saved to {csv_path}")
    
    # Display Table
    print("\n--- Results Summary ---")
    print(df_results.drop(columns=['Scores']))

    # Visualization 1: Bar Chart for Average Coherence
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_results, x='Chunking Method', y='Avg Coherence', palette='viridis')
    plt.title('Average Semantic Coherence by Chunking Strategy', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Coherence Score')
    plt.ylim(bottom=max(0, df_results['Avg Coherence'].min() * 0.9))
    plt.tight_layout()
    
    # Save Bar Chart
    bar_chart_path = os.path.join(output_dir, "avg_coherence_chart.png")
    plt.savefig(bar_chart_path)
    print(f"Bar chart saved to {bar_chart_path}")
    plt.show()

    # Visualization 2: Box Plot for Score Distribution
    all_scores_data = []
    for res in results:
        for score in res['Scores']:
            all_scores_data.append({'Method': res['Chunking Method'], 'Score': score})
    df_scores = pd.DataFrame(all_scores_data)

    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df_scores, x='Method', y='Score', palette='Set2')
    plt.title('Distribution of Coherence Scores per Strategy', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Coherence Score')
    plt.tight_layout()
    
    # Save Box Plot
    box_plot_path = os.path.join(output_dir, "coherence_distribution_boxplot.png")
    plt.savefig(box_plot_path)
    print(f"Box plot saved to {box_plot_path}")
    plt.show()
else:
    print("No results to display.")

# %% [markdown]
# ### 9. Recommendation

# %%
if results:
    best_method = df_results.sort_values(by='Avg Coherence', ascending=False).iloc[0]
    recommendation = (
        f"--- Recommendation ---\n"
        f"Best Strategy: {best_method['Chunking Method']}\n"
        f"Reason: It achieved the highest average coherence score of {best_method['Avg Coherence']:.4f}, \n"
        f"with a variance of {best_method['Avg Variance']:.4f} across {best_method['Num Chunks']} chunks."
    )
    print("\n" + recommendation)
    
    # Save Recommendation to text file
    rec_path = os.path.join(output_dir, "recommendation.txt")
    with open(rec_path, "w") as f:
        f.write(recommendation)
    print(f"Recommendation saved to {rec_path}")

else:
    print("Could not determine a recommendation as no experiments were successfully run.")

