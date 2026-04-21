import os
import json
import ast
import traceback
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util

OUTPUT_DIR = "eda_outputs"
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "eda_summary.txt")

def log_summary(text):
    """Utility to print to console and write to the summary file."""
    print(text)
    with open(SUMMARY_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # clear or create the summary file
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("EDA Pipeline Report\n")
        f.write("===================\n\n")

def load_data(filepath="dataset.jsonl"):
    log_summary(f"Loading data from {filepath}...")
    records = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        log_summary(f"Error parsing JSON on line {line_no}")
    except FileNotFoundError:
        log_summary(f"File {filepath} not found.")
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    log_summary(f"Loaded {len(df)} records.")
    return df

def plot_token_distributions(df):
    log_summary("--- Token Distribution Analysis ---")
    model_id = "ByteDance-Seed/Seed-Coder-8B-Instruct"
    log_summary(f"Loading tokenizer: {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        log_summary(f"Failed to load tokenizer {model_id}: {e}")
        return df
    
    desc_lengths = []
    code_lengths = []
    
    for idx, row in df.iterrows():
        desc = row.get("description", "")
        code = row.get("manim_code", "")
        desc_lengths.append(len(tokenizer.encode(desc)) if desc else 0)
        code_lengths.append(len(tokenizer.encode(code)) if code else 0)
        
    df["desc_tokens"] = desc_lengths
    df["code_tokens"] = code_lengths
    
    # Calculate stats
    for col, name in zip(["desc_tokens", "code_tokens"], ["Description", "Manim Code"]):
        p90 = np.percentile(df[col], 90)
        p95 = np.percentile(df[col], 95)
        p99 = np.percentile(df[col], 99)
        log_summary(f"{name} tokens -> 90th: {p90:.2f}, 95th: {p95:.2f}, 99th: {p99:.2f}")

    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df["desc_tokens"], bins=20, kde=True, color='blue')
    plt.title("Description Token Length Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    sns.histplot(df["code_tokens"], bins=20, kde=True, color='green')
    plt.title("Manim Code Token Length Distribution")
    plt.xlabel("Token Count")
    
    out_path = os.path.join(OUTPUT_DIR, "token_distributions.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    log_summary(f"Saved token distributions plot to {out_path}\n")
    return df

class ManimClassVisitor(ast.NodeVisitor):
    def __init__(self):
        self.class_counts = Counter()
        self.common_manim_classes = {
            "Scene", "Text", "MathTex", "Create", "Write", "FadeIn", "FadeOut", 
            "Indicate", "Transform", "ReplacementTransform", "VGroup", 
            "Rectangle", "Circle", "Dot", "Line", "Arrow", "NumberLine", "Cross"
        }

    def visit_Call(self, node):
        # We look for direct function/class calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.common_manim_classes:
                self.class_counts[func_name] += 1
        self.generic_visit(node)

def validate_and_extract_ast(df):
    log_summary("--- AST Validation & Class Frequency ---")
    valid_indices = []
    visitor = ManimClassVisitor()
    syntax_errors = 0

    for idx, row in df.iterrows():
        code = row.get("manim_code", "")
        # Remove any leading spaces or lines that might break top-level parse
        try:
            tree = ast.parse(code)
            visitor.visit(tree)
            valid_indices.append(idx)
        except SyntaxError as e:
            syntax_errors += 1
            log_summary(f"SyntaxError in row {idx}: {e}")
        except Exception as e:
            syntax_errors += 1
            log_summary(f"Unexpected error parsing row {idx}: {e}")
            
    df_clean = df.loc[valid_indices].copy()
    log_summary(f"Dropped {syntax_errors} rows with invalid syntax.")
    log_summary(f"Rows remaining: {len(df_clean)}")

    # Plot top 20 classes
    top_classes = visitor.class_counts.most_common(20)
    log_summary(f"Top 5 Manim classes instantiated: {top_classes[:5]}")
    
    if top_classes:
        labels, counts = zip(*top_classes)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(labels), hue=list(labels), palette='viridis', legend=False)
        plt.title("Top 20 Most Frequent Manim Classes Instantiated")
        plt.xlabel("Frequency")
        plt.ylabel("Class Name")
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "class_frequency.png")
        plt.savefig(out_path)
        plt.close()
        log_summary(f"Saved class frequency bar plot to {out_path}\n")
    else:
        log_summary("No corresponding Manim classes found.\n")

    return df_clean

def analyze_lexical_diversity_and_ratios(df):
    log_summary("--- Lexical Diversity & Ratios ---")
    if df.empty:
        log_summary("DataFrame is empty, skipping.")
        return df

    # Lexical Diversity (N-grams)
    descriptions = df['description'].fillna("").tolist()
    # Using ngram_range for bi-grams and tri-grams
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english', max_features=10)
    try:
        X = vectorizer.fit_transform(descriptions)
        sum_words = X.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        log_summary(f"Top repeating prompt templates (bi-grams & tri-grams):")
        for word, freq in words_freq:
            log_summary(f"  '{word}': {freq}")
    except ValueError as e:
        log_summary(f"N-gram generation failed (perhaps empty vocab): {e}")

    # Ratios
    df['desc_char_len'] = df['description'].apply(lambda x: len(str(x)))
    df['code_char_len'] = df['manim_code'].apply(lambda x: len(str(x)))
    df['ratio_desc_to_code'] = df['desc_char_len'] / (df['code_char_len'] + 1e-9)

    mean_ratio = df['ratio_desc_to_code'].mean()
    log_summary(f"Mean Description/Code character ratio: {mean_ratio:.4f}")

    # Correlation Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='desc_char_len', y='code_char_len', color='purple', alpha=0.6)
    plt.title("Correlation: Description Length vs Code Length")
    plt.xlabel("Description Character Length")
    plt.ylabel("Manim Code Character Length")
    out_path = os.path.join(OUTPUT_DIR, "length_correlation.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    log_summary(f"Saved correlation plot to {out_path}\n")

    return df

def deduplicate_dataset(df):
    log_summary("--- Deduplication & Semantic Near-Duplicates ---")
    original_count = len(df)
    
    # 1. Exact Deduplication
    df = df.drop_duplicates(subset=['manim_code'])
    exact_duplicates_dropped = original_count - len(df)
    log_summary(f"Exact duplicates dropped: {exact_duplicates_dropped}")
    
    if df.empty:
        return df

    # 2. Semantic Near-Duplicates
    model_name = 'all-MiniLM-L6-v2'
    log_summary(f"Loading sentence-transformers model: {model_name} ...")
    embedder = SentenceTransformer(model_name)
    
    code_corpus = df['manim_code'].tolist()
    log_summary("Encoding code snippets...")
    corpus_embeddings = embedder.encode(code_corpus, convert_to_tensor=True, show_progress_bar=False)
    
    log_summary("Computing cosine similarities...")
    # Compute dot score (since all-MiniLM normalizes, dot score is cos sim)
    cosine_scores = util.cos_sim(corpus_embeddings, corpus_embeddings)
    
    threshold = 0.95
    near_dupes_flagged = set()
    pairs = []
    
    for i in range(len(cosine_scores)):
        for j in range(i + 1, len(cosine_scores)):
            val = cosine_scores[i][j].item()
            if val > threshold:
                pairs.append((i, j, val))
                near_dupes_flagged.add(i)
                near_dupes_flagged.add(j)

    log_summary(f"Flagged {len(pairs)} pairs with cosine similarity > {threshold}")
    log_summary(f"Total rows involved in near-duplicates: {len(near_dupes_flagged)}")
    
    if pairs:
        log_summary("Top 3 closest pairs:")
        # sort by similarity descending
        pairs.sort(key=lambda x: x[2], reverse=True)
        for i, j, score in pairs[:3]:
            # Fetch snippet lengths to see what we are comparing
            len_i = len(code_corpus[i])
            len_j = len(code_corpus[j])
            log_summary(f"  Score: {score:.4f} between row {i} (len {len_i}) and row {j} (len {len_j})")

    return df

def main():
    ensure_output_dir()
    df = load_data()
    
    if df.empty:
        log_summary("No data to process. Exiting.")
        return
        
    df = plot_token_distributions(df)
    df = validate_and_extract_ast(df)
    df = analyze_lexical_diversity_and_ratios(df)
    df = deduplicate_dataset(df)
    
    log_summary("EDA Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
