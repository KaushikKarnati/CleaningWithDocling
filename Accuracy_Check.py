import numpy as np
import pandas as pd
import re
import string
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\\s+", " ", text).strip()  # Remove extra spaces
    return text

# Load original text from Markdown file
def load_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        text = re.sub(r"[#*`>-]", "", text)  # Remove markdown symbols
        text = re.sub("\\s+", " ", text).strip()
        return preprocess_text(text)

# Load Q&A from JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            messages = entry.get("messages", [])
            
            # Extract user question and assistant answer
            question = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
            answer = next((msg["content"] for msg in messages if msg["role"] == "assistant"), None)
            
            if question and answer:
                data.append({"Question": question, "Generated Answer": answer})
    
    return pd.DataFrame(data)

# Generate BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Compute BERT-based similarity
def compute_bert_similarity(original_text, qa_dataframe):
    results = []
    original_embedding = get_bert_embedding(original_text)
    
    for index, row in qa_dataframe.iterrows():
        question = preprocess_text(row["Question"])
        generated_answer = preprocess_text(row["Generated Answer"])
        
        answer_embedding = get_bert_embedding(generated_answer)
        similarity_score = cosine_similarity(original_embedding, answer_embedding)[0][0]
        
        results.append({
            "Question": row["Question"],
            "Generated Answer": row["Generated Answer"],
            "BERT Similarity Score": similarity_score
        })
    
    return pd.DataFrame(results)

# Visualize similarity scores and save results
def visualize_and_save_results(result_df, threshold=0.7, csv_output="bert_similarity_results.csv", txt_output="low_quality_questions.txt"):
    # Save full results to CSV
    result_df.to_csv(csv_output, index=False)
    print(f"Full similarity results saved to: {csv_output}")

    # Plot similarity scores
    plt.figure(figsize=(10, 5))
    sns.histplot(result_df["BERT Similarity Score"], bins=20, kde=True)
    plt.axvline(threshold, color='red', linestyle='dashed', label=f'Threshold = {threshold}')
    plt.xlabel("BERT Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Similarity Scores")
    plt.legend()
    plt.show()

    # Filter low-quality answers
    low_quality = result_df[result_df["BERT Similarity Score"] < threshold]
    
    if not low_quality.empty:
        print("Low-Quality Answers (Below Threshold):")
        print(low_quality[["Question", "Generated Answer", "BERT Similarity Score"]])
        
        # Save low-quality questions to a text file
        with open(txt_output, "w", encoding="utf-8") as f:
            for _, row in low_quality.iterrows():
                f.write(f"Question: {row['Question']}\n")
                f.write(f"Generated Answer: {row['Generated Answer']}\n")
                f.write(f"Similarity Score: {row['BERT Similarity Score']:.4f}\n\n")
        
        print(f"Low-quality questions saved to: {txt_output}")
    else:
        print("No low-quality answers detected!")

# Example Usage
if __name__ == "__main__":
    markdown_path = "extracted_markdown/Genome Manipulation Advances in Selected.md"  # Replace with actual file path
    jsonl_path = "jsonl_output/Genome Manipulation Advances in Selected.jsonl"  # Replace with actual file path
    
    original_text = load_markdown(markdown_path)
    qa_data = load_jsonl(jsonl_path)
    
    result_df = compute_bert_similarity(original_text, qa_data)
    
    # Visualize and save results
    visualize_and_save_results(result_df, threshold=0.7)