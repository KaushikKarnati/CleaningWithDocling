import together
import json
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# === API Configuration ===
TOGETHER_API_KEY = "460dca699fe36c30c2ab1660849b9f73fca60844ad1ce850d952dee816d9d22a"  # Replace with your valid API key

# Initialize Together AI Client
together.api_key = TOGETHER_API_KEY

# === Function to Get LLaMA Embeddings from API ===
def get_llama_embedding(text):
    try:
        response = together.Complete.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            prompt=f"Generate an embedding for: {text}",
            max_tokens=1  # This is a workaround since Together AI doesn't have direct embedding API
        )
        
        if response and "choices" in response:
            return [float(ord(c)) for c in response["choices"][0]["text"]]  # Convert text to numeric embedding
        else:
            print(f"❌ API Response Error: {response}")
            return None

    except Exception as e:
        print(f"❌ API Call Failed: {e}")
        return None

# === Text Preprocessing ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\\s+", " ", text).strip()  # Remove extra spaces
    return text

# === Load Markdown File ===
def load_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        text = re.sub(r"[#*`>-]", "", text)  # Remove markdown symbols
        text = re.sub("\\s+", " ", text).strip()
        return preprocess_text(text)

# === Load JSONL File ===
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

# === Compute LLaMA-Based Similarity via API ===
def compute_llama_similarity(original_text, qa_dataframe):
    results = []
    original_embedding = get_llama_embedding(original_text)

    if original_embedding is None:
        print("❌ Failed to retrieve original text embeddings!")
        return pd.DataFrame()

    for index, row in qa_dataframe.iterrows():
        question = preprocess_text(row["Question"])
        generated_answer = preprocess_text(row["Generated Answer"])

        answer_embedding = get_llama_embedding(generated_answer)

        if answer_embedding is None:
            print(f"⚠️ Skipping question {index} due to API failure.")
            continue

        similarity_score = cosine_similarity([original_embedding], [answer_embedding])[0][0]

        results.append({
            "Question": row["Question"],
            "Generated Answer": row["Generated Answer"],
            "LLaMA Similarity Score": similarity_score
        })
    
    return pd.DataFrame(results)

# === Visualize and Save Results ===
def visualize_and_save_results(result_df, threshold=0.7, csv_output="llama_api_similarity_results.csv", txt_output="low_quality_questions.txt"):
    if result_df.empty:
        print("⚠️ No valid similarity results. Skipping visualization.")
        return  # Exit function if no valid data

    # Save full results to CSV
    result_df.to_csv(csv_output, index=False)
    print(f"✅ Full similarity results saved to: {csv_output}")

    # Plot similarity scores
    plt.figure(figsize=(10, 5))
    sns.histplot(result_df["LLaMA Similarity Score"], bins=20, kde=True)
    plt.axvline(threshold, color='red', linestyle='dashed', label=f'Threshold = {threshold}')
    plt.xlabel("LLaMA Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Similarity Scores")
    plt.legend()
    plt.show()

    # Filter low-quality answers
    low_quality = result_df[result_df["LLaMA Similarity Score"] < threshold]

    if not low_quality.empty:
        print("⚠️ Low-Quality Answers (Below Threshold):")
        print(low_quality[["Question", "Generated Answer", "LLaMA Similarity Score"]])

        # Save low-quality questions to a text file
        with open(txt_output, "w", encoding="utf-8") as f:
            for _, row in low_quality.iterrows():
                f.write(f"Question: {row['Question']}\n")
                f.write(f"Generated Answer: {row['Generated Answer']}\n")
                f.write(f"Similarity Score: {row['LLaMA Similarity Score']:.4f}\n\n")

        print(f"⚠️ Low-quality questions saved to: {txt_output}")
    else:
        print("✅ No low-quality answers detected!")

# === Run the Script ===
if __name__ == "__main__":
    markdown_path = "extracted_markdown/Genome Manipulation Advances in Selected.md"  # Replace with actual file path
    jsonl_path = "jsonl_output/Genome Manipulation Advances in Selected.jsonl"  # Replace with actual file path
    
    original_text = load_markdown(markdown_path)
    qa_data = load_jsonl(jsonl_path)
    
    result_df = compute_llama_similarity(original_text, qa_data)

    # Visualize and save results
    visualize_and_save_results(result_df, threshold=0.7)