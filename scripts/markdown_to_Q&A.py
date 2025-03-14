import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import tiktoken  # Install with `pip install tiktoken`
from together import Together

class Config:
    def __init__(self):
        self.CHUNK_SIZE = 512  # Adjusted for RAG optimization
        self.SYSTEM_PROMPTS = [
            'You are an expert aquaculture researcher with extensive knowledge of marine biology, fish farming, and sustainable aquaculture practices.',
            'You are a specialized AI assistant with deep expertise in aquaculture science, focusing on research methodology, water quality management, and aquatic species cultivation.',
            'You are an aquaculture specialist with comprehensive knowledge of both theoretical and practical aspects of aquatic farming systems.'
        ]

class MarkdownProcessor:
    def __init__(self, markdown_folder="extracted_markdown", jsonl_folder="jsonl_output", json_output_path="chunked_output.json"):
        """Initialize with Together AI API key and setup JSON and JSONL output files."""
        self.config = Config()
        self.markdown_folder = markdown_folder
        self.jsonl_folder = jsonl_folder
        self.json_output_path = json_output_path
        self.client = Together(api_key="460dca699fe36c30c2ab1660849b9f73fca60844ad1ce850d952dee816d9d22a")

        # Create output folders
        os.makedirs(self.jsonl_folder, exist_ok=True)

        # Token tracking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.total_tokens = 0

    def read_markdown_file(self, md_path: str) -> str:
        """Read content from markdown file."""
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
            
        with open(md_path, 'r', encoding='utf-8') as file:
            return file.read()

    def split_into_chunks(self, content: str):
        """Split markdown content into logical chunks optimized for RAG."""
        chunks = []
        chunk_id = 1
        current_chunk = []
        current_size = 0

        for line in content.split('\n'):
            if line.strip().startswith('#'):  # Header detected
                if current_chunk:
                    chunks.append({
                        "chunk_id": str(chunk_id),
                        "text": '\n'.join(current_chunk)
                    })
                    chunk_id += 1
                current_chunk = [line]
                current_size = len(line)
            else:
                if current_size + len(line) > self.config.CHUNK_SIZE:
                    chunks.append({
                        "chunk_id": str(chunk_id),
                        "text": '\n'.join(current_chunk)
                    })
                    chunk_id += 1
                    current_chunk = [line]
                    current_size = len(line)
                else:
                    current_chunk.append(line)
                    current_size += len(line)

        if current_chunk:
            chunks.append({
                "chunk_id": str(chunk_id),
                "text": '\n'.join(current_chunk)
            })

        return chunks

    def generate_qa_pairs(self, text: str):
        """Generate Q&A pairs from markdown content using Together AI API."""
        prompt = f"""
        Given the following text from a technical document, generate 3-4 detailed question-answer pairs.
        Focus on key concepts, methodologies, and important technical details.

        Text:
        {text}

        Generate the response in this exact JSON format:
        [
            {{"question": "Detailed question about the content?", "answer": "Comprehensive answer from the content..."}}
        ]
        """

        prompt_tokens = len(self.tokenizer.encode(prompt))
        self.total_tokens += prompt_tokens

        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.choices[0].message.content
            response_tokens = len(self.tokenizer.encode(response_text))
            self.total_tokens += response_tokens

            return json.loads(response_text)

        except Exception as e:
            print(f"❌ Error generating Q&A pairs: {e}")
            return []

    def extract_metadata(self, md_path: str):
        """Extract metadata like document title, author, and date (placeholder for now)."""
        filename = Path(md_path).stem
        return {
            "document_id": filename,
            "title": filename.replace("_", " "),
            "author": "Unknown",
            "published_date": "2025-03-09",
            "metadata": {
                "source": "Research Paper",
                "tokens_per_chunk": self.config.CHUNK_SIZE
            }
        }

    def process_markdown(self):
        """Process Markdown to generate Q&A JSONL and structured chunked JSON."""
        md_files = [f for f in os.listdir(self.markdown_folder) if f.endswith('.md')]
        documents = []

        for md_file in tqdm(md_files, desc="Processing Markdown"):
            md_path = os.path.join(self.markdown_folder, md_file)
            jsonl_path = os.path.join(self.jsonl_folder, f"{Path(md_file).stem}.jsonl")

            try:
                content = self.read_markdown_file(md_path)
                chunks = self.split_into_chunks(content)
                metadata = self.extract_metadata(md_path)

                # Generate Q&A JSONL
                with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
                    for chunk in chunks:
                        qa_pairs = self.generate_qa_pairs(chunk["text"])
                        for qa in qa_pairs:
                            jsonl_entry = {
                                "messages": [
                                    {"role": "user", "content": qa["question"]},
                                    {"role": "assistant", "content": qa["answer"]}
                                ]
                            }
                            jsonl_file.write(json.dumps(jsonl_entry) + '\n')

                # Save structured JSON
                document = {**metadata, "chunks": chunks}
                documents.append(document)

            except Exception as e:
                print(f"❌ Error processing {md_path}: {e}")

        with open(self.json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(documents, json_file, indent=4)

        print(f"✅ JSONL and JSON saved successfully.")
        print(f"📊 Total API Tokens Used: {self.total_tokens}")

if __name__ == "__main__":
    processor = MarkdownProcessor()
    processor.process_markdown()