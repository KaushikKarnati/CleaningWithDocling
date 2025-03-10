import os
import json
import random
import re
from pathlib import Path
from tqdm import tqdm
import tiktoken  # Install with `pip install tiktoken`
from together import Together
from docling.document_converter import DocumentConverter

class Config:
    def __init__(self):
        self.CHUNK_SIZE = 512  # Adjusted for RAG optimization
        self.SYSTEM_PROMPTS = [
            'You are an expert aquaculture researcher with extensive knowledge of marine biology, fish farming, and sustainable aquaculture practices.',
            'You are a specialized AI assistant with deep expertise in aquaculture science, focusing on research methodology, water quality management, and aquatic species cultivation.',
            'You are an aquaculture specialist with comprehensive knowledge of both theoretical and practical aspects of aquatic farming systems.'
        ]

class DoclingProcessor:
    def __init__(self, input_folder="research_papers", markdown_folder="extracted_markdown"):
        self.input_folder = input_folder
        self.markdown_folder = markdown_folder
        os.makedirs(self.markdown_folder, exist_ok=True)

        # Initialize Docling Converter
        self.converter = DocumentConverter()

    def process_documents(self):
        """Processes all PDFs in the input folder and converts them to Markdown."""
        for file in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, file)
            if file.endswith(".pdf"):
                self.process_file(file_path)

    def process_file(self, file_path):
        """Converts a document using Docling and saves it as Markdown."""
        print(f"📄 Processing PDF: {file_path}")

        try:
            result = self.converter.convert(file_path)
            doc = result.document
            
            # Convert the document to Markdown
            markdown_content = doc.export_to_markdown()

            # Save as Markdown
            self.save_output(file_path, markdown_content)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    def save_output(self, file_path, markdown_content):
        """Saves extracted text as a Markdown file."""
        filename = Path(file_path).stem
        md_path = os.path.join(self.markdown_folder, f"{filename}.md")

        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_content)

        print(f"✅ Saved Markdown: {md_path}")

class MarkdownToChunkedJSON:
    def __init__(self, json_output_path="chunked_output.json"):
        """Initialize with Together AI API key and setup JSON output file."""
        self.config = Config()
        self.json_output_path = json_output_path
        self.client = Together(api_key="64b179a6566d904ca5d70b70adb3be49f997a01a15f5f7fa9978b330936331c1")

        # Token tracking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Uses OpenAI's tokenizer
        self.total_tokens = 0  # Keeps track of API token usage

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

    def extract_metadata(self, md_path: str):
        """Extract metadata like document title, author, and date (placeholder for now)."""
        filename = Path(md_path).stem
        return {
            "document_id": filename,  # Using filename as a unique document ID
            "title": filename.replace("_", " "),  # Assuming filename as title
            "author": "Unknown",  # Placeholder (can be extracted from document)
            "published_date": "2025-03-09",  # Placeholder
            "metadata": {
                "source": "Research Paper",
                "tokens_per_chunk": self.config.CHUNK_SIZE
            }
        }

    def convert_to_chunked_json(self, markdown_folder):
        """Convert markdown files to a chunked JSON format optimized for RAG."""
        md_files = [f for f in os.listdir(markdown_folder) if f.endswith('.md')]
        documents = []

        for md_file in tqdm(md_files, desc="Converting Markdown to Chunked JSON"):
            md_path = os.path.join(markdown_folder, md_file)

            try:
                content = self.read_markdown_file(md_path)
                chunks = self.split_into_chunks(content)
                metadata = self.extract_metadata(md_path)

                # Construct document JSON structure
                document = {
                    **metadata,
                    "chunks": chunks
                }

                documents.append(document)

            except Exception as e:
                print(f"❌ Error processing {md_path}: {e}")

        # Save all documents to a single JSON file
        with open(self.json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(documents, json_file, indent=4)

        print(f"✅ Saved Chunked JSON output: {self.json_output_path}")
        print(f"📊 Total API Tokens Used: {self.total_tokens}")  # Print total token count

def main():
    # Step 1: Convert PDFs to Markdown
    pdf_processor = DoclingProcessor()
    pdf_processor.process_documents()

    # Step 2: Convert Markdown to Chunked JSON with Token Tracking
    json_converter = MarkdownToChunkedJSON()
    json_converter.convert_to_chunked_json("extracted_markdown")

    print("\n🎉 Processing Complete! Markdown and JSON files are ready.")

if __name__ == "__main__":
    main()