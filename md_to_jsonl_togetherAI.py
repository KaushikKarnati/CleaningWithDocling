import os
import json
import random
import datetime
import re
from pathlib import Path
from tqdm import tqdm
from together import Together
from docling.document_converter import DocumentConverter

class Config:
    def __init__(self):
        self.CHUNK_SIZE = 1900
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
        """Converts a document using Docling and saves as Markdown."""
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

class MarkdownToJSONL:
    def __init__(self, jsonl_folder="jsonl_output"):
        """Initialize with Together AI API key and setup JSONL output folder."""
        self.config = Config()
        os.makedirs(jsonl_folder, exist_ok=True)

        # Initialize Together AI API
        self.client = Together(api_key= "64b179a6566d904ca5d70b70adb3be49f997a01a15f5f7fa9978b330936331c1")
        self.jsonl_folder = jsonl_folder

    def read_markdown_file(self, md_path: str) -> str:
        """Read content from markdown file."""
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
            
        with open(md_path, 'r', encoding='utf-8') as file:
            return file.read()

    def split_into_sections(self, content: str):
        """Split markdown content into logical sections."""
        sections = []
        current_section = []

        for line in content.split('\n'):
            if line.strip().startswith('#'):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append('\n'.join(current_section))

        # Further split large sections if needed
        final_sections = []
        for section in sections:
            if len(section) > self.config.CHUNK_SIZE:
                paragraphs = re.split(r'\n\s*\n', section)
                current_chunk = []
                current_size = 0
                
                for para in paragraphs:
                    if current_size + len(para) > self.config.CHUNK_SIZE:
                        if current_chunk:
                            final_sections.append('\n\n'.join(current_chunk))
                        current_chunk = [para]
                        current_size = len(para)
                    else:
                        current_chunk.append(para)
                        current_size += len(para)
                
                if current_chunk:
                    final_sections.append('\n\n'.join(current_chunk))
            else:
                final_sections.append(section)

        return final_sections

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

        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"❌ Error generating Q&A pairs: {e}")
            return []

    def create_jsonl_entry(self, question: str, answer: str, system_prompt: str):
        """Create a JSONL entry in the required format."""
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }

    def convert_to_jsonl(self, markdown_folder):
        """Convert markdown files to JSONL format."""
        md_files = [f for f in os.listdir(markdown_folder) if f.endswith('.md')]
        
        for md_file in tqdm(md_files, desc="Converting Markdown to JSONL"):
            md_path = os.path.join(markdown_folder, md_file)
            output_path = os.path.join(self.jsonl_folder, f"{Path(md_file).stem}.jsonl")

            try:
                content = self.read_markdown_file(md_path)
                sections = self.split_into_sections(content)

                with open(output_path, 'w', encoding='utf-8') as f:
                    for section in tqdm(sections, desc=f"Processing {md_file}"):
                        qa_pairs = self.generate_qa_pairs(section)
                        for qa in qa_pairs:
                            system_prompt = random.choice(self.config.SYSTEM_PROMPTS)
                            jsonl_entry = self.create_jsonl_entry(
                                qa['question'],
                                qa['answer'],
                                system_prompt
                            )
                            f.write(json.dumps(jsonl_entry) + '\n')

                print(f"✅ Converted to JSONL: {output_path}")

            except Exception as e:
                print(f"❌ Error processing {md_path}: {e}")

def main():
    # Step 1: Convert PDFs to Markdown
    pdf_processor = DoclingProcessor()
    pdf_processor.process_documents()

    # Step 2: Convert Markdown to JSONL
    jsonl_converter = MarkdownToJSONL()
    jsonl_converter.convert_to_jsonl("extracted_markdown")

    print("\n🎉 Processing Complete! Markdown and JSONL files are ready.")

if __name__ == "__main__":
    main()