import os
from pathlib import Path
from docling.document_converter import DocumentConverter

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

if __name__ == "__main__":
    processor = DoclingProcessor()
    processor.process_documents()