import os
from pathlib import Path
from docling.document_converter import DocumentConverter

class DoclingProcessor:
    def __init__(self, input_folder, output_folder="processed_documents"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize Docling Converter
        self.converter = DocumentConverter()

    def process_documents(self):
        """Processes all documents in the input folder."""
        for file in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, file)
            if file.endswith((".pdf", ".jpg", ".png")):
                self.process_file(file_path)

    def process_file(self, file_path):
        """Converts a document using Docling and saves as Markdown."""
        print(f"Processing: {file_path}")

        try:
            # Convert document to Docling format
            result = self.converter.convert(file_path)
            doc = result.document
            
            # Convert the document to Markdown
            markdown_content = doc.export_to_markdown()

            # Save as Markdown
            self.save_output(file_path, markdown_content)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def save_output(self, file_path, markdown_content):
        """Saves converted text as a Markdown file."""
        filename = Path(file_path).stem
        md_path = os.path.join(self.output_folder, f"{filename}.md")

        # Save Markdown file
        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_content)

        print(f"Saved: {md_path}")

# Run the processor
if __name__ == "__main__":
    input_folder = "research_papers"  # Make sure to put your PDFs/images here
    processor = DoclingProcessor(input_folder)
    processor.process_documents()