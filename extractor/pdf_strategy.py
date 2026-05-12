# extractor/pdf_strategy.py
from pdfminer.high_level import extract_text
from .base import ExtractionStrategy

class PDFExtractionStrategy(ExtractionStrategy):
    def extract(self, file_path):
        return extract_text(file_path)
