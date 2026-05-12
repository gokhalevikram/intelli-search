# extractor/text_strategy.py
from .base import ExtractionStrategy

class TextFileExtractionStrategy(ExtractionStrategy):
    def extract(self, file_path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
