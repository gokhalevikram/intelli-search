from pdfminer.high_level import extract_text
from extractor.ocr_strategy import OCRExtractionStrategy

class PDFExtractionStrategy:
    def __init__(self):
        self.ocr = OCRExtractionStrategy()

    def extract(self, file_path):
        # Try normal PDF text extraction
        text = extract_text(file_path)

        # If pdfminer returns empty text → use OCR
        if not text or len(text.strip()) < 20:
            print("Using OCR for scanned PDF:", file_path)
            text = self.ocr.extract(file_path)

        return text
