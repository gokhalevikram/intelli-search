import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import fitz  # PyMuPDF

class OCRExtractionStrategy:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def pdf_to_images_fallback(self, path):
        doc = fitz.open(path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images

    def extract(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        

        # OCR for images
        if ext in [".jpg", ".jpeg", ".png"]:
            try:
                img = Image.open(file_path)
            except:
                raise ValueError("Unsupported image format/type")

            # Convert CMYK → RGB, progressive → normal
            img = img.convert("RGB")
            return pytesseract.image_to_string(img)


        # OCR for scanned PDFs
        try:
            pages = convert_from_path(
            file_path,
            use_pdftocairo=True,
            poppler_path=r"C:\poppler\Library\bin"
        )

        except:
            pages = self.pdf_to_images_fallback(file_path)

        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page)
        return text
