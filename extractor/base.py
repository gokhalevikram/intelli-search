# extractor/base.py
class ExtractionStrategy:
    def extract(self, file_path):
        raise NotImplementedError
