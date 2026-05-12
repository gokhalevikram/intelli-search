# extractor/extractor.py
class TextExtractor:
    def __init__(self):
        self.strategies = {}

    def register_strategy(self, extension, strategy):
        self.strategies[extension] = strategy

    def extract(self, file_path):
        ext = file_path.split(".")[-1].lower()
        strategy = self.strategies.get(ext)
        if strategy:
            return strategy.extract(file_path)
        return ""
