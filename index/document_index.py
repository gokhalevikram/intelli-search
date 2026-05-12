# index/document_index.py
class DocumentIndex:
    def __init__(self):
        self.index = []

    def add(self, file_path, text, embedding):
        self.index.append({
            "file": file_path,
            "text": text,
            "embedding": embedding
        })
