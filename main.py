import sys

from crawler.file_crawler import FileCrawler
from extractor.extractor import TextExtractor
from extractor.pdf_strategy import PDFExtractionStrategy
from extractor.text_strategy import TextFileExtractionStrategy
from embedding.embedding_engine import EmbeddingEngine
from index.document_index import DocumentIndex
from search.search_engine import SearchEngine

import numpy as np
import os
import json
import hashlib

MAX_SIZE_MB = 5
ALLOWED_EXT = {".pdf", ".txt"}
CACHE_DIR = "cache"

os.makedirs(CACHE_DIR, exist_ok=True)

def file_hash(path):
    """Hash file contents to detect changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def load_cache(path_hash):
    cache_path = os.path.join(CACHE_DIR, f"{path_hash}.json")
    if not os.path.exists(cache_path):
        return None

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # convert embedding list → numpy array
    data["embedding"] = np.array(data["embedding"])
    return data


def save_cache(path_hash, text, embedding):
    cache_path = os.path.join(CACHE_DIR, f"{path_hash}.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({
            "text": text,
            "embedding": embedding.tolist()  # <-- convert ndarray → list
        }, f)


def build_index(root_paths):
    crawler = FileCrawler(root_paths)

    extractor = TextExtractor()
    extractor.register_strategy("pdf", PDFExtractionStrategy())
    extractor.register_strategy("txt", TextFileExtractionStrategy())

    embedder = EmbeddingEngine()
    index = DocumentIndex()

    print("Indexing files...")

    for file in crawler.crawl():

        # 1. File type filter
        ext = os.path.splitext(file)[1].lower()
        if ext not in ALLOWED_EXT:
            continue

        # 2. Large file filter
        size_mb = os.path.getsize(file) / (1024 * 1024)
        if size_mb > MAX_SIZE_MB:
            print(f"Skipping large file ({size_mb:.1f} MB):", file)
            continue

        # 3. Cache check
        h = file_hash(file)
        cached = load_cache(h)

        if cached:
            print("Using cached:", file)
            index.add(file, cached["text"], cached["embedding"])
            continue

        # 4. Extract text
        text = extractor.extract(file).strip()
        if not text:
            print("Skipping empty/scanned PDF:", file)
            continue

        # 5. Embed
        embedding = embedder.embed(text)

        # 6. Add to index
        index.add(file, text, embedding)

        # 7. Save to cache
        save_cache(h, text, embedding)

    return index, embedder


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"search query\"")
        sys.exit(1)

    query = sys.argv[1]

    # Adjust this path to whatever folder you want to index
    root_paths = [r"C:\Users\gokha\OneDrive\Documents"]


    index, embedder = build_index(root_paths)

    searcher = SearchEngine(index, embedder)

    print(f"\nSearching for: {query}\n")
    results = searcher.search(query)

    for score, file in results:
        print(f"{score:.3f}  {file}")


if __name__ == "__main__":
    main()
