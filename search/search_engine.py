# search/search_engine.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

class SearchEngine:
    def __init__(self, index, embedder, classifier=None):
        self.index = index
        self.embedder = embedder
        self.classifier, self.label_map = joblib.load("classifier/document_classifier.pkl")

    def compute_score(self, query_embedding, doc, keyword):
        filename = doc["file"]
        text = doc["text"]
        embedding = doc["embedding"]

        # semantic similarity
        semantic_score = float(
            np.dot(embedding, query_embedding) /
            (np.linalg.norm(embedding) * np.linalg.norm(query_embedding))
        )

        # filename score
        filename_score = 1.0 if keyword.lower() in filename.lower() else 0.0

        # keyword frequency
        freq = text.lower().count(keyword.lower())
        keyword_freq_score = min(freq * 0.1, 1.0)

        # classifier score
        classifier_score = 0.0
        if self.classifier:
            proba = self.classifier.predict_proba([embedding])[0]
            classifier_score = proba[self.label_map[keyword]]

        # weighted sum
        final_score = (
            0.50 * semantic_score +
            0.30 * filename_score +
            0.10 * keyword_freq_score +
            0.40 * classifier_score
        )

        return final_score


    def search(self, query, top_k=5):
        query_emb = self.embedder.embed(query)
        results = []
        keyword = query.lower()
        for doc in self.index.index:
            score = self.compute_score(query_emb, doc, keyword)

            results.append((score, doc["file"]))

        results.sort(reverse=True)
        return results[:top_k]
