import os
import joblib
import numpy as np
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# 1. Define your document classes
label_map = {
    "passport": 0,
    "tax_return": 1,
    "w2": 2,
    "paystub": 3,
    "visa": 4
}

# 2. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

X = []
y = []

# 3. Loop through training folders
for label_name, label_id in label_map.items():
    folder = f"training_data/{label_name}"
    if not os.path.exists(folder):
        print(f"Skipping missing folder: {folder}")
        continue

    for file in os.listdir(folder):
        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(folder, file)
        print("Extracting:", path)

        try:
            text = extract_text(path)
        except Exception as e:
            print("Extraction failed:", e)
            continue

        text = text[:5000]  # limit for stability
        emb = model.encode(text)

        X.append(emb)
        y.append(label_id)

# 4. Convert to arrays
X = np.array(X)
y = np.array(y)

print("Samples loaded:", len(X))
print("Labels loaded:", len(y))
print("Unique labels:", set(y))

# 5. Train classifier
clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
clf.fit(X, y)

print("Classifier coefficients shape:", clf.coef_.shape)

# 6. Save classifier
joblib.dump((clf, label_map), "classifier/document_classifier.pkl")

print("Training complete. Saved to document_classifier.pkl")
