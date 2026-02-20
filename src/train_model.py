import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from preprocessing import clean_text


df = pd.read_csv("../data/raw/it_tickets.csv")

text_col = "Document"
label_col = "Topic_group"

print("Dataset:", df.shape)

df["clean_text"] = df[text_col].apply(clean_text)


print("\nLoading transformer model...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

X = embedder.encode(
    df["clean_text"].tolist(),
    show_progress_bar=True
)

y = df[label_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n=== Semantic Model Performance ===")
print(classification_report(y_test, pred))

joblib.dump(model, "../model/semantic_model.pkl")
joblib.dump(embedder, "../model/embedder.pkl")

print("\n✅ Semantic model saved!")
