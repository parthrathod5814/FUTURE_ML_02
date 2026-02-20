import joblib
from preprocessing import clean_text

print("Loading models...")

tfidf_model = joblib.load("../model/category_model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

semantic_model = joblib.load("../model/semantic_model.pkl")
embedder = joblib.load("../model/embedder.pkl")


def assign_priority(category):

    high = ["Hardware", "HR Support"]
    medium = ["Access", "Administrative rights"]

    if category in high:
        return "High"
    elif category in medium:
        return "Medium"
    else:
        return "Low"


def predict_ticket(text):

    cleaned = clean_text(text)

    vec_tfidf = vectorizer.transform([cleaned])
    pred_tfidf = tfidf_model.predict(vec_tfidf)[0]

    vec_semantic = embedder.encode([cleaned])
    pred_semantic = semantic_model.predict(vec_semantic)[0]

    print("\nTicket:", text)
    print("TF-IDF prediction:", pred_tfidf)
    print("Semantic prediction:", pred_semantic)

    if pred_tfidf == pred_semantic:
        print("✅ Models agree")
    else:
        print("⚠ Models disagree — review suggested")

    priority = assign_priority(pred_semantic)

    print("Priority:", priority)


while True:

    text = input("\nEnter ticket (exit to quit): ")

    if text.lower() == "exit":
        break

    predict_ticket(text)
