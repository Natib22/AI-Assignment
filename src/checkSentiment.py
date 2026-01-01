import joblib
from config import MODEL_PATH

def predict(text):
    model, vectorizer = joblib.load(MODEL_PATH)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "Positive" if pred == 1 else "Negative"

if __name__ == "__main__":
    review = "The movie was boring and too long"
    print("Prediction:", predict(review))