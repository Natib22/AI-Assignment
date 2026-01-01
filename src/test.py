import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from loader import load_data
from preprocessing import vectorize_text
from config import MODEL_PATH, TEST_SIZE, RANDOM_STATE

def evaluate():
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model, vectorizer = joblib.load(MODEL_PATH)

    X_test_vec = vectorizer.transform(X_test)
    preds = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    evaluate()