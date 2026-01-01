from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from loader import load_data
from preprocessing import vectorize_text
from config import (
    TEST_SIZE, RANDOM_STATE,
    N_ESTIMATORS, LEARNING_RATE, MAX_DEPTH, MODEL_PATH
)

def train():
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    model = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH
    )

    model.fit(X_train_vec, y_train)

    joblib.dump((model, vectorizer), MODEL_PATH)
    print("Model trained and saved.")

if __name__ == "__main__":
    train()