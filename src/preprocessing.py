from sklearn.feature_extraction.text import CountVectorizer
from config import MAX_FEATURES

def vectorize_text(train_texts, test_texts):
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=MAX_FEATURES
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer