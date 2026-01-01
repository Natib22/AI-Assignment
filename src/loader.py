import pandas as pd
from config import DATA_PATH, MAX_ROWS

def load_data():
    df = pd.read_csv(DATA_PATH)

    if MAX_ROWS is not None:
        df = df.sample(n=MAX_ROWS, random_state=42)

    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df