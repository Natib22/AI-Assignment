 # AI Assignment: Sentiment Analysis with Gradient Boosting

This project demonstrates sentiment analysis using **gradient tree boosting (XGBoost)** on the IMDB movie review dataset. The model predicts whether a movie review is **positive** or **negative** based on the text content.

---

## 📂 Dataset

This project uses the **IMDB Dataset of 50K Movie Reviews** from Stanford:  

[Download the dataset here](https://ai.stanford.edu/~amaas/data/sentiment/)

**Important:** After downloading, place the dataset in the following folder:

Data/raw/IMDB Dataset.csv



## ⚙️ Setup

1. **Create a virtual environment** (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate

	2.	Install dependencies:

pip install -r requirements.txt


⸻

▶️ Running the Project
	1.	Train the model:

python3 src/train.py

	2.	Test the model:

python3 src/test.py

	3.	(Optional) Use the model to check sentiment of new text:

python3 src/checkSentiment.py


⸻

⚠️ Notes
	•	The dataset is not included in the repository due to file size restrictions. Please download it manually and place it in Data/raw/.
	•	You can limit the number of rows processed for testing by modifying MAX_ROWS in config.py.
	•	Hyperparameters such as tree depth, number of trees, and learning rate are set in train.py and can be adjusted to improve performance.

⸻

📊 Results

On the IMDB dataset (full 50,000 reviews):
	•	Accuracy: ~79%
	•	Precision, recall, and F1-score vary slightly between positive and negative reviews
	•	Model uses sequential depth-2 trees with learning rate 0.1 for stable training

