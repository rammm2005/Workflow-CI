import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# =====================
# CONFIG
# =====================
DATA_PATH = "bbc_news_preprocessing.csv"

TEXT_COL = "clean_text"
NUM_COLS = [
    "no_sentences",
    "flesch_reading_ease_score",
    "dale_chall_readability_score",
]
TARGET_COL = "label_encoded"


def main():
    mlflow.autolog()

    with mlflow.start_run():
        df = pd.read_csv(DATA_PATH)

        X = df[[TEXT_COL] + NUM_COLS]
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("text", TfidfVectorizer(max_features=5000), TEXT_COL),
                ("num", "passthrough", NUM_COLS),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_metric("accuracy_manual", acc)
        mlflow.log_metric("f1_score_manual", f1)

        print(f"Training selesai | ACC={acc:.4f} | F1={f1:.4f}")


if __name__ == "__main__":
    main()
