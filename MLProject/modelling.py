import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import os

DATA_PATH = "bbc_news_preprocessing.csv"   
TARGET_COL = "label"
TEXT_COL = "text"


def main():
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True
    )

    with mlflow.start_run() as run:

        df = pd.read_csv(DATA_PATH)

        X = df[TEXT_COL]
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english"
            )),
            ("clf", LogisticRegression(
                max_iter=300,
                n_jobs=-1
            ))
        ])


        pipeline.fit(X_train, y_train)

   
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_test.iloc[:5],
            registered_model_name=None
        )


        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print("Training selesai")
        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()
