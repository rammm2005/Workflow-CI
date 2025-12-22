import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score


# =====================
# Load Dataset
# =====================
DATA_PATH = "bbc_news_preprocessing.csv"
df = pd.read_csv(DATA_PATH)

X = df[
    [
        "clean_text",
        "no_sentences",
        "flesch_reading_ease_score",
        "dale_chall_readability_score",
    ]
]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# =====================
# Preprocessing + Model
# =====================
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=3000), "clean_text"),
        (
            "num",
            "passthrough",
            [
                "no_sentences",
                "flesch_reading_ease_score",
                "dale_chall_readability_score",
            ],
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=3000)),
    ]
)

# =====================
# MLflow Tracking
# =====================
mlflow.set_experiment("BBC News CI Training")

with mlflow.start_run(run_name="ci-retraining"):
    # WAJIB untuk Kriteria Basic
    mlflow.sklearn.autolog()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy_test", acc)
    mlflow.log_metric("f1_weighted_test", f1)

print("✅ CI retraining selesai dengan MLflow Project")
