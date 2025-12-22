import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

# =====================
# Load Dataset
# =====================
df = pd.read_csv("bbc_news_preprocessing.csv")

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
# Pipeline (END-TO-END)
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

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=5000)),
    ]
)

# =====================
# Hyperparameter Tuning
# =====================
param_grid = {
    "classifier__C": [0.1, 1, 10],
    "classifier__solver": ["liblinear", "lbfgs"],
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1,
)

mlflow.set_experiment("BBC News Classification - Tuning")

with mlflow.start_run(run_name="LogReg-Tuned"):
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("cv_best_f1", grid.best_score_)
    mlflow.log_metric(
        "accuracy", accuracy_score(y_test, y_pred)
    )
    mlflow.log_metric(
        "precision",
        precision_score(y_test, y_pred, average="weighted"),
    )
    mlflow.log_metric(
        "recall",
        recall_score(y_test, y_pred, average="weighted"),
    )
    mlflow.log_metric(
        "f1_score",
        f1_score(y_test, y_pred, average="weighted"),
    )

    mlflow.sklearn.log_model(best_model, "model")

    # =====================
    # Artifact 1: Confusion Matrix
    # =====================
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test
    )
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # =====================
    # Artifact 2: TF-IDF Feature List
    # =====================
    tfidf = best_model.named_steps[
        "preprocessing"
    ].named_transformers_["text"]

    with open("tfidf_features.txt", "w") as f:
        for feat in tfidf.get_feature_names_out():
            f.write(feat + "\n")

    mlflow.log_artifact("tfidf_features.txt")

print("✅ modelling.py selesai & siap CI")
