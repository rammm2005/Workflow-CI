import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack

experiment_name = "CI-BBC-News"
mlflow.set_experiment(experiment_name)

df = pd.read_csv("bbc_news_preprocessing.csv")

X_text = df["clean_text"]
X_num = df[[
    "no_sentences",
    "flesch_reading_ease_score",
    "dale_chall_readability_score"
]]
y = df["label_encoded"]

Xtr_txt, Xte_txt, Xtr_num, Xte_num, ytr, yte = train_test_split(
    X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=3000)
Xtr_tfidf = tfidf.fit_transform(Xtr_txt)
Xte_tfidf = tfidf.transform(Xte_txt)

Xtr = hstack([Xtr_tfidf, Xtr_num])
Xte = hstack([Xte_tfidf, Xte_num])

model = LogisticRegression(max_iter=1000)

with mlflow.start_run():
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)

    acc = accuracy_score(yte, preds)
    f1 = f1_score(yte, preds, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")

print("CI training selesai")
