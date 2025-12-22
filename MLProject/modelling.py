import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack

df = pd.read_csv("bbc_news_preprocessing.csv")

X_text = df["clean_text"]
X_num = df[["no_sentences", "flesch_reading_ease_score", "dale_chall_readability_score"]]
y = df["label_encoded"]

Xtr_txt, Xte_txt, Xtr_num, Xte_num, ytr, yte = train_test_split(
    X_text, X_num, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=3000)
Xtr_tfidf = tfidf.fit_transform(Xtr_txt)
Xte_tfidf = tfidf.transform(Xte_txt)

scaler = StandardScaler()
Xtr_num_scaled = scaler.fit_transform(Xtr_num)
Xte_num_scaled = scaler.transform(Xte_num)

Xtr = hstack([Xtr_tfidf, Xtr_num_scaled])
Xte = hstack([Xte_tfidf, Xte_num_scaled])

model = LogisticRegression(max_iter=5000)

with mlflow.start_run() as run:
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)

    acc = accuracy_score(yte, preds)
    f1 = f1_score(yte, preds, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")

    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)

print("CI training selesai. run_id:", run_id)
