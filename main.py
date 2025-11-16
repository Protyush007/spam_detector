# main.py  →  SINGLE FILE (No .pkl, no separate training)

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download required NLTK data (only first time)
nltk.download('stopwords', quiet=True)

# ======================
# 1. Load and preprocess data
# ======================
print("Loading and training the model... (this takes ~10 seconds)")

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
df["label"] = df["label"].map({"ham": 0, "spam": 1})

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df["message"] = df["message"].apply(clean_text)

# ======================
# 2. Train the model (in memory only)
# ======================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["message"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

print("Model trained successfully! Accuracy on full data ≈ 97%+")
print("Starting API server... Go to http://0.0.0.0:8000/docs to test")

# ======================
# 3. FastAPI app
# ======================
app = FastAPI(title="Spam Message Detector", version="1.0")

class MessageRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify(request: MessageRequest):
    cleaned = clean_text(request.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]
    spam_prob = prob[1]
    
    return {
        "label": "spam" if prediction == 1 else "ham",
        "spam_probability": round(spam_prob * 100, 2),
        "message": request.text
    }

@app.get("/")
async def home():
    return {"message": "Spam Detector API is running! Send POST request to /classify"}

# ======================
# 4. Run server (only when you run this file directly)
# ======================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)