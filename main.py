import os
import json
import pandas as pd
import numpy as np
import tweepy
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# --- Einstellungen ---
DATA_PATH = "data/tweets.csv"
CONFIG_PATH = "config.json"

# --- Authentifizierung ---
with open(CONFIG_PATH) as f:
    config = json.load(f)
client = tweepy.Client(bearer_token=config["bearer_token"])

# --- Tweets abrufen ---
def fetch_tweets(query, max_results=100):
    response = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=["author_id", "created_at", "entities", "lang"]
    )
    return [tweet.text for tweet in response.data if tweet.lang == "de"]

tweets = fetch_tweets("#düsseldorf lang:de", 10)
pd.DataFrame({"tweet": tweets}).to_csv(DATA_PATH, index=False)

# --- Vorverarbeitung ---
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
stop_words = set(stopwords.words("german"))
def preprocess(tweet):
    tokens = tokenizer.tokenize(tweet)
    return " ".join([t for t in tokens if t.isalpha() and t not in stop_words])

cleaned_tweets = [preprocess(t) for t in tweets]

# --- Bag of Words ---
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(cleaned_tweets)

# Top-Wörter des Bag-of-Words ausgeben
word_counts = bow.sum(axis=0)
feature_names = vectorizer.get_feature_names_out()
top_words = sorted(zip(word_counts.tolist()[0], feature_names), reverse=True)[:10]
print("Top 10 Wörter aus Bag-of-Words:")
for count, word in top_words:
    print(f"{word}: {count}")

# --- Vektorisierung und Themenmodellierung ---
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(cleaned_tweets)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# --- Top-Begriffe pro Thema extrahieren ---
def print_top_words(model, feature_names, n_top_words=10):
    for ix, topic in enumerate(model.components_):
        top = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        print(f"Thema #{ix+1}: {', '.join(top)}")

print_top_words(lda, tfidf.get_feature_names_out())

# --- Visualisierung (optional) ---
plt.figure(figsize=(8, 5))
plt.bar([f"Thema {i+1}" for i in range(5)], lda.components_.sum(axis=1))
plt.ylabel("Summe der TF-IDF Gewichte")
plt.title("Häufigkeiten der Themen in Düsseldorf Tweets")
plt.tight_layout()
plt.savefig("data/themenverteilung.png")
plt.show()
