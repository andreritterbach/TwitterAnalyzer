import os
import json
import csv
import tweepy
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from collections import Counter

# Einstellungen
DATA_PATH = "data/tweets.csv"
JSON_PATH = "data/tweets.json"
CONFIG_PATH = "config.json"

# Authentifizierung
with open(CONFIG_PATH) as f:
    config = json.load(f)
client = tweepy.Client(bearer_token=config["bearer_token"])

# Tweets abrufen (force_refresh=True) oder aus JSON laden (force_refresh=False)
def fetch_or_load_tweets(query, max_results=100, force_refresh=False):
    if os.path.exists(JSON_PATH) and not force_refresh:
        print("Lade Tweets aus JSON-Datei...")
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            tweets_data = json.load(f)
        return tweets_data
    else:
        print("Rufe neue Tweets von der API ab...")
        response = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["author_id", "created_at", "entities", "public_metrics", "lang"]
        )
        
        # Response in JSON-Format konvertieren
        tweets_data = []
        for tweet in response.data:
            tweet_dict = {
                'id': tweet.id,
                'text': tweet.text,
                'author_id': tweet.author_id,
                'created_at': str(tweet.created_at),
                'lang': tweet.lang,
                'entities': tweet.entities if hasattr(tweet, 'entities') else None,
                'public_metrics': tweet.public_metrics if hasattr(tweet, 'public_metrics') else None
            }
            tweets_data.append(tweet_dict)
        
        # In JSON-Datei speichern
        os.makedirs('data', exist_ok=True)
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(tweets_data, f, ensure_ascii=False, indent=2)
        print(f"Tweets in {JSON_PATH} gespeichert.")
        
        return tweets_data

query = "#düsseldorf OR #duesseldorf OR #NRW lang:de"
tweets_data = fetch_or_load_tweets(query, 100)

# Tweets extrahieren
tweets = [tweet['text'] for tweet in tweets_data if tweet['lang'] == "de"]
with open(DATA_PATH, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['tweet'])  # Kopfzeile
    for tweet in tweets:
        writer.writerow([tweet])
# Häufigste Hashtags
def extract_hashtags(tweets_data):
    hashtags = []
    for tweet in tweets_data:
        if tweet.get('entities') and tweet['entities'].get('hashtags'):
            for tag in tweet['entities']['hashtags']:
                hashtags.append(tag['tag'].lower())
    return hashtags

# Aktivste User (Mentions und Retweets)
def extract_active_users(tweets_data):
    user_mentions = []
    user_retweets = Counter()
    for tweet in tweets_data:
        # user_mentions aus Entities
        if tweet.get('entities') and tweet['entities'].get('mentions'):
            for mention in tweet['entities']['mentions']:
                user_mentions.append(mention['username'].lower())
        # Retweet-Zählung
        if tweet.get('public_metrics') and tweet['public_metrics'].get('retweet_count'):
            user_retweets[str(tweet['author_id'])] += tweet['public_metrics']['retweet_count']

    return Counter(user_mentions), user_retweets

hashtags = extract_hashtags(tweets_data)
user_mentions_counter, user_retweets_counter = extract_active_users(tweets_data)

print("\nTop 5 Hashtags:")
for tag, count in Counter(hashtags).most_common(5):
    print(f"#{tag}: {count}")

print("\nTop 5 Nutzer-Mentions:")
for user, count in user_mentions_counter.most_common(5):
    print(f"@{user}: {count}")

print("\nTop 5 User nach Retweet-Zahlen:")
for user_id, rt_count in user_retweets_counter.most_common(5):
    print(f"User {user_id}: {rt_count} Retweets")

# Vorverarbeitung
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
stop_words = set(stopwords.words("german"))
def preprocess(tweet):
    tokens = tokenizer.tokenize(tweet)
    return " ".join([t for t in tokens if t.isalpha() and t not in stop_words])

cleaned_tweets = [preprocess(t) for t in tweets]

# Bag of Words
vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(cleaned_tweets)
word_counts = bow.sum(axis=0)
feature_names = vectorizer.get_feature_names_out()
top_words = sorted(zip(word_counts.tolist()[0], feature_names), reverse=True)[:10]
print("\nTop 10 Wörter aus Bag-of-Words:")
for count, word in top_words:
    print(f"{word}: {count}")

# TF-IDF und LDA
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(cleaned_tweets)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

def print_top_words(model, feature_names, n_top_words=10):
    for ix, topic in enumerate(model.components_):
        top = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        print(f"Thema #{ix+1}: {', '.join(top)}")

print("\nTop-Begriffe der extrahierten Themen (LDA):")
print_top_words(lda, tfidf.get_feature_names_out())

# Visualisierung
plt.figure(figsize=(8, 5))
plt.bar([f"Thema {i+1}" for i in range(5)], lda.components_.sum(axis=1))
plt.ylabel("Summe der TF-IDF Gewichte")
plt.title("Häufigkeiten der Themen in Düsseldorf & NRW Tweets")
plt.tight_layout()
plt.savefig("data/themenverteilung.png")
plt.show()