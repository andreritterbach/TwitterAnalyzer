import os
import json
import csv
import tweepy
import numpy as np
import nltk
import matplotlib
from matplotlib.backends import backend_registry, BackendFilter
nltk.download('stopwords', quiet=True)
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from collections import Counter
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

# Einstellungen
DATA_PATH = "data/tweets.csv" # Pfad zu Tweets die in CSV für weitere Verarbeitung zwischengespeichert werden
JSON_PATH = "data/tweets.json" # Pfad zu Tweets welche als JSON abgespeichert wurden
CONFIG_PATH = "config.json" # Datei in welcher der Bearer Token hinterlegt wird

if __name__ == '__main__': # Muss bei Windows verwendet werden, da ansonsten Code rekursiv ausgeführt wird und zu einer Schleife führt
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
        writer.writerow(['tweet'])
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
            if tweet.get('entities') and tweet['entities'].get('mentions'):
                for mention in tweet['entities']['mentions']:
                    user_mentions.append(mention['username'].lower())
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

    # Tokenisierte Tweets für Gensim (wichtig für Coherence-Berechnung, da diese eine Liste von Token benötigt und keinen String)
    def preprocess_for_gensim(tweet):
        tokens = tokenizer.tokenize(tweet)
        return [t for t in tokens if t.isalpha() and t not in stop_words]

    tokenized_tweets = [preprocess_for_gensim(t) for t in tweets]

    # Bag of Words
    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(cleaned_tweets)
    word_counts = bow.sum(axis=0)
    feature_names = vectorizer.get_feature_names_out()
    top_words = sorted(zip(word_counts.tolist()[0], feature_names), reverse=True)[:10]
    print("\nTop 10 Wörter aus Bag-of-Words:")
    for count, word in top_words:
        print(f"{word}: {count}")

    # TF-IDF Vektorisierung
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(cleaned_tweets)

    # Gensim Dictionary für Coherence-Berechnung
    gensim_dictionary = Dictionary(tokenized_tweets)

    # Funktion zur Berechnung des Coherence Scores
    def compute_coherence_values(corpus, dictionary, k_values, tfidf_matrix):
        """
        Berechnet Coherence Scores für verschiedene Topic-Anzahlen
        
        Parameter:
        ----------
        corpus : list
            Tokenisierte Dokumente
        dictionary : gensim.corpora.Dictionary
            Gensim Dictionary
        k_values : list
            Liste der zu testenden Topic-Anzahlen
        tfidf_matrix : sparse matrix
            TF-IDF Matrix von sklearn
        
        Returns:
        -------
        coherence_values : list
            Coherence Scores für jede Topic-Anzahl
        """
        coherence_values = []
        
        for k in k_values:
            print(f"Berechne Coherence für {k} Topics...")
            # LDA Model trainieren
            lda_model = LatentDirichletAllocation(
                n_components=k, 
                random_state=42,
                max_iter=10,
                learning_method='online'
            )
            lda_model.fit(tfidf_matrix)
            
            # Topics als Listen von Wörtern extrahieren (Top 10 Wörter pro Topic)
            feature_names = tfidf.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda_model.components_):
                top_indices = topic.argsort()[-10:][::-1]
                topic_words = [feature_names[i] for i in top_indices]
                topics.append(topic_words)
            
            # Coherence Score mit Gensim berechnen (c_v Methode)
            coherence_model = CoherenceModel(
                topics=topics,
                texts=corpus,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            coherence_values.append(coherence_score)
            print(f"  Coherence Score für {k} Topics: {coherence_score:.4f}")
        
        return coherence_values

    # Coherence Scores für verschiedene Topic-Anzahlen berechnen
    print("\nCoherence Score Analyse")
    k_values = range(2, 6)  # Teste 2 bis 6 Topics
    coherence_scores = compute_coherence_values(
        tokenized_tweets, 
        gensim_dictionary, 
        k_values,
        tfidf_matrix
    )

    # Optimale Anzahl finden
    optimal_k = k_values[np.argmax(coherence_scores)]
    print(f"\nOptimale Anzahl Topics: {optimal_k} (Coherence: {max(coherence_scores):.4f})")

    # Visualisierung der Coherence Scores (Elbow-Methode)
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, coherence_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal: {optimal_k} Topics')
    plt.xlabel('Anzahl Topics (k)', fontsize=12)
    plt.ylabel('Coherence Score (C_v)', fontsize=12)
    plt.title('Coherence Score vs. Anzahl Topics (Elbow-Methode)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/coherence_elbow.png", dpi=300)
    interactive_backends = backend_registry.list_builtin(BackendFilter.INTERACTIVE) #Überprüft ob der Code in einer Interaktiven Umgebung ausgeführt wird und gibt die Zeichnung aus.
    if matplotlib.get_backend() in interactive_backends:
         plt.show(block=False)
    else:
        print(f"\nDas Bild des Coherence Elbow wurde unter data/coherence_elbow.png gespeichert. Bitte per Bildbetrachter öffnen.")

    # Finales LDA-Modell mit optimaler Topic-Anzahl trainieren
    print(f"\nTrainiere finales LDA-Modell mit {optimal_k} Topics")
    lda_final = LatentDirichletAllocation(
        n_components=optimal_k, 
        random_state=42,
        max_iter=20,
        learning_method='online'
    )
    lda_final.fit(tfidf_matrix)

    feature_names = tfidf.get_feature_names_out()
    # Ausgabe der Top 10 Wörter
    def print_top_words(model, feature_names, n_top_words=10):
        for ix, topic in enumerate(model.components_):
            top = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
            print(f"Thema #{ix+1}: {', '.join(top)}")

    print("\nTop-Begriffe der extrahierten Themen (Optimales LDA-Modell):")
    print_top_words(lda_final, tfidf.get_feature_names_out())

      #  Topic-Labels generieren
    def generate_topic_labels(model, feature_names, n_top_words=3):
        labels = {}
        for ix, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            labels[ix] = " / ".join(top_words)
        return labels
    
    topic_labels = generate_topic_labels(lda_final, feature_names, n_top_words=3)
    
    # Output mit Labels
    print("\nTopics mit automatischen Labels \n")
    for topic_id, label in topic_labels.items():
        print(f"Thema {topic_id + 1}: {label.upper()}")
    
    plt.figure(figsize=(12, 6))
    
    # Labels für x-Achse
    x_labels = [f"Thema {i+1}\n({topic_labels[i]})" for i in range(optimal_k)]
    
    plt.bar(x_labels, lda_final.components_.sum(axis=1), color='steelblue')
    plt.ylabel("Summe der TF-IDF Gewichte", fontsize=12)
    plt.title(f"Häufigkeiten der {optimal_k} Themen in Düsseldorf & NRW Tweets", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("data/themenverteilung_mit_labels.png", dpi=300)
    if matplotlib.get_backend() in interactive_backends:
         plt.show(block=False)
    else:
        print(f"\nDas Bild der Themenverteilung mit den generierten Labels wurde unter data/themenverteilung_mit_labels.png gespeichert. Bitte per Bildbetrachter öffnen.")
    if matplotlib.get_backend() in interactive_backends:  
        plt.show() # wird benötigt damit die mit Matplotlib erstellten Bilder geöffnet bleiben, bis sie geschlossen wurden