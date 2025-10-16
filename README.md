# Twitter Themenanalyse Region Düsseldorf/NRW

Dieses Projekt analysiert und extrahiert die fünf häufigsten Themen aus Twitter-Nachrichten rund um Düsseldorf und NRW.

## Installation und Setup
1. Umgebung vorbereiten:
    - uv venv .venv
    - .venv\Scripts\activate 
2. Bibliotheken installieren:
uv pip install -r requirements.txt
3. Twitter API Schlüssel in `config.json` hinterlegen (siehe `config.example.json`).

## Verwendung
Das Skript `main.py` ruft Tweets ab, bereinigt sie und führt die Themenanalyse durch. Ergebnisse werden in `data/` abgelegt.

## Struktur
- `main.py` – Hauptskript für die Analyse
- `requirements.txt` – alle nötigen Bibliotheken
- `data/` – Zwischendaten und Exporte
