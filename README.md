# Twitter Themenanalyse Region Düsseldorf/NRW

Dieses Projekt analysiert und extrahiert die fünf häufigsten Themen aus Twitter-Nachrichten rund um Düsseldorf und NRW.

## Installation und Setup

1. Umgebung vorbereiten:
    - Installiere uv auf Deinem System [UV Installationsanleitung](https://docs.astral.sh/uv/getting-started/installation/)
    - Erstelle eine neue Virtuel Environment ```uv venv .venv```
    - Aktiviere die neu erstelle Virtuel Environment ```.venv\Scripts\activate```
2. Installiere die benötigten Ressourcen:
```uv pip install -r requirements.txt```
3. Hinterleg deinen Twitter API Schlüssel in `config.json` (siehe `config.example.json`).

## Verwendung
Das Skript `main.py` ruft Tweets ab, bereinigt sie und führt die Themenanalyse durch.  
Ergebnisse werden in `data/` abgelegt.  
Um das Skript auszuführen, muss in der zuvor aktivierten .venv das Skript ausgeführt werden: ```python .\main.py ```

## Struktur
- `main.py` – Hauptskript für die Analyse
- `requirements.txt` – alle nötigen Bibliotheken
- `data/` – Zwischenspeicher für Tweets und optionale Visualisierung
