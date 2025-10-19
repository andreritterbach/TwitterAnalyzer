# Twitter Themenanalyse Region Düsseldorf/NRW

Dieses Projekt analysiert und extrahiert die fünf häufigsten Themen aus Twitter-Nachrichten rund um Düsseldorf und NRW.  
Das Skript `main.py` ruft Tweets ab, bereinigt sie und führt die Themenanalyse durch.  
Ergebnisse werden in `data/` abgelegt.  

## Installation und Setup

1. Umgebung vorbereiten:
    - Installiere uv auf Deinem System [UV Installationsanleitung](https://docs.astral.sh/uv/getting-started/installation/)
    - Erstelle eine neue Virtuelle Umgebung ```uv venv .venv```
    - Aktiviere die neu erstellte Virtuelle Umgebung ```.venv\Scripts\activate```
2. Installiere die benötigten Ressourcen:
```uv pip install -r requirements.txt```
3. Hinterleg deinen Twitter API Schlüssel in `config.json` (siehe `config.example.json`).

## Verwendung

Lege zunächst fest, ob neue Tweets abgerufen werden sollen oder bereits vorhandene genutzt werden sollen, falls du den Code anpassen möchstest.  
Ändere dazu den Wert ``force_refresh`` auf `True` wenn neue Tweets abgerufen werden sollen.  
Wenn du bereits Heruntergeladene Tweets verweden möchtest, ändere den Wert ``force_refresh`` auf `False`.  
Um das Skript auszuführen, muss in der zuvor aktivierten .venv das Skript ausgeführt werden: ```python .\main.py ```  

## Struktur

- `main.py` – Hauptskript für die Analyse
- `requirements.txt` – Hier sind alle benötigten Ressourcen für das Hauptskript hinterlegt
- `data/` – Zwischenspeicher für Tweets und optionale Visualisierung
- `config.example.json` - Enthält die Beispiel Konfiguration für den Twitter/X API-Token
