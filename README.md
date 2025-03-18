# GraphAST-ML

GraphAST-ML ist ein **Open-Source-Projekt**, das Python-Quellcode in **Abstrakte Syntaxbäume (ASTs)** umwandelt, diese in **Graph-Neuronale Netze (GNNs) mit PyTorch Geometric** transformiert und damit die Grundlage für **Code-Analysen und Optimierungen** bietet.

Das Projekt eignet sich für:
✅ **Code-Bewertung** (z. B. Code-Qualität analysieren)  
✅ **Erkennung von Code-Smells** (z. B. ungenutzte Variablen oder ineffiziente Strukturen)  
✅ **Automatisiertes Refactoring** (z. B. Vorschläge zur Code-Verbesserung generieren)  

---

## 🚀 Features
- **AST-Parsing**: Wandelt Python-Code in einen **Abstrakten Syntaxbaum (AST)** um.
- **Graph-Transformation**: Konvertiert ASTs in **Graph-Strukturen**, die von GNNs verarbeitet werden können.
- **Neuronales Netzwerk**: Verarbeitet die Graphen mit **PyTorch Geometric**.
- **Automatisierte Tests**: Sicherstellung der Funktionsweise mit **pytest**.
- **Erweiterbar**: Perfekte Basis für Projekte zur statischen Code-Analyse oder KI-gestütztes Refactoring.

---

## 📌 Installation
### 1️⃣ Repository klonen
```bash
git clone https://github.com/deinusername/GraphAST-ML.git
cd GraphAST-ML
```

### 2️⃣ Virtuelle Umgebung erstellen und aktivieren
```bash
python3.12 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate   # Windows (PowerShell)
```

### 3️⃣ Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

---

## 🔍 Nutzung
### 1️⃣ **ASTs aus Quellcode generieren**
```bash
python scripts/ast_parser.py
```
🔹 **Liest Python-Dateien aus `data/raw/` und erzeugt ASTs**

### 2️⃣ **ASTs in Graphen umwandeln**
```bash
python scripts/data_pipeline.py
```
🔹 **Konvertiert ASTs in Graph-Datenstrukturen** (PyTorch Geometric-kompatibel)

### 3️⃣ **Modell trainieren**
```bash
python scripts/model_training.py
```
🔹 **Trainiert ein einfaches GNN-Modell auf den generierten Graph-Daten**

### 4️⃣ **Tests ausführen**
```bash
pytest tests/
```
🔹 **Stellt sicher, dass die Code-Pipeline einwandfrei funktioniert**

---

## 📊 Architektur
Die Applikation folgt einer modularen Struktur:
```
GraphAST-ML/
├── data/                # Rohdaten und verarbeitete Daten
│   ├── raw/             # Ursprünglicher Python-Code
│   ├── processed/       # Konvertierte Graph-Daten
│
├── scripts/             # Kernmodule
│   ├── ast_parser.py    # Parsen von Python-Code zu ASTs
│   ├── data_pipeline.py # Konvertierung von ASTs in Graph-Daten
│   ├── model_training.py# Training eines Graph-Neuronalen Netzes
│
├── tests/               # Testskripte für Unit-Tests
│   ├── test_ast_parser.py
│   ├── test_data_pipeline.py
│
├── requirements.txt     # Benötigte Python-Bibliotheken
├── README.md            # Projektbeschreibung
```

---

## ⚙️ Technologien & Bibliotheken
- **Python 3.12**: Hauptprogrammiersprache
- **PyTorch Geometric**: Verarbeitung von Graph-Daten
- **NetworkX**: Erstellung und Bearbeitung von Graphen
- **pytest**: Test-Framework

---

## 📌 Erweiterungen & Zukunftsideen
- **Feinere AST-Features**: Mehr Kontextinformationen aus Code extrahieren.
- **Fortgeschrittene ML-Modelle**: GAT, TreeLSTM oder Transformer für bessere Vorhersagen.
- **Refactoring-Vorschläge**: Automatische Code-Verbesserung anhand AST-Muster.
- **Web-Interface**: GUI zur Visualisierung und Analyse von Code.

---

## 🤝 Mitwirken
Du möchtest beitragen? Gerne!
1. Forke das Repository 🚀
2. Erstelle einen neuen Branch (`feature/deinFeature`)
3. Commits mit klaren Messages (`git commit -m "Neue Funktion: AST-Visualisierung"`)
4. Pull Request stellen 🔥

---

## 📜 Lizenz
**MIT License** – Frei für kommerzielle & private Nutzung. Fork it, build it, improve it! 😎

---

## 🌟 Starte jetzt mit GraphAST-ML! 🚀

