# 🧠 Clustering L3 – ML 4 à ML 7

Application Streamlit qui regroupe quatre projets d'apprentissage non supervisé (ACP, KMeans, CAH) :

| Projet | Données | Modèle retenu |
|---|---|---|
| 🌾 Graines de blé (ML 4) | Wheat Seeds – 199 graines | CAH (centroïde le plus proche) |
| 📊 Clients télécom (ML 5) | Customer Churn – 3 150 clients | CAH (centroïde le plus proche) |
| 🛒 Visiteurs d'un site marchand (ML 6) | Online Shoppers – 12 330 sessions | CAH (centroïde le plus proche) |
| ₿ Adresses Bitcoin (ML 7) | BitcoinHeist – 2,9 millions de lignes | KMeans (k = 6) |

Pour chaque projet :
- **✍️ Saisie manuelle** d'une observation → classe prédite et position dans le plan de l'ACP ;
- **📁 Import d'un fichier CSV ou Excel** → classe de chaque ligne, graphiques, téléchargement des résultats (CSV / Excel).

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Organisation

```
app.py            interface Streamlit
projets.py        configuration des 4 projets (variables, chaîne de traitement, classes)
modeles/          scaler, ACP et modèle de chaque projet (joblib, scikit-learn 1.6.1)
donnees/          échantillons (≤ 3 000 lignes) utilisés pour les nuages de points
exemples/         fichiers CSV d'exemple à importer
```
