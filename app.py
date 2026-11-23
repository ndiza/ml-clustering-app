"""Application Streamlit qui regroupe les 4 projets de clustering (ML 4 à ML 7).

Lancement :  streamlit run app.py
Pour chaque projet : saisie manuelle d'une observation, ou import d'un fichier CSV / Excel.
"""
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from projets import PROJETS

DOSSIER = Path(__file__).parent
MAX_POINTS_GRAPHE = 5000      # nombre max de points dans les nuages (lisibilité / vitesse)
MAX_LIGNES_EXCEL = 100_000    # au-delà, l'export Excel devient trop lent
COULEURS = px.colors.qualitative.Set2

st.set_page_config(page_title="Clustering – ML 4 à ML 7", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 4rem; max-width: 1200px;}
.badge {display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.8rem;
        font-weight: 600; background: rgba(99, 110, 250, 0.15); color: rgb(99, 110, 250); margin-right: .4rem;}
.resultat {border-radius: 12px; padding: 1.2rem 1.4rem; margin: .5rem 0 1rem 0;
           border-left: 6px solid var(--couleur); background: rgba(128, 128, 128, 0.08);}
.resultat h3 {margin: 0 0 .3rem 0;}
.resultat p {margin: 0; opacity: .85;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- chargement
@st.cache_resource
def charger_modeles(cle):
    scaler, pca, modele = PROJETS[cle]["fichiers"]
    return tuple(joblib.load(DOSSIER / "modeles" / f"{nom}.joblib") for nom in (scaler, pca, modele))


@st.cache_data
def charger_reference(cle):
    fichier = DOSSIER / "donnees" / PROJETS[cle]["reference"]
    return pd.read_csv(fichier) if fichier.exists() else None


@st.cache_data
def lire_exemple(cle):
    return (DOSSIER / "exemples" / PROJETS[cle]["exemple"]).read_bytes()


# ----------------------------------------------------------------------------- traitement
def colonnes(cle):
    return [v["nom"] for v in PROJETS[cle]["variables"]]


def nom_classe(cle, c):
    return PROJETS[cle]["classes"].get(int(c), (f"classe {c}", ""))[0]


def predire(cle, donnees):
    """Même chaîne que dans le notebook. Renvoie (classes, coordonnées PC1/PC2)."""
    scaler, pca, modele = charger_modeles(cle)
    X = donnees[colonnes(cle)].astype(float)
    if PROJETS[cle]["chaine"] == "normalizer":
        # Normalizer -> modèle ; l'ACP ne sert qu'au graphique
        x = scaler.transform(X)
        return modele.predict(x), pca.transform(x)[:, :2]
    # log1p -> StandardScaler -> ACP -> modèle
    x = pd.DataFrame(scaler.transform(np.log1p(X)), columns=colonnes(cle))
    pcs = pca.transform(x)
    return modele.predict(pcs), pcs[:, :2]


def cle_colonne(nom):
    """Nom de colonne simplifié : 'Call  Failure ' et 'call failure' donnent la même clé."""
    return " ".join(str(nom).strip().lower().replace("_", " ").split())


def en_nombres(serie):
    """Convertit une colonne en nombres (accepte la virgule décimale française)."""
    if not pd.api.types.is_numeric_dtype(serie):
        serie = serie.astype(str).str.strip().str.replace(" ", "").str.replace(" ", "").str.replace(",", ".")
    return pd.to_numeric(serie, errors="coerce")


def lire_fichier(fichier):
    nom = fichier.name.lower()
    if nom.endswith((".xlsx", ".xls")):
        return pd.read_excel(fichier)
    # Séparateur deviné sur la ligne d'en-tête : « ; » pour un export Excel français, sinon tabulation ou « , »
    entete = fichier.getvalue().split(b"\n", 1)[0].decode("utf-8", errors="ignore")
    sep = max([";", "\t", ","], key=entete.count)
    return pd.read_csv(fichier, sep=sep)


def preparer_import(cle, brut):
    """Retrouve les colonnes attendues et repère les lignes inutilisables.
    Renvoie (données numériques, masque des lignes valides, colonnes manquantes)."""
    correspondance = {cle_colonne(c): c for c in brut.columns}
    manquantes = [c for c in colonnes(cle) if cle_colonne(c) not in correspondance]
    if manquantes:
        return None, None, manquantes
    donnees = pd.DataFrame({c: en_nombres(brut[correspondance[cle_colonne(c)]]) for c in colonnes(cle)},
                           index=brut.index)
    valides = donnees.notna().all(axis=1)
    if PROJETS[cle]["chaine"] == "log_acp":
        valides &= (donnees >= 0).all(axis=1)   # log1p impossible sur des valeurs négatives
    return donnees, valides, []


def vers_excel(df):
    tampon = BytesIO()
    df.to_excel(tampon, index=False, sheet_name="resultats")
    return tampon.getvalue()


# ----------------------------------------------------------------------------- graphiques
def ordre_classes(cle):
    return [nom for nom, _ in PROJETS[cle]["classes"].values()]


def couleur(cle, c):
    return COULEURS[list(PROJETS[cle]["classes"]).index(int(c)) % len(COULEURS)]


def nuage(cle, pcs, classes, titre, point=None):
    plan = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    plan["Classe"] = [nom_classe(cle, c) for c in classes]
    if len(plan) > MAX_POINTS_GRAPHE:
        plan = plan.sample(MAX_POINTS_GRAPHE, random_state=42)
    # render_mode="svg" : s'affiche même dans un navigateur sans WebGL
    fig = px.scatter(plan, x="PC1", y="PC2", color="Classe", opacity=0.45, title=titre, render_mode="svg",
                     category_orders={"Classe": ordre_classes(cle)}, color_discrete_sequence=COULEURS)
    if point is not None:
        fig.add_scatter(x=[point[0]], y=[point[1]], mode="markers", name=f"Votre {PROJETS[cle]['objet']}",
                        marker=dict(size=20, symbol="star", color="black", line=dict(width=2, color="white")))
    fig.update_layout(legend_title_text="", margin=dict(t=50, l=10, r=10, b=10), height=520,
                      legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0))
    return fig


def barres(cle, classes):
    noms = pd.Series([nom_classe(cle, c) for c in classes])
    compte = noms.value_counts().reindex(ordre_classes(cle)).dropna()
    # Barres horizontales : les noms de classes restent lisibles
    fig = px.bar(y=compte.index, x=compte.values, color=compte.index, text_auto=True, orientation="h",
                 color_discrete_sequence=COULEURS, category_orders={"y": ordre_classes(cle)},
                 labels={"y": "", "x": "Nombre de lignes"}, title="Répartition des classes")
    fig.update_layout(showlegend=False, margin=dict(t=50, l=10, r=10, b=10), height=520)
    return fig


def carte_resultat(cle, c):
    nom, description = PROJETS[cle]["classes"].get(int(c), (f"classe {c}", ""))
    st.markdown(f"""<div class="resultat" style="--couleur:{couleur(cle, c)}">
        <h3>{nom.capitalize()} <span style="opacity:.6;font-size:.9rem">· classe {int(c)}</span></h3>
        <p>{description}</p></div>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- onglets
def onglet_saisie(cle):
    p = PROJETS[cle]
    st.write(f"Renseignez les caractéristiques d'une **{p['objet']}** puis cliquez sur **Prédire**. "
             "Les valeurs proposées par défaut sont les médianes du jeu de données.")
    with st.form(f"form_{cle}"):
        valeurs = {}
        cols = st.columns(3)
        for i, v in enumerate(p["variables"]):
            aide = (v["aide"] + " " if v["aide"] else "") + f"Dans les données : de {v['min']} à {v['max']}."
            with cols[i % 3]:
                if v["entier"]:
                    valeurs[v["nom"]] = st.number_input(v["libelle"], min_value=int(v["min"]), value=int(v["defaut"]),
                                                        step=int(v["pas"]), help=aide, key=f"{cle}_{v['nom']}")
                else:
                    valeurs[v["nom"]] = st.number_input(v["libelle"], min_value=0.0, value=float(v["defaut"]),
                                                        step=float(v["pas"]), format="%.4f", help=aide,
                                                        key=f"{cle}_{v['nom']}")
        envoyer = st.form_submit_button("🔮 Prédire", type="primary", width="stretch")

    if envoyer:
        observation = pd.DataFrame([valeurs])[colonnes(cle)].astype(float)
        for col, facteur in p.get("conversion", {}).items():
            observation[col] *= facteur
        classes, pcs = predire(cle, observation)
        st.session_state[f"saisie_{cle}"] = (int(classes[0]), pcs[0], observation)

    if f"saisie_{cle}" not in st.session_state:
        return
    classe, point, observation = st.session_state[f"saisie_{cle}"]

    st.divider()
    gauche, droite = st.columns([1, 2], gap="large")
    with gauche:
        st.subheader("Résultat")
        carte_resultat(cle, classe)
        if p.get("avertissement"):
            st.caption(p["avertissement"])
        m1, m2 = st.columns(2)
        m1.metric("PC1", f"{point[0]:.3f}")
        m2.metric("PC2", f"{point[1]:.3f}")
        with st.expander("Valeurs envoyées au modèle"):
            st.dataframe(observation.T.rename(columns={0: "valeur"}), width="stretch")
    with droite:
        ref = charger_reference(cle)
        if ref is not None:
            classes_ref, pcs_ref = predire(cle, ref)
            st.plotly_chart(nuage(cle, pcs_ref, classes_ref, f"Votre {p['objet']} dans le plan de l'ACP",
                                  point=point), width="stretch")


def onglet_import(cle):
    p = PROJETS[cle]
    info, exemple = st.columns([3, 1], vertical_alignment="center")
    with info:
        st.write("Importez un fichier **CSV** ou **Excel** : une ligne par observation, "
                 "avec au moins les colonnes suivantes (les autres colonnes sont conservées) :")
        st.markdown(" ".join(f"`{c}`" for c in colonnes(cle)))
        if p.get("note_import"):
            st.info(p["note_import"], icon="ℹ️")
    with exemple:
        st.download_button("⬇️ Fichier exemple", lire_exemple(cle), file_name=p["exemple"],
                           mime="text/csv", width="stretch")

    fichier = st.file_uploader("Déposez votre fichier", type=["csv", "xlsx", "xls"], key=f"upload_{cle}")
    if fichier is None:
        return

    try:
        brut = lire_fichier(fichier)
    except Exception as e:
        st.error(f"Impossible de lire le fichier : {e}")
        return
    if brut.empty:
        st.warning("Le fichier ne contient aucune ligne.")
        return

    donnees, valides, manquantes = preparer_import(cle, brut)
    if manquantes:
        st.error("Colonnes manquantes dans le fichier : " + ", ".join(f"`{c}`" for c in manquantes))
        with st.expander("Colonnes trouvées dans votre fichier"):
            st.write(list(brut.columns))
        return

    n_valides = int(valides.sum())
    if n_valides == 0:
        st.error("Aucune ligne exploitable : vérifiez que les colonnes contiennent des nombres"
                 + (" positifs." if p["chaine"] == "log_acp" else "."))
        return

    with st.spinner("Prédiction en cours…"):
        classes, pcs = predire(cle, donnees[valides])

    resultats = brut.copy()
    resultats["classe"] = pd.Series(pd.NA, index=brut.index, dtype="Int64")
    resultats.loc[valides, "classe"] = classes
    resultats["profil"] = resultats["classe"].map(lambda c: "ligne invalide" if pd.isna(c) else nom_classe(cle, c))

    m1, m2, m3 = st.columns(3)
    m1.metric("Lignes importées", f"{len(brut):,}".replace(",", " "))
    m2.metric("Lignes classées", f"{n_valides:,}".replace(",", " "))
    m3.metric("Lignes invalides", f"{len(brut) - n_valides:,}".replace(",", " "))
    if n_valides < len(brut):
        st.warning("Certaines lignes contiennent des valeurs manquantes, non numériques"
                   + (" ou négatives" if p["chaine"] == "log_acp" else "")
                   + " : elles sont marquées « ligne invalide ».")

    g1, g2 = st.columns(2, gap="large")
    g1.plotly_chart(barres(cle, classes), width="stretch")
    g2.plotly_chart(nuage(cle, pcs, classes, "Les lignes dans le plan de l'ACP"), width="stretch")

    st.subheader("Résultats")
    profils = st.multiselect("Filtrer par profil", ordre_classes(cle) + ["ligne invalide"], key=f"filtre_{cle}",
                             placeholder="Tous les profils")
    affichage = resultats[resultats["profil"].isin(profils)] if profils else resultats
    # Colonnes mêlant texte et nombres : affichées comme texte (sinon le tableau ne s'affiche pas)
    texte = affichage.select_dtypes(exclude=["number", "bool", "datetime"]).columns
    affichage = affichage.astype({c: str for c in texte})
    st.dataframe(affichage, width="stretch", hide_index=True,
                 column_config={"profil": st.column_config.TextColumn("profil", pinned=True)})

    base = Path(fichier.name).stem + "_resultats"
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇️ Télécharger en CSV", resultats.to_csv(index=False).encode("utf-8"),
                       file_name=f"{base}.csv", mime="text/csv", width="stretch")
    if len(resultats) <= MAX_LIGNES_EXCEL:
        d2.download_button("⬇️ Télécharger en Excel", vers_excel(resultats), file_name=f"{base}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width="stretch")
    else:
        d2.caption(f"Export Excel disponible jusqu'à {MAX_LIGNES_EXCEL:,} lignes : utilisez le CSV.".replace(",", " "))


# ----------------------------------------------------------------------------- pages
def aller_a(cle):
    """Callback : change de page avant le prochain affichage du menu."""
    st.session_state["page"] = cle


def page_accueil():
    st.title("🧠 Apprentissage non supervisé – ML 4 à ML 7")
    st.write("Quatre projets de **clustering** réalisés en Licence 3. Pour chacun : nettoyage des données, "
             "ACP, KMeans et CAH, puis choix du meilleur modèle selon l'indice de silhouette. "
             "Choisissez un projet dans le menu à gauche.")
    st.write("")
    cartes = st.columns(2, gap="medium")
    for i, (cle, p) in enumerate(PROJETS.items()):
        with cartes[i % 2].container(border=True):
            st.markdown(f"<span class='badge'>{p['cours']}</span>", unsafe_allow_html=True)
            st.subheader(f"{p['icone']} {p['titre']}")
            st.write(p["resume"])
            st.caption(f"📁 {p['donnees']}  \n🤖 Modèle retenu : {p['modele']}  \n"
                       f"🏷️ Classes : {', '.join(ordre_classes(cle))}")
            st.button("Ouvrir le projet →", key=f"ouvrir_{cle}", width="stretch",
                      on_click=aller_a, args=(cle,))
    st.write("")
    st.info("Chaque projet propose deux modes : **✍️ Saisie manuelle** d'une observation, "
            "ou **📁 Import** d'un fichier CSV / Excel pour classer beaucoup de lignes d'un coup, "
            "avec téléchargement des résultats.", icon="💡")


def page_projet(cle):
    p = PROJETS[cle]
    st.markdown(f"<span class='badge'>{p['cours']}</span><span class='badge'>{p['modele']}</span>",
                unsafe_allow_html=True)
    st.title(f"{p['icone']} {p['titre']}")
    st.write(p["resume"])
    try:
        charger_modeles(cle)
    except FileNotFoundError as e:
        st.error(f"Fichier modèle introuvable : `{Path(e.filename).name}`. "
                 "Copiez les fichiers .joblib du notebook dans le dossier `modeles/`.")
        return
    saisie, imp, classes = st.tabs(["✍️ Saisie manuelle", "📁 Import CSV / Excel", "🏷️ Les classes"])
    with saisie:
        onglet_saisie(cle)
    with imp:
        onglet_import(cle)
    with classes:
        for c in p["classes"]:
            carte_resultat(cle, c)


# ----------------------------------------------------------------------------- navigation
OPTIONS = ["accueil"] + list(PROJETS)
if "page" not in st.session_state:
    st.session_state["page"] = "accueil"

with st.sidebar:
    st.markdown("## 🧠 Clustering L3")
    st.radio("Projet", OPTIONS, key="page", label_visibility="collapsed",
             format_func=lambda k: "🏠 Accueil" if k == "accueil"
             else f"{PROJETS[k]['icone']} {PROJETS[k]['titre']} ({PROJETS[k]['cours']})")
    st.divider()
    st.caption("Chaîne de traitement identique aux notebooks : mêmes variables, même normalisation, "
               "même ACP et même modèle sauvegardé (joblib).")

if st.session_state["page"] == "accueil":
    page_accueil()
else:
    page_projet(st.session_state["page"])
