"""Configuration des 4 projets de clustering (ML 4 à ML 7).

Chaque projet décrit :
- les fichiers du modèle (scaler, ACP, modèle) dans le dossier `modeles/` ;
- la chaîne de traitement utilisée dans le notebook :
    * "normalizer" : Normalizer -> modèle            (ACP seulement pour le graphique)
    * "log_acp"    : log1p -> StandardScaler -> ACP -> modèle
- les variables d'entrée, dans le MÊME ORDRE qu'à l'entraînement ;
- le nom et la description de chaque classe (les mêmes que dans le notebook).
"""


def var(nom, libelle, defaut, vmin, vmax, entier=False, pas=None, aide=""):
    """Décrit une variable du formulaire de saisie."""
    if pas is None:
        pas = 1 if entier else (0.001 if vmax <= 1 else 0.01)
    return dict(nom=nom, libelle=libelle, defaut=defaut, min=vmin, max=vmax,
                entier=entier, pas=pas, aide=aide)


PROJETS = {
    # ------------------------------------------------------------------ ML 4
    "ble": dict(
        titre="Graines de blé",
        icone="🌾",
        cours="ML 4",
        objet="graine",
        resume="Classer une graine de blé à partir de 7 mesures de sa forme.",
        donnees="Wheat Seeds – 199 graines, 7 mesures",
        modele="CAH (affectation par centroïde le plus proche)",
        chaine="normalizer",
        fichiers=("scaler_wheat", "pca_encoder_wheat", "model_wheat"),
        reference="reference_ble.csv",
        exemple="exemple_ble.csv",
        variables=[
            var("area A", "Surface (area A)", 14.43, 10.59, 21.18),
            var("perimeter", "Périmètre", 14.37, 12.41, 17.25),
            var("compactness", "Compacité", 0.8734, 0.8081, 0.9183, pas=0.001),
            var("length of kernel", "Longueur du grain", 5.541, 4.899, 6.675),
            var("width of kernel", "Largeur du grain", 3.245, 2.63, 4.033),
            var("asymmetry coefficient", "Coefficient d'asymétrie", 3.631, 0.7651, 8.315),
            var("length of kernel groove", "Longueur du sillon", 5.228, 4.519, 6.55),
        ],
        classes={
            0: ("petite graine", "Graine petite et étroite (surface ≈ 12, largeur ≈ 2,8), "
                                 "avec un coefficient d'asymétrie élevé (≈ 4,9)."),
            1: ("grosse graine", "Graine grande et large (surface ≈ 16, largeur ≈ 3,5), "
                                 "plus compacte et plus symétrique (asymétrie ≈ 2,9)."),
        },
    ),
    # ------------------------------------------------------------------ ML 5
    "clients": dict(
        titre="Clients télécom",
        icone="📊",
        cours="ML 5",
        objet="client",
        resume="Repérer si un client d'opérateur télécom est fidèle ou non.",
        donnees="Customer Churn – 3 150 clients, 12 variables",
        modele="CAH (affectation par centroïde le plus proche)",
        chaine="normalizer",
        fichiers=("scaler_costumer", "pca_encoder_customer", "model_customer"),
        reference="reference_clients.csv",
        exemple="exemple_clients.csv",
        variables=[
            var("Call  Failure", "Appels échoués", 6, 0, 36, entier=True),
            var("Complains", "Réclamation (0 = non, 1 = oui)", 0, 0, 1, entier=True),
            var("Subscription  Length", "Ancienneté de l'abonnement (mois)", 35, 3, 47, entier=True),
            var("Charge  Amount", "Niveau de facturation (0 à 10)", 0, 0, 10, entier=True),
            var("Seconds of Use", "Durée d'appel (secondes)", 2990, 0, 17090, entier=True),
            var("Frequency of use", "Nombre d'appels", 54, 0, 255, entier=True),
            var("Frequency of SMS", "Nombre de SMS", 21, 0, 522, entier=True),
            var("Distinct Called Numbers", "Numéros différents appelés", 21, 0, 97, entier=True),
            var("Age Group", "Tranche d'âge (1 à 5)", 3, 1, 5, entier=True),
            var("Status", "Statut (1 = actif, 2 = inactif)", 1, 1, 2, entier=True),
            var("Age", "Âge", 30, 15, 55, entier=True),
            var("Customer Value", "Valeur du client", 228.48, 0.0, 2165.28, pas=1.0),
        ],
        classes={
            0: ("non fidèle", "Client presque inactif : aucun appel, aucun SMS, "
                              "valeur nulle, souvent au statut inactif."),
            1: ("fidèle", "Client qui utilise régulièrement son abonnement "
                          "(≈ 57 appels, ≈ 23 SMS, valeur ≈ 243)."),
        },
    ),
    # ------------------------------------------------------------------ ML 6
    "visiteurs": dict(
        titre="Visiteurs d'un site marchand",
        icone="🛒",
        cours="ML 6",
        objet="session",
        resume="Identifier le type de visiteur à partir d'une session de navigation.",
        donnees="Online Shoppers – 12 330 sessions, 10 variables",
        modele="CAH (affectation par centroïde le plus proche)",
        chaine="log_acp",
        fichiers=("scaler_shoppers", "pca_encoder_shoppers", "model_shoppers"),
        reference="reference_visiteurs.csv",
        exemple="exemple_visiteurs.csv",
        variables=[
            var("Administrative", "Pages « compte / administratif » vues", 1, 0, 27, entier=True),
            var("Administrative_Duration", "Durée sur ces pages (s)", 9.0, 0.0, 3398.75, pas=1.0),
            var("Informational", "Pages d'information vues", 0, 0, 24, entier=True),
            var("Informational_Duration", "Durée sur ces pages (s)", 0.0, 0.0, 2549.38, pas=1.0),
            var("ProductRelated", "Pages produits vues", 18, 0, 705, entier=True),
            var("ProductRelated_Duration", "Durée sur les pages produits (s)", 608.94, 0.0, 63973.52, pas=10.0),
            var("BounceRates", "Taux de rebond (0 à 0,2)", 0.0029, 0.0, 0.2, pas=0.001),
            var("ExitRates", "Taux de sortie (0 à 0,2)", 0.025, 0.0, 0.2, pas=0.001),
            var("PageValues", "Valeur moyenne des pages", 0.0, 0.0, 361.76, pas=1.0),
            var("SpecialDay", "Proximité d'un jour spécial (0 à 1)", 0.0, 0.0, 1.0, pas=0.2),
        ],
        classes={
            0: ("visiteur engagé", "Beaucoup de pages vues et des durées longues. "
                                   "Environ 17 % de ces sessions se terminent par un achat."),
            1: ("visiteur de passage (rebond)", "2 à 3 pages vues, fort taux de rebond et de sortie. "
                                                "Presque aucune de ces sessions ne se termine par un achat."),
        },
    ),
    # ------------------------------------------------------------------ ML 7
    "bitcoin": dict(
        titre="Adresses Bitcoin",
        icone="₿",
        cours="ML 7",
        objet="adresse",
        resume="Décrire le comportement d'une adresse Bitcoin sur une journée.",
        donnees="BitcoinHeist – 2,9 millions de lignes, 6 variables",
        modele="KMeans (k = 6)",
        chaine="log_acp",
        fichiers=("scaler_bitcoin", "pca_encoder_bitcoin", "model_bitcoin"),
        reference="reference_bitcoin.csv",
        exemple="exemple_bitcoin.csv",
        note_import="Dans le fichier, `income` doit être en **satoshi** (1 BTC = 100 000 000 satoshi), "
                    "comme dans le jeu de données d'origine.",
        # Dans le formulaire, income est saisi en BTC puis converti en satoshi
        conversion={"income": 100_000_000},
        variables=[
            var("length", "Longueur de la plus longue chaîne (0 à 144)", 8, 0, 144, entier=True),
            var("weight", "Poids : fusion de pièces reçues", 0.25, 0.0, 1943.75, pas=0.01),
            var("count", "Transactions de départ", 1, 1, 14497, entier=True),
            var("looped", "Transactions reliées par plusieurs chemins", 0, 0, 14496, entier=True),
            var("neighbors", "Transactions qui envoient à l'adresse", 2, 1, 12920, entier=True),
            var("income", "Montant reçu (en BTC)", 2.0, 0.0, 499644.0, pas=0.1,
                aide="Saisi en BTC, converti automatiquement en satoshi."),
        ],
        classes={
            0: ("adresse de collecte (très active)", "Fusionne beaucoup de pièces venant de nombreux expéditeurs. "
                                                    "≈ 3,5 % de rançongiciels, soit 2,5 fois la moyenne (1,42 %)."),
            1: ("petits paiements", "Chaîne moyenne, peu de transactions, petits montants. ≈ 2 % de rançongiciels."),
            2: ("gros montants", "Gros montants reçus. Presque aucun rançongiciel (≈ 0,3 %)."),
            3: ("longue chaîne sans boucle", "Longue chaîne issue de nombreuses transactions. ≈ 1,6 % de rançongiciels."),
            4: ("paiement direct", "Chaîne très courte : paiement presque direct. ≈ 1,7 % de rançongiciels."),
            5: ("longue chaîne avec boucles (mélange)", "Les pièces se séparent puis se regroupent beaucoup. "
                                                        "≈ 0,6 % de rançongiciels."),
        },
        avertissement="Un profil décrit un comportement : il ne prouve pas qu'une adresse est liée à un rançongiciel.",
    ),
}
