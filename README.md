# Face_ID_v1
Reconnaissance faciale, cadre académique code en python! 



# Reconnaissance faciale intelligente des étudiants

## 1. Présentation du projet

Ce projet consiste à développer une application de reconnaissance faciale capable de détecter un visage à partir d'une webcam ou d'une image, puis d'identifier l'étudiant correspondant parmi cinq personnes enregistrées dans une base de données.

L'application devra également reconnaître les situations dans lesquelles une personne ne figure pas dans la base de données et afficher **« Inconnu »** lorsque la correspondance obtenue n'est pas suffisamment fiable.

Ce projet est réalisé dans le cadre d'un travail académique en groupe.

## 2. Objectifs du projet

Les principaux objectifs sont les suivants :

* Détecter les visages présents dans une image ou capturés par une webcam.
* Enregistrer et organiser les images des cinq étudiants participants.
* Extraire des caractéristiques numériques permettant de comparer les visages.
* Identifier les étudiants enregistrés dans la base de données.
* Afficher le nom de la personne reconnue et un score de confiance ou de similarité.
* Afficher « Inconnu » lorsqu'aucune correspondance ne satisfait le seuil retenu.
* Évaluer les performances du système dans différentes conditions.
* Documenter les étapes de développement, les résultats et les limites de l'application.

## 3. Fonctionnalités prévues

### 3.1. Détection faciale

* Accès à la webcam.
* Chargement d'une image depuis l'ordinateur.
* Détection d'un ou de plusieurs visages.
* Affichage d'un cadre autour de chaque visage détecté.

### 3.2. Reconnaissance faciale

* Enregistrement des représentations faciales des étudiants participants.
* Comparaison des nouveaux visages avec les données enregistrées.
* Identification de l'étudiant correspondant.
* Affichage du nom et du score de confiance ou de similarité.

### 3.3. Gestion des personnes inconnues

* Définition d'un seuil de décision.
* Rejet des correspondances insuffisantes.
* Affichage de la mention « Inconnu » lorsqu'une personne ne peut pas être identifiée de manière suffisamment fiable.

### 3.4. Évaluation

* Tests sur les étudiants enregistrés.
* Tests avec des variations de pose, d'expression, de distance et d'éclairage.
* Tests sur au moins une personne absente de la base de données.
* Enregistrement et analyse des résultats.

**État du développement :** en cours.

Les fonctionnalités seront marquées comme terminées uniquement après leur implémentation et leur vérification.

## 4. Technologies utilisées

| Technologie                                           | Utilisation                                                 |
| ----------------------------------------------------- | ----------------------------------------------------------- |
| Python 3                                              | Langage de programmation principal                          |
| OpenCV                                                | Traitement d'images, accès à la webcam et détection faciale |
| NumPy                                                 | Opérations numériques                                       |
| Bibliothèque de reconnaissance faciale à sélectionner | Extraction et comparaison des représentations faciales      |
| Jupyter Notebook                                      | Expérimentations et analyse des résultats                   |
| Visual Studio Code                                    | Environnement de développement                              |
| Git                                                   | Gestion des versions du code                                |

La bibliothèque de reconnaissance faciale sera choisie en fonction de sa compatibilité avec l'environnement de développement et des besoins du projet.

## 5. Structure du projet

L'organisation prévue du projet est la suivante :

```text
face-recognition-project/
│
├── dataset/
│   ├── student_01/
│   ├── student_02/
│   ├── student_03/
│   ├── student_04/
│   └── student_05/
│
├── src/
│   ├── detect_faces.py
│   ├── encode_faces.py
│   └── recognize_faces.py
│
├── notebook/
│   └── experiments.ipynb
│
├── results/
│   ├── screenshots/
│   └── test_results.csv
│
├── main.py
├── requirements.txt
├── README.md
└── report.pdf
```

Cette structure pourra évoluer selon les besoins de l'implémentation.

### Description des principaux éléments

* `dataset/` : contient les images des étudiants participants.
* `src/detect_faces.py` : gère la détection des visages.
* `src/encode_faces.py` : prépare les représentations numériques des visages enregistrés.
* `src/recognize_faces.py` : compare les visages et détermine les correspondances.
* `notebook/experiments.ipynb` : documente les expérimentations.
* `results/` : contient les captures d'écran et les résultats des tests.
* `main.py` : sert de point d'entrée à l'application.
* `requirements.txt` : liste les dépendances Python.
* `report.pdf` : contient le rapport final du projet.

## 6. Prérequis

Avant de lancer le projet, les outils suivants doivent être disponibles :

* Python 3.
* pip, le gestionnaire de paquets Python.
* Visual Studio Code ou un éditeur équivalent.
* Git, si le projet est géré avec un dépôt Git.
* Une webcam pour les tests en temps réel, ou des images pour les tests hors ligne.

Les versions précises des bibliothèques seront définies et enregistrées dans `requirements.txt` après validation de l'environnement.

## 7. Installation

### Étape 1 : récupérer le projet

Si le projet est hébergé dans un dépôt Git :

```bash
git clone <URL_DU_DEPOT>
cd face-recognition-project
```

Remplacer `<URL_DU_DEPOT>` par l'adresse réelle du dépôt.

Si le projet est déjà présent sur votre ordinateur, ouvrez directement son dossier dans le terminal.

### Étape 2 : créer un environnement virtuel

Sous Windows :

```powershell
python -m venv .venv
```

Activer l'environnement :

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloque l'activation, il est possible d'utiliser directement l'interpréteur de l'environnement virtuel ou de suivre la documentation Python relative à l'activation des environnements.

Sous Linux ou macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Étape 3 : installer les dépendances

Lorsque le fichier `requirements.txt` contient les bibliothèques validées :

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si ce fichier n'est pas encore prêt, les dépendances devront être installées après le choix de la méthode de reconnaissance faciale.

## 8. Organisation du jeu de données

Le jeu de données doit contenir environ 20 à 40 images pour chacun des cinq étudiants participants.

Exemple :

```text
dataset/
├── student_01/
│   ├── student_01_001.jpg
│   ├── student_01_002.jpg
│   └── ...
├── student_02/
├── student_03/
├── student_04/
└── student_05/
```

Les images devraient présenter des variations raisonnables :

* Visage de face et légèrement tourné.
* Expressions faciales différentes.
* Distances variables par rapport à la caméra.
* Conditions d'éclairage différentes.

Une partie des images sera utilisée pour l'enregistrement des étudiants. D'autres seront réservées aux tests afin d'évaluer le système sur des images qu'il n'a pas utilisées pendant cette phase.

Les images doivent être collectées avec le consentement des participants et utilisées uniquement dans le cadre pédagogique annoncé.

## 9. Utilisation de l'application

Les commandes exactes seront ajoutées après l'implémentation de l'application.

Le fonctionnement prévu est le suivant :

1. L'utilisateur lance l'application.
2. Il choisit d'utiliser la webcam ou de charger une image.
3. Le système détecte les visages présents.
4. Chaque visage détecté est comparé aux représentations enregistrées.
5. Le système vérifie si la meilleure correspondance satisfait le seuil de décision.
6. Le nom de l'étudiant et le score correspondant sont affichés si la correspondance est acceptée.
7. La mention « Inconnu » est affichée si aucune correspondance n'est suffisamment fiable.

Les instructions de lancement seront précisées dès que le point d'entrée et les modules seront opérationnels.

## 10. Tests et évaluation

L'évaluation portera au minimum sur trois catégories de tests.

| Test   | Description                                                 | Résultat attendu                                             |
| ------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| Test A | Étudiant enregistré dans des conditions normales            | Identification correcte                                      |
| Test B | Étudiant enregistré avec une pose ou un éclairage différent | Identification correcte lorsque les conditions le permettent |
| Test C | Personne absente de la base de données                      | Rejet de la correspondance et affichage de « Inconnu »       |

Pour chaque test, les informations suivantes seront consignées :

* Identifiant du test.
* Identité attendue.
* Identité prédite.
* Score de confiance ou de similarité.
* Détection réussie ou non du visage.
* Résultat correct ou incorrect.

Le seuil de décision sera ajusté à partir des résultats de validation. Le score affiché ne doit pas être interprété comme une probabilité d'identité correcte, sauf si la méthode utilisée permet réellement cette interprétation.

## 11. Limites du système

Les performances de l'application peuvent dépendre de plusieurs facteurs :

* Qualité et résolution des images.
* Conditions d'éclairage.
* Orientation du visage.
* Présence d'obstacles ou d'occlusions.
* Qualité de la webcam.
* Nombre et diversité des images enregistrées.
* Méthode et seuil de reconnaissance utilisés.

Le jeu de données étant limité à cinq étudiants, les résultats ne pourront pas être généralisés à l'ensemble de la population.

L'application constitue un prototype pédagogique et ne doit pas être utilisée comme un système de surveillance ou comme un dispositif de contrôle d'identité à fort enjeu.

## 12. Confidentialité et utilisation responsable

La collecte et le traitement des images faciales doivent respecter les principes suivants :

* Obtenir l'accord des participants avant la collecte des images.
* Expliquer clairement l'objectif du projet.
* Limiter l'accès aux images aux personnes autorisées.
* Éviter de publier les images ou les données faciales dans un dépôt public.
* Ne conserver les données que pendant la durée nécessaire au projet.
* Supprimer ou traiter les données conformément aux consignes de l'établissement à la fin du projet.

Le dépôt Git ne doit contenir ni données faciales personnelles ni informations confidentielles.

## 13. Répartition du travail

Le travail est organisé autour de plusieurs responsabilités complémentaires :

* **Collecte et préparation des données :** organisation du jeu de données et préparation des images.
* **Détection faciale :** détection des visages à partir d'images ou de la webcam.
* **Reconnaissance faciale :** extraction des représentations numériques et comparaison des visages.
* **Intégration :** assemblage des modules dans une application fonctionnelle.
* **Tests et documentation :** évaluation des performances, préparation du notebook et rédaction du rapport.

Chaque membre doit contribuer concrètement au projet et être capable d'expliquer la partie qu'il a réalisée.

## 14. Livrables attendus

À la fin du projet, les éléments suivants devront être disponibles :

* Code Python complet.
* Jeu de données organisé.
* Notebook d'expérimentation.
* Fichier `requirements.txt`.
* Guide d'installation et d'utilisation.
* Rapport de 3 à 5 pages.
* Captures d'écran ou courte démonstration.
* Résultats des tests sur des personnes enregistrées et inconnues.

## 15. État d'avancement

* [ ] Environnement Python configuré.
* [ ] Structure du projet créée.
* [ ] Détection faciale fonctionnelle.
* [ ] Jeu de données préparé.
* [ ] Méthode de reconnaissance choisie.
* [ ] Reconnaissance des étudiants fonctionnelle.
* [ ] Gestion des personnes inconnues implémentée.
* [ ] Tests réalisés et documentés.
* [ ] Notebook finalisé.
* [ ] Documentation complétée.
* [ ] Rapport final rédigé.
* [ ] Démonstration validée.

---

**Projet académique :** Reconnaissance faciale intelligente des étudiants
**Langage principal :** Python
**Équipe :** 5 membres
**Statut :** En cours de développement
