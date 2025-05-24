Usage
=====

Le modèle MarocExplore final peut être utilisé de deux façons distinctes selon vos besoins : l'**évaluation** sur un dataset existant avec des coordonnées de référence, ou la **prédiction** de localisation pour une image individuelle. Ces deux modes d'utilisation permettent respectivement de mesurer les performances du modèle et d'exploiter ses capacités de géolocalisation en production.

.. attention::
   **Prérequis techniques**

   - Python 3.8+
   - PyTorch 1.12+
   - timm 0.6+
   - pandas, numpy, PIL
   - CUDA recommandé pour l'inférence rapide

Prérequis Communs
-----------------

Avant d'utiliser le modèle, vous devez :

1. **Télécharger le modèle pré-entraîné**
   
   - Depuis Google Drive : https://drive.google.com/drive/folders/1jGl0Un0FehxftKpgI8i775mmQbTB11M2?usp=drive_link
   - Ou depuis le repository GitHub : https://github.com/amineEssahraoui/MaroxExplorer

2. **Obtenir le fichier de métadonnées** (obligatoire)
   
   Le fichier ``data_red/Métadonnées_villes.csv`` contenant les caractéristiques des villes est disponible dans le repository GitHub.

Mode 1 : Évaluation
-------------------

Ce mode permet d'évaluer les performances du modèle sur un dataset avec des coordonnées de référence connues.

**Prérequis spécifiques :**

- Un dossier contenant des images avec des identifiants et les villes qui correspondants
- Un fichier CSV contenant les coordonnées réelles de ces images
- Le fichier de métadonnées des villes 

**Utilisation :**

.. code-block:: python

   # Charger le modèle et évaluer sur un dataset
    model_path = "/content/drive/MyDrive/model/MarocExplorer_model.pth"  # Path du modèle , Remplace par ton chemin
    csv_path = "/content/maroc_data_finale.csv" # Pour les cordonnées 
    city_features_path = "/content/drive/MyDrive/Datasets/data_red/M_villes_avec_coords.csv" # Métadonnées 
    images_folder = '/content/drive/MyDrive/Datasets/data_red/iamges_red' # Images 

    evaluate_model_on_images(
        model_path=model_path,
        csv_path=csv_path,
        city_features_path=city_features_path,
        images_folder=images_folder,
        n_images=10  # Tu peux changer ce nombre
    )

**Sorties :**

- Les prédictions 
- Distance entre prédictions et coordonnées réelles

Mode 2 : Prédiction pour une Image
----------------------------------

Ce mode permet de prédire la localisation d'une image individuelle.

**Prérequis spécifiques :**

- Une image à géolocaliser
- Sélection de la ville depuis la liste disponible dans le notebook

**Utilisation :**

.. code-block:: python

    MODEL_PATH = "/content/drive/MyDrive/model/MarocExplorer_model.pth"
    CITY_FEATURES_PATH = "/content/drive/MyDrive/Datasets/data_red/M_villes_avec_coords.csv"

    IMAGE_PATH = "/content/tour-hassan-rabat-morocco-by-migel.jpeg"  # Remplacer par votre chemin
    CITY_NAME = "Rabat"  # Nom de ville supportée

    # Utiliser une des deux fonctions selon le besoin :
    # Résultat simple :
    results = predict_single_image(
        image_path=IMAGE_PATH,
        city_name=CITY_NAME,
        model_path=MODEL_PATH,
        city_features_path=CITY_FEATURES_PATH
    )

    # Résultat avec visualisation :
    # results = predict_and_visualize(
    #     image_path=IMAGE_PATH,
    #     city_name=CITY_NAME,
    #     model_path=MODEL_PATH,
    #     city_features_path=CITY_FEATURES_PATH
    # )


**Sorties :**

- Coordonnées GPS prédites (latitude, longitude)
- Distance entre la prédiction et le centre de ville 
- Visualisation optionnelle de l'image

.. note::
   La liste des villes disponibles est définie dans le notebook principal. Assurez-vous de sélectionner une ville présente dans le fichier de métadonnées pour obtenir des prédictions fiables.

.. attention::
   **Cellules à Exécuter Obligatoirement**
   
   Pour utiliser la prédiction d'une image, vous devez obligatoirement exécuter les cellules suivantes dans le notebook :
   
   1. **Cellule d'importation des bibliothèques** - Contient tous les imports nécessaires
   2. **Cellule de la fonction** ``load_model(model_path)`` - Charge le modèle pré-entraîné
   3. **Cellule de la fonction** ``prepare_city_features(city_features_path)`` - Prépare les caractéristiques des villes
   4. **Cellule de la fonction** ``haversine_distance(lat1, lon1, lat2, lon2)`` - Calcule la distance entre coordonnées GPS
   
   Ces cellules doivent être exécutées dans l'ordre avant toute tentative de prédiction.

Consultez notre **notebook Colab** :

.. raw:: html

   <a href="https://colab.research.google.com/drive/1Eepqzt34FIDokAWNGlNAyGWUfQMjIWuW#scrollTo=t9PPRKtYU3YN" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
