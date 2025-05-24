Optimisation
==============================================

Introduction
------------

L'optimisation d'un modèle de géolocalisation commence avant tout par l'**architecture elle-même**. Avant de se concentrer sur l'ajustement des hyperparamètres ou l'augmentation des données, il est crucial d'explorer différentes approches architecturales pour maximiser la capacité du modèle à exploiter les informations visuelles et contextuelles disponibles. Cette section présente quatre architectures innovantes, chacune adoptant une philosophie distincte pour résoudre le défi complexe de la prédiction GPS à partir d'images et de métadonnées urbaines.

Mécanisme d'Attention et Couches Profondes
-----------------------------------------------------------

**Idée principale :** Utiliser un mécanisme d'attention pour pondérer intelligemment les caractéristiques visuelles et contextuelles.

**Approche :** Réseau dense 3 couches pour les métadonnées + mécanisme d'attention + connexions résiduelles + architecture en entonnoir (1024 → 512 → 256 → 2).

**Avantage :** Le modèle peut "se concentrer" sur les aspects les plus pertinents de chaque image selon le contexte urbain.

Branches Séparées et Fusion Tardive
----------------------------------------------------

**Idée principale :** Traiter séparément les informations visuelles et contextuelles avant de les fusionner.

**Approche :** Branche image (ResNet → 1024 → 512 → 256) + branche ville (features → 64 → 128 → 256) + fusion tardive + prédicteurs séparés lat/lon.

**Avantage :** Chaque modalité (visuelle/contextuelle) peut développer sa propre expertise avant la combinaison finale.

Auto-Encodeur et Attention Multi-Têtes
-------------------------------------------------------

**Idée principale :** Apprendre une représentation compacte des données urbaines et analyser sous plusieurs perspectives.

**Approche :** Auto-encodeur pour compression urbaine (features → 32 → 16 → 32 → 64) + 4 têtes d'attention multi-perspectives + architecture en entonnoir progressive (512 → 256 → 128 → 64 → 2).

**Avantage :** Capture différents patterns géographiques (climat, architecture, végétation, etc.) en parallèle avec une représentation dense des caractéristiques urbaines.

Ensemble avec Vote Pondéré
-------------------------------------------

**Idée principale :** Combiner plusieurs "experts" spécialisés pour améliorer la robustesse des prédictions.

**Approche :** 3 experts avec architectures différentes (simple 64→64, avec dropout 128→64, complexe 32→96→64) + régresseurs spécialisés + vote pondéré adaptatif avec poids apprenables.

**Avantage :** Combine les forces de plusieurs approches, réduisant le risque de sur-apprentissage et améliorant la généralisation.

.. list-table:: Philosophies et Caractéristiques des Architectures
   :widths: 20 25 15 40
   :header-rows: 1

   * - Architecture
     - Philosophie
     - Complexité
     - Focus Principal
   * - **V1**
     - "Attention sélective"
     - Moyenne
     - Pondération intelligente des features
   * - **V2**
     - "Spécialisation puis fusion"
     - Faible
     - Traitement séparé des modalités
   * - **V3**
     - "Représentation dense + perspectives multiples"
     - Élevée
     - Compression et analyse multi-angles
   * - **V4**
     - "Sagesse collective"
     - Très élevée
     - Consensus d'experts diversifiés

Résultats Expérimentaux
----------------------

Les quatre architectures ont été évaluées dans des conditions identiques avec les mêmes hyperparamètres pour garantir une comparaison équitable. Les tests ont été menés sur le même dataset avec une validation croisée robuste.

**Résultat principal :** L'Architecture V1 (Mécanisme d'Attention et Couches Profondes) s'est révélée être la plus performante, démontrant l'efficacité de l'approche basée sur l'attention pour la géolocalisation d'images.

.. figure:: _static/architecture_benchmark.png
   :alt: Comparaison des performances des quatre architectures
   :width: 800px
   :align: center
   
   Benchmark des architectures

Les résultats révèlent des performances différenciées selon les métriques évaluées. L'Architecture V1 domine sur les distances médianes et moyennes avec les erreurs les plus faibles, confirmant sa supériorité pour la précision de localisation. L'Architecture V4 (Ensemble) montre une performance mitigée : excellente sur le R² Score mais moins efficace sur les métriques de distance, suggérant une bonne capacité de généralisation mais des erreurs ponctuelles plus importantes.

Cette analyse comparative valide l'Architecture V1 comme solution optimale pour les applications nécessitant une précision géographique élevée et constante.

.. note::

   Pour obtenir les meilleures performances du modèle ResNetGPS, une optimisation automatique des hyperparamètres a été réalisée en utilisant le framework **Optuna**.

   L'optimisation a été effectuée sur 20 essais avec 10 époques chacun, permettant un compromis optimal entre temps de calcul et qualité d'exploration.
   
   **Meilleur essai :**
   
   - **Distance médiane :** 24.01 km
   
   **Hyperparamètres optimaux :**
   
   .. code-block:: text
   
      batch_size: 64
      learning_rate: 3.24e-04
      weight_decay: 4.86e-06
      dropout_city_1: 0.123
      dropout_city_2: 0.093
      dropout_fc_1: 0.132
      dropout_fc_2: 0.189
      dropout_fc_3: 0.123
      optimizer: AdamW
      scheduler_factor: 0.375
      scheduler_patience: 5
   
   **Configuration de l'optimisation :**
   
   - Nombre d'essais : 20
   - Époques par essai : 10
   - Métrique d'optimisation : Distance médiane (km)
   
   .. figure:: _static/optuna_optimization_plot.png
      :alt: Graphique d'optimisation Optuna
      :width: 80%
      :align: center
      
      Évolution de l'optimisation des hyperparamètres avec Optuna
      
Modèle Final  
----------------------

Ce modèle est une architecture hybride combinant un extracteur de caractéristiques visuelles (ResNet50) avec un traitement profond des métadonnées contextuelles (caractéristiques des villes). Il intègre un **mécanisme d'attention** permettant de pondérer dynamiquement l'importance relative des différentes caractéristiques, suivi d'un **réseau de régression profond** avec **connexions résiduelles**. L'objectif est de prédire des coordonnées GPS à partir d'une image et de métadonnées associées.

Code du Modèle
~~~~~~~~~~~~~

.. code-block:: python

    class ResNetGPSModelV1(nn.Module):
        def __init__(self, num_city_features):
            super(ResNetGPSModelV1, self).__init__()

            # ResNet50 comme extracteur de caractéristiques (inchangé)
            self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=0)

            # Gel des premières couches
            for name, param in self.resnet.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

            resnet_features = 2048

            # Traitement des caractéristiques de ville avec plus de profondeur
            self.city_features_processor = nn.Sequential(
                nn.Linear(num_city_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)
            )

            # Mécanisme d'attention simple pour combiner les features
            self.attention = nn.Sequential(
                nn.Linear(resnet_features + 128, 512),
                nn.ReLU(),
                nn.Linear(512, resnet_features + 128),
                nn.Sigmoid()
            )

            # Régression avec architecture plus profonde et connexions résiduelles
            self.fc1 = nn.Linear(resnet_features + 128, 1024)
            self.bn1 = nn.BatchNorm1d(1024)
            self.dropout1 = nn.Dropout(0.15)

            self.fc2 = nn.Linear(1024, 512)
            self.bn2 = nn.BatchNorm1d(512)
            self.dropout2 = nn.Dropout(0.1)

            self.fc3 = nn.Linear(512, 256)
            self.bn3 = nn.BatchNorm1d(256)
            self.dropout3 = nn.Dropout(0.05)

            # Connexion résiduelle
            self.residual_connection = nn.Linear(resnet_features + 128, 256)

            self.fc_final = nn.Linear(256, 2)

        def forward(self, image, city_features):
            # Traitement de l'image
            image_feats = self.resnet(image)

            # Traitement des caractéristiques de ville
            city_feats = self.city_features_processor(city_features)

            # Combinaison des caractéristiques
            combined = torch.cat((image_feats, city_feats), dim=1)

            # Mécanisme d'attention
            attention_weights = self.attention(combined)
            combined = combined * attention_weights

            # Régression avec connexion résiduelle
            x = self.fc1(combined)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.dropout1(x)

            x = self.fc2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = self.dropout2(x)

            x = self.fc3(x)
            x = self.bn3(x)
            x = torch.relu(x)
            x = self.dropout3(x)

            # Connexion résiduelle
            residual = self.residual_connection(combined)
            x = x + residual

            coordinates = self.fc_final(x)

            return coordinates

Explication des Étapes
~~~~~~~~~~~~~

1. **Extraction des caractéristiques visuelles**

   - **ResNet50** : réseau convolutif pré-entraîné (2048 neurones en sortie).
   - Couches gelées sauf ``layer4``.

2. **Encodage des données de ville**

   Réseau dense à 3 couches :
   
   - ``Linear(num_city_features → 128)``, puis ``ReLU``
   - ``Linear(128 → 256)``, puis ``ReLU``
   - ``Linear(256 → 128)``
   
   Résultat : 128 neurones.

3. **Mécanisme d'attention**

   - Combine image (2048) + ville (128) → vecteur de taille 2176.
   - Passage dans :
   
     - ``Linear(2176 → 512)``, ``ReLU``
     - ``Linear(512 → 2176)``, ``Sigmoid``
     
   - Applique un masque attentionnel :
   
   .. code-block:: text
   
      combined_att = combined ⊙ σ(W₂(ReLU(W₁ · combined)))

4. **Réseau de régression profond (4 couches)**

   - ``fc1 : 2176 → 1024``, ``ReLU``, ``Dropout(0.15)``
   - ``fc2 : 1024 → 512``, ``ReLU``, ``Dropout(0.1)``
   - ``fc3 : 512 → 256``, ``ReLU``, ``Dropout(0.05)``
   - Résidu ajouté : ``Linear(2176 → 256)``
   
   Formule finale :
   
   .. code-block:: text
   
      x = fc3(fc2(fc1(combined_att))) + residual(combined)

5. **Prédiction finale**

   - ``fc_final : 256 → 2`` (coordonnées GPS).

Résumé des Dimensions
~~~~~~~~~~~~~

.. list-table:: Dimensions par étape
   :widths: 50 25
   :header-rows: 1
   
   * - Étape
     - Dimensions (neurones)
   * - Image (ResNet50)
     - 2048
   * - Ville (MLP 3 couches)
     - 128
   * - Concatenation
     - 2176
   * - Attention (interne)
     - 2176
   * - fc1
     - 1024
   * - fc2
     - 512
   * - fc3
     - 256
   * - Connexion résiduelle
     - 256
   * - Sortie finale
     - 2

Architecture Générale
~~~~~~~~~~~~~

Le flux de données suit cette architecture :

::

    Image (224x224x3)
           ↓
    ResNet50 (features: 2048)
           ↓
           ├─────────────────────┐
           ↓                     ↓
    City Features (n_features)   │
           ↓                     │
    MLP 3 couches (128)          │
           ↓                     │
    Concatenation (2176) ←───────┘
           ↓
    Mécanisme d'Attention
           ↓
    Réseau de Régression (4 couches)
           ↓
    Coordonnées GPS (lat, lon)

.. figure:: _static/training_results_optimal.png
      :alt: Résultats du modèle finale
      :width: 80%
      :align: center
      
      Entrainement du modèle final 

.. toctree::
   :maxdepth: 3
   :caption: Contenu :

   .. prev:: Modèle
   .. next:: Usage