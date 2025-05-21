Modèle
===================

L'objectif du projet **MarocExplorer** est de prédire les coordonnées géographiques (latitude et longitude) à partir des **images marocaines** et de leurs **métadonnées associées**.  
Ce problème n'est pas un simple problème de **Computer Vision (CV)** classique, mais plutôt un problème de **régression**, où les **caractéristiques cachées des images** et les **métadonnées** jouent un rôle central.  


Architecture du modèle
----------------------

Notre approche repose sur l'idée d'utiliser un **modèle préentraîné** pour extraire les **caractéristiques visuelles** des images.  
Ensuite, nous ajoutons des **couches supplémentaires** afin d'intégrer également les **données géographiques** (villes, régions, etc.) et de **transformer la tâche en un problème de régression**, avec pour cibles les coordonnées **latitude** et **longitude**.  

.. note::
   Cet apprentissage est **coûteux en ressources**, notamment lorsque l'on utilise des modèles lourds comme **ResNet50** ou **DenseNet121**.
   Il est généralement préférable d'utiliser un **taux d'apprentissage plus faible** pour éviter de **déstabiliser les poids préentraînés**.  

**Principales étapes de l'architecture :**  

1. **Input :**  
   - Les entrées du modèle sont les **images marocaines** accompagnées de **métadonnées** sur leur ville.  

2. **Extraction des caractéristiques :**  

   - **Modèle d'images :** Utilise un modèle préentraîné (**ResNet18**, par exemple) pour extraire les **caractéristiques visuelles** de l'image.  

   - **Modèle de caractéristiques géographiques :** Utilise un **réseau de neurones multicouche (MLP)** pour extraire les **caractéristiques de la ville**.  

3. **Fusion des caractéristiques :**  
   - Les caractéristiques extraites des images et des villes sont **concatenées** pour former un **vecteur combiné**.  

4. **Modèle de régression :**  
   - Ce modèle prend en entrée le **vecteur fusionné** et prédit les **coordonnées GPS (latitude, longitude)**.  
  
.. container:: Architecture-container
   
   .. image:: _static/modèle.drawio.png
      :alt: Architecture du modèle
      :width: 85%
      :align: center

.. raw:: html

   <style>
   .pipeline-container {
       margin-bottom: 40px;
   }
   </style>

Méthodologie expérimentale
--------------------------

Prétraitement des données
~~~~~~~~~~~~~~~~~~~~~~~~~

Pour garantir l'**uniformité** entre les modèles, nous avons appliqué le même prétraitement aux données:

1. Conversion des images en **tensors** de taille spécifique selon chaque modèle.
   
   * **ResNet50**: taille d'images **(3, 224, 224)**
   * **ResNet18**: taille d'images **(3, 224, 224)**
   * **EfficientB0**: taille d'images **(3, 224, 224)**
   * **DenseNet121**: taille d'images **(3, 224, 224)**
   .. code-block:: python

    # Transformations d'images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


2. Préparation des **données des villes**, qui sont textuelles, en **encodant les caractéristiques** pour les rendre compatibles avec les modèles de régression.
    .. code-block:: python

     city_features = pd.read_csv(city_features_path)
        
     # Encodage des caractéristiques catégorielles
     categorical_cols = ['Architecture', 'Terrain', 'Climat', 'Eau_proche', 
                         'Montagne', 'Taille', 'Points_intérêt', 'Grande_ville_proche']
        
     for col in categorical_cols:
         le = LabelEncoder()
         city_features[col] = le.fit_transform(city_features[col])
        
     # Normalisation des caractéristiques numériques
     numerical_cols = ['Distance_côte']
     scaler = StandardScaler()
     city_features[numerical_cols] = scaler.fit_transform(city_features[numerical_cols])
        
     # Sélection des caractéristiques à utiliser
     feature_cols = categorical_cols + numerical_cols
     city_features = city_features[['Ville'] + feature_cols + ['lat', 'lng']]
    
3. Normalisation des données pour **améliorer la stabilité de l'apprentissage**.

.. note::
   La **normalisation des coordonnées géographiques** est essentielle pour éviter les écarts trop importants lors de l'entraînement.

Protocole expérimental
~~~~~~~~~~~~~~~~~~~~~

Pour garantir l'**équité de la comparaison** entre les modèles, nous avons appliqué les règles suivantes:

- Utilisation de la même **partition des données** pour tous les modèles (80% entraînement, 20% validation).
- Entraînement sur un nombre d'**époques fixé** (maximum 20) pour assurer la convergence.
- Calcul des métriques sur les **données de validation** à chaque époque pour observer l'évolution des performances.
- Sauvegarde du **meilleur modèle** selon (**Median Distance Error**).
- Utiliser les mêmes hyperparamètres pour tous les modèles

Hyperparamètres communs
^^^^^^^^^^^^^^^^^^^^^^

- **Optimiseur** : Adam  
- **Taux d'apprentissage (learning rate)** : 1e-4 (valeur faible pour éviter de **déstabiliser les poids préentraînés**)  
- **Taille de batch (batch size)** : 32  
- **Nombre d'epochs** : 15 (le calcul est très coûteux)  
- **Fonction de perte** : MSE (Mean Squared Error)  
- **Dimensions cachées pour les caractéristiques de ville** :  

  - **city_hidden_dim1** : 128  
  - **city_hidden_dim2** : 74  
- **Dropout pour le sous-réseau de ville** :  

  - **Après city_hidden_dim1** : 0.3  
- **Dimensions pour la régression combinée** :  

  - **Regression dim1** : 512  
  - **Regression dim2** : 128  

- **Dropout pour la régression combinée** :  

  - **Après Regression dim1** : 0.3  
  - **Après Regression dim2** : 0.2

Exemple (resnet50)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class ResNet50GPSModel(nn.Module):
        def __init__(self, num_city_features):
            super(ResNet50GPSModel, self).__init__()
            
            # ResNet50 comme extracteur de caractéristiques
            self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=0)
            
            # Gel des premières couches pour un entraînement plus rapide et prévenir le surapprentissage
            for name, param in self.resnet.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
            
            # Nombre de caractéristiques de sortie du ResNet50
            resnet_features = 2048  # ResNet50 a 2048 caractéristiques de sortie
            
            # Traitement des caractéristiques de ville
            self.city_features_processor = nn.Sequential(
                nn.Linear(num_city_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 74),  # 74 villes comme demandé
                nn.ReLU()
            )
            combined_features = resnet_features + 74  # 2048 (image) + 74 (ville)
            
            # Régression combinée
            self.regression = nn.Sequential(
                nn.Linear(combined_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 2)  # Sortie: [latitude, longitude]
            )
        
        def forward(self, image, city_features):
            # Traitement de l'image
            image_feats = self.resnet(image)
            
            # Traitement des caractéristiques de ville
            city_feats = self.city_features_processor(city_features)
            
            # Combinaison des caractéristiques
            combined = torch.cat((image_feats, city_feats), dim=1)
            
            # Prédiction des coordonnées
            coordinates = self.regression(combined)
            
            return coordinates

Comparaison des modèles
-----------------------

Sélection des architectures candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pour extraire les **caractéristiques visuelles des images**, nous avons testé plusieurs modèles de réseaux de neurones convolutifs (**CNN**) préentraînés sur **ImageNet**.  
Ces modèles sont reconnus pour leur efficacité dans la **classification d'images** et peuvent être adaptés à notre tâche de **régression géographique** en ajoutant des couches spécifiques.  

**Tableau comparatif des modèles :**  

.. list-table:: Comparaison des modèles CNN pour l'extraction des caractéristiques
   :header-rows: 1
   :widths: 20 40 20 20

   * - Modèle
     - Définition simple
     - Particularité
     - Inconvénient
   * - ResNet50
     - Réseau à 50 couches avec des connexions résiduelles
     - Bonne performance pour la classification, profondeur importante
     - Poids importants, coûteux en calcul
   * - ResNet18
     - Réseau à 18 couches avec des connexions résiduelles
     - Architecture plus légère, rapide à entraîner
     - Moins performant pour des tâches complexes
   * - EfficientNet_B0
     - Modèle compact et efficace pour la vision
     - Bon compromis entre performance et légèreté
     - Moins performant que les versions plus grandes (B4, B7)
   * - DenseNet121
     - Réseau dense avec 121 couches, forte réutilisation des caractéristiques
     - Faible nombre de paramètres par rapport à sa profondeur
     - Plus lent en pratique, consommation mémoire élevée

.. note:: 
   Le choix du modèle dépend d'un **équilibre entre précision et efficacité**.  
   Pour des **applications en temps réel**, un modèle léger comme **ResNet18** ou **EfficientNet_B0** est préférable.  
   Pour des résultats plus précis, **ResNet50** et **DenseNet121** offrent une meilleure extraction de caractéristiques, mais au prix d'une **complexité accrue**.

Évaluation des performances
--------------------------

Critères d'évaluation  
~~~~~~~~~~~~~~~~~~~~~

Pour chaque modèle testé, les performances ont été mesurées en utilisant les **métriques suivantes** :  

.. list-table:: Métriques d'évaluation des modèles
   :header-rows: 1
   :widths: 20 50 30

   * - Métrique
     - Description
     - Particularité
   * - MSE
     - Mean Squared Error
     - Évalue l'erreur quadratique moyenne entre les prédictions et les valeurs réelles.
   * - R²
     - Coefficient de détermination
     - Indique la proportion de la variance expliquée par le modèle.
   * - Mean Distance Error
     - Distance moyenne entre les coordonnées prédites et les coordonnées réelles
     - Mesure l'exactitude géographique.
   * - Median Distance Error
     - Médiane de la distance entre les coordonnées prédites et les coordonnées réelles
     - Réduit l'impact des valeurs aberrantes.
   * - Epoch du meilleur résultat
     - Numéro de l'époque où la perte a été minimale
     - Permet d'identifier le meilleur point d'entraînement.  

.. note:: 
   L'utilisation de plusieurs métriques permet d'avoir une **évaluation complète** des performances du modèle,  
   en tenant compte à la fois de la **précision géographique** et de la **stabilité des prédictions**.  

Procédure d'évaluation  
~~~~~~~~~~~~~~~~~~~~~

Pour chaque architecture de modèle, nous avons suivi les étapes suivantes :

1. **Préparation des données** :  
   - Diviser les données en deux ensembles : entraînement et validation.
   
2. **Entraînement** :  
   - Former le modèle sur les données d'entraînement.  
   - À chaque époque, évaluer les performances sur l'ensemble de validation en calculant les erreurs de prédiction (distance moyenne et médiane des erreurs).  
   - Si la médiane de l'erreur est la plus faible observée jusqu'à présent, sauvegarder le modèle.  

3. **Sélection du meilleur modèle** :  
   - Choisir le modèle sauvegardé ayant la plus faible médiane d'erreur sur l'ensemble de validation.  

4. **Évaluation finale** :  
   - Évaluer le meilleur modèle sur l'ensemble de test pour mesurer les performances finales.  
  
.. code-block:: python

  def main():
    print("Preparing data...")
    
    # Charger les caractéristiques des villes
    city_features, feature_cols = prepare_city_features(city_features_path)
    
    if city_features is None:
        print("City features not found. Please check the path.")
        return
    
    print(f"Using model with both image and city features (total city features: {len(feature_cols)})")
    
    # Créer le jeu de données
    dataset = ImageGPSDataset(csv_file=csv_path, 
                             images_folder=images_folder, 
                             city_features=city_features, 
                             feature_cols=feature_cols, 
                             transform=transform)
    
    # Vérifier si le dataset est vide après filtrage
    if len(dataset) == 0:
        print("ERROR: No valid data left after filtering out images without city metadata!")
        return
    
    # Division du jeu de données
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print("Creating model...")
    model = ResNet50SModel(num_city_features=len(feature_cols)).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    print("Starting training...")
    train_losses = []
    val_losses = []
    mean_distances = []
    median_distances = []
    r2_scores = []
    mse_scores = []
    
    best_model_info = {
        'epoch': 0,
        'median_distance': float('inf'),
        'mean_distance': 0,
        'r2': 0,
        'mse': 0,
        'state_dict': None
    }
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        val_loss, mean_distance, median_distance, r2, mse = validate_model(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        mean_distances.append(mean_distance)
        median_distances.append(median_distance)
        r2_scores.append(r2)
        mse_scores.append(mse)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Mean Distance: {mean_distance:.2f} km, Median Distance: {median_distance:.2f} km")
        print(f"R² Score: {r2:.4f}, MSE: {mse:.6f}")
        
        scheduler.step(val_loss)
        
        # Sauvegarde du meilleur modèle basé sur la médiane de distance
        if median_distance < best_model_info['median_distance']:
            best_model_info = {
                'epoch': epoch + 1,
                'median_distance': median_distance,
                'mean_distance': mean_distance,
                'r2': r2,
                'mse': mse,
                'state_dict': model.state_dict().copy()
            }
            print(f"New best model found at epoch {epoch+1} with median distance: {median_distance:.2f} km")
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_city_features': len(feature_cols),
                'best_median_distance': median_distance
            }, output_model_path)
    
    print("\nTraining complete!")
    
    # Affichage des résultats de la meilleure époque
    print("\n--- Best Model Performance ---")
    print(f"Best Epoch: {best_model_info['epoch']}")
    print(f"Median Distance: {best_model_info['median_distance']:.2f} km")
    print(f"Mean Distance: {best_model_info['mean_distance']:.2f} km")
    print(f"R² Score: {best_model_info['r2']:.4f}")
    print(f"MSE: {best_model_info['mse']:.6f}")
    
    # Tracer les résultats d'entraînement
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(mean_distances, label='Mean Distance (km)')
    plt.plot(median_distances, label='Median Distance (km)')
    plt.axvline(x=best_model_info['epoch']-1, color='r', linestyle='--', label=f'Best Epoch ({best_model_info["epoch"]})')
    plt.title('Prediction Error')
    plt.xlabel('Epoch')
    plt.ylabel('Distance (km)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(r2_scores, label='R² Score')
    plt.axvline(x=best_model_info['epoch']-1, color='r', linestyle='--', label=f'Best Epoch ({best_model_info["epoch"]})')
    plt.title('R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(mse_scores, label='MSE')
    plt.axvline(x=best_model_info['epoch']-1, color='r', linestyle='--', label=f'Best Epoch ({best_model_info["epoch"]})')
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/resnet50_training_results.png')
    plt.show()
    
    # Charger le meilleur modèle et faire une évaluation finale
    best_model = ResNetGPSModel(num_city_features=len(feature_cols)).to(device)
    best_model.load_state_dict(best_model_info['state_dict'])
    _, final_mean, final_median, final_r2, final_mse = validate_model(
        best_model, val_loader, criterion, device
    )
    
    print("\n--- Final Evaluation with Best Model ---")
    print(f"Mean Distance: {final_mean:.2f} km")
    print(f"Median Distance: {final_median:.2f} km")
    print(f"R² Score: {final_r2:.4f}")
    print(f"MSE: {final_mse:.6f}")

  if __name__ == "__main__":
    main()
Les modèles ont été évalués sur deux aspects principaux :

- **Qualité de reconstruction des coordonnées géographiques**.  
- **Précision des prédictions de localisation**.  

Résultats et analyses
--------------------

Benchmark des performances
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Performance comparative des modèles
   :header-rows: 1
   :widths: 15 15 10 20 20 20
   
   * - Modèle
     - MSE
     - R2
     - Median Distance Error
     - Mean Distance Error
     - Meilleure Époque
   * - ResNet50
     - 0.882663
     - 0.6430
     - 102.14 km
     - 118.21 km
     - 12
   * - ResNet18
     - 0.865191
     - 0.6426
     - 108.83 km
     - 121.43 km
     - 15
   * - EfficientB0
     - 2.167512
     - 0.1287
     - 163.87 km
     - 188.39 km
     - 12
   * - DenseNet121
     - 1.532988
     - 0.3627
     - 141.02 km
     - 160.35 km
     - 15

.. note::
   Cette évaluation comparative permet d'identifier le modèle offrant le meilleur compromis entre précision et efficacité pour notre tâche de régression géographique.

Visualisation des performances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. figure:: _static/resnet50_curve.jpg
          :width: 100%
          :alt: Courbe d'apprentissage ResNet50
          
          **ResNet50**
     
     - .. figure:: _static/resnet18_curve.jpg
          :width: 100%
          :alt: Courbe d'apprentissage ResNet18
          
          **ResNet18**
   
   * - .. figure:: _static/efficientb0_curve.jpg
          :width: 100%
          :alt: Courbe d'apprentissage EfficientB0
          
          **EfficientB0**
     
     - .. figure:: _static/densenet121_curve.jpg
          :width: 100%
          :alt: Courbe d'apprentissage DenseNet121
          
          **DenseNet121**

.. remarque::  
   Nous avons d'abord essayé d'entraîner le modèle uniquement sur les images et leurs coordonnées, sans ajouter d'autres informations (comme les **City\_features**), mais les résultats obtenus n'étaient pas satisfaisants.

Voici les résultats pour ResNet18. On remarque aisément la différence entre l'entraînement sans **City\_features** et avec.

.. list-table:: Performance de Resnet18 
   :header-rows: 1
   :widths: 15 15 10 20 20 20
   
   * - Modèle
     - MSE
     - R2
     - Median Distance Error
     - Mean Distance Error
     - Meilleure Époque
   * - ResNet18
     - 2.743270
     - 0.1461
     - 172.64 km
     - 198.79 km
     - 15

.. container:: Resnet18
   
   .. image:: _static/resnet_18_sf.jpg
      :alt: Resnet18
      :width: 60%
      :align: center

Analyse des courbes d'apprentissage
----------------------------

Les graphiques montrent l'évolution des erreurs de prédiction (moyenne et médiane) au cours des époques d'entraînement:

1. **ResNet50**:
   * Diminution progressive de l'erreur avec quelques oscillations
   * Convergence autour de l'époque 12
   * Erreur moyenne finale ~118 km, erreur médiane ~102 km

2. **ResNet18**:
   * Diminution plus régulière de l'erreur
   * Meilleure performance à l'époque 15
   * Erreur moyenne finale ~121 km, erreur médiane ~109 km

3. **EfficientB0**:
   * Comportement plus instable avec des oscillations importantes
   * Pic d'erreur vers l'époque 10
   * Performance globale inférieure aux modèles ResNet

4. **DenseNet121**:
   * Courbe d'apprentissage similaire à EfficientB0 mais avec des performances légèrement meilleures
   * Erreurs plus élevées que les modèles ResNet

Conclusion
~~~~~~~~~~~~~

1. **Meilleur modèle global**: ResNet50 présente globalement les meilleures performances avec les erreurs de distance les plus faibles et un bon R².

2. **Efficacité des architectures**: Les architectures ResNet semblent mieux adaptées à cette tâche de prédiction géographique que EfficientB0 ou DenseNet121.

3. **Compromise complexité-performance**: ResNet18, malgré sa taille plus réduite que ResNet50, offre des performances très proches, ce qui pourrait en faire un choix optimal si les ressources de calcul sont limitées.

Recommandations
~~~~~~~~~~~~~

Pour cette tâche spécifique, ResNet50 serait le modèle recommandé en priorité, suivi de près par ResNet18 qui offre un bon compromis entre complexité et performance.

.. container:: Architecture_resnet50
   
   .. image:: _static/RSN50.jpg
      :alt: Architecture de resnet50
      :width: 85%
      :align: center

1. Zero Padding : Ajoute des zéros autour de l'image d'entrée pour conserver les dimensions après la convolution.
2. Convolution (CONV) : Opération linéaire qui applique des filtres sur l'entrée pour extraire des caractéristiques.
3. Batch Normalization (Batch Norm) : Normalise les activations d'une couche précédente afin de stabiliser et d'accélérer l'apprentissage.
4. ReLU (Rectified Linear Unit) : Fonction d'activation non linéaire utilisée pour introduire de la non-linéarité dans le réseau.
5. Max Pooling (Max Pool) : Réduit les dimensions de l'image en sélectionnant la valeur maximale dans chaque région de la carte d'activation.
6. Convolutional Block (Conv Block) : Bloc résiduel contenant des couches de convolution avec une connexion raccourcie (skip connection) pour apprendre les résidus.
7. Identity Block (ID Block) : Bloc sans modification de dimension qui apprend à ajuster l'identité de l'entrée.
8. Average Pooling (Avg Pool) : Moyenne les valeurs dans une région spécifique pour réduire la taille des cartes d'activation.
9. Flattening : Transforme la sortie 2D des cartes d'activation en un vecteur 1D pour la couche entièrement connectée (FC).
10. Fully Connected Layer (FC) : Couche dense qui effectue la classification finale après la phase de convolution et de pooling.

.. indication::  
   Le modèle resnet50 (ainsi que tous les autres modèles candidats avec leurs versions) est déjà disponible dans Torch.
   Inutile de chercher ailleurs, il suffit d'avoir Python 3.8 ou une version plus récente. Vous pouvez l'importer directement depuis **torchvision.models**

Pour plus de détails sur la démarche et les calculs, consultez notre **notebook Colab** :

.. attention::
   Veuillez remplacer les chemins des fichiers existants dans les cellules du notebook par ceux correspondant à votre environnement.

.. raw:: html

   <a href="https://colab.research.google.com/drive/19aClr0Klj6ZdcKfQF3c6RGXOZfhqCptF#scrollTo=Mk6e2ud77eXI" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

.. note:: 
   En raison de la taille importante des données, nous avons créé un ensemble de données réduit contenant 200 images. Cela vous permet de tester les modèles vous-même, sans avoir à passer par toutes les étapes de préparation des données. Pour l’utiliser, suivez les étapes ci-dessous :

   1. Téléchargez les données réduites depuis ce lien : https://drive.google.com/drive/folders/16s7_JoakN-84FbYH6WjBfOHoRJf_tqC5?usp=drive_link

   2. Placez le fichier téléchargé dans votre Google Drive.

   3. Adaptez les chemins d'accès dans le notebook en remplaçant les chemins existants par ceux correspondant à votre environnement.

   4. Exécutez les fonctions principales du notebook.

   5. Testez le modèle et visualisez quelques prédictions.




.. toctree::
   :maxdepth: 3
   :caption: Contenu :

   .. prev:: Données