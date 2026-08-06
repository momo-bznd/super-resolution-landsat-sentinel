# Super-Résolution Hybride : Landsat-8 vers Sentinel-2

## À propos de ce projet
Ce dépôt héberge le code source et la méthodologie développés dans le cadre du mémoire de Master de Mohamed Bouzenad (Université de Lausanne, 2026) intitulé : *Conception d'une Architecture de Super-Résolution Hybride : Réconciliation de la Fidélité Spectrale et de la Cohérence Perceptive*[cite: 2].

L'objectif de ce projet est d'utiliser l'apprentissage profond (Deep Learning) pour harmoniser les archives historiques Landsat-8 (30 m) vers les standards de résolution de Sentinel-2 (10 m), en gérant le compromis complexe entre la précision mathématique du signal (radiométrie) et le réalisme visuel des textures[cite: 2].

## Pipeline de Traitement et Codes

Le processus de traitement est entièrement automatisé, allant de la constitution du jeu de données (Data-Centric AI) jusqu'à l'inférence des modèles de super-résolution.

### Étape 0 : Curation des données via Google Earth Engine
*   **`CodeGEE.txt`** : Script JavaScript exécuté sur Google Earth Engine pour extraire un jeu de données mondial représentatif. Il filtre les nuages (masques QA60/MSK_CLOUD), recherche les paires d'images Landsat-8/Sentinel-2 avec un écart maximum de 10 jours, et exporte automatiquement les intersections spatiales avec une projection UTM dynamique[cite: 7].

### Phase 1 : Préparation du Dataset
*   **`1_patch_extracteur.py`** : Pipeline d'extraction et d'alignement. Ce script découpe les images brutes en sous-images, calcule un alignement sub-pixel par corrélation de phase croisée, et filtre les paires de mauvaise qualité à l'aide de masques NoData et des métriques de similarité (SSIM/ZNCC)[cite: 3]. Il divise ensuite le corpus en un jeu d'entraînement (Train) et un jeu de validation (Val)[cite: 3].
*   **`2_prepare_data.py`** : Script d'optimisation. Il convertit les paires GeoTIFF validées en tuiles (patchs) au format `.npy` pour accélérer drastiquement les temps de lecture lors de l'entraînement[cite: 4]. Il extrait spécifiquement les bandes R, G, B et NIR (Bandes 2, 3, 4, 5 pour L8 et 2, 3, 4, 8 pour S2) via des processus parallèles (multiprocessing)[cite: 4].

### Phase 2 : Modèle EDSR (Fidélité Mathématique et Spectrale)
*   **`3_Entrainement_EDSR.py`** : Entraîne le réseau de neurones convolutifs EDSR-Lite. Ce script gère l'augmentation de données (rotations/flips), utilise l'accélération *Mixed Precision* (AMP) pour les GPU récents, et génère des grilles de visualisation comparatives (Input/SR/Target) à chaque époque[cite: 5].
*   **`4_Applique_EDSR.py`** : Applique le modèle EDSR entraîné sur des images satellitaires complètes (Inférence). Pour éviter les artefacts de bords ou de discontinuité, ce script utilise une technique de "Sliding Window" avec recouvrement (overlap) pondérée par un masque de *Hanning*, et met à jour dynamiquement les métadonnées de géoréférencement (Rasterio)[cite: 6].

### Phase 3 : Modèle ESRGAN (Fidélité Perceptive)
*   **`5_Entrainement_ESRGAN.py`** : Entraîne un Generative Adversarial Network (ESRGAN). Ce script est conçu en deux étapes : une phase de "Warmup" (apprentissage basé uniquement sur la perte L1 pour stabiliser la géométrie), suivie de l'activation du Discriminateur (*Relativistic GAN*) et de la perte perceptuelle (*VGG Loss*)[cite: 8]. Astuce technique : la bande infrarouge (NIR) est dupliquée sur 3 canaux pour simuler une image compatible avec les poids ImageNet du VGG19[cite: 8].
*   **`6_Applique_ESRGAN.py`** : Script d'inférence pour le modèle génératif. Tout comme l'EDSR, il s'applique de manière tuilée (*Tiling*) sur de grandes scènes pour produire des images aux textures ultra-réalistes, reconstruites tout en préservant les métadonnées géospatiales[cite: 9].

### Phase 4 : Modèle Hybride (Le compromis)
*   **`7_Entrainement_Hybride.py`** : Le cœur de l'innovation du projet. Ce script procède à un *fine-tuning* (ré-entraînement fin) en partant des poids du modèle EDSR pré-entraîné[cite: 10]. Il utilise une fonction de perte (Loss) composite unique qui équilibre :
    *   La perte **L1** (fidélité pixel à pixel)[cite: 10].
    *   La perte **Adversariale et Perceptuelle** (pour regagner des textures fines)[cite: 10].
    *   La perte **SAM (Spectral Angle Mapper)**, qui contraint physiquement le réseau à préserver la signature colorimétrique/spectrale exacte du pixel, évitant ainsi les hallucinations classiques des GANs[cite: 10].
*   **`8_Applique_Hybride.py`** : Inférence finale permettant d'appliquer le modèle Hybride sur le jeu de données test. Le résultat délivre des cartes alliant la précision spectrale requise pour la science (calcul d'indices) à la définition visuelle propre aux modèles génératifs[cite: 11].

## Auteur
*   **Mohamed Bouzenad** - *Master of Science in Environmental Science* - [Université de Lausanne (UNIL)](https://www.unil.ch/masterenvi)[cite: 2].
