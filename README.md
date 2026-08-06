# Conception d'une Architecture de Super-Résolution Hybride : Landsat-8 vers Sentinel-2

## À propos du projet
Ce dépôt contient le code source, les algorithmes de préparation de données et la méthodologie développés dans le cadre du travail de Master de Mohamed Bouzenad (Université de Lausanne - UNIL, Janvier 2026)[cite: 2].

L'observation de la Terre fait traditionnellement face à un compromis spatio-temporel (*spatio-temporal trade-off*)[cite: 2]. L'objectif de ce projet est de combler l'écart technologique entre les archives historiques du programme Landsat (profondeur temporelle exceptionnelle mais résolution spatiale native de 30 mètres) et la constellation Sentinel-2 (résolution fine de 10 mètres mais historique récent)[cite: 2]. 

En s'appuyant sur l'apprentissage profond (*Deep Learning*), ce pipeline explore la Super-Résolution d'Image Unique (SISR) appliquée à la télédétection multispectrale (Bandes Bleu, Vert, Rouge, et Proche Infrarouge)[cite: 2, 4]. Le défi principal étudié ici est la gestion du "Compromis Perception-Distorsion" (*Perception-Distortion Tradeoff*) : comment générer des textures visuellement réalistes sans corrompre la signature radiométrique/spectrale du pixel indispensable aux calculs d'indices environnementaux comme le NDVI[cite: 2].

##  Reproduction du Dataset (Data-Centric AI)
L'intelligence artificielle générative étant dépourvue de compréhension sémantique, la performance des réseaux dépend strictement de la qualité du corpus d'entraînement (*Data-Centric AI*)[cite: 2]. Le jeu de données utilisé n'étant pas hébergé dans ce dépôt, **vous devez générer les données brutes via Google Earth Engine (GEE)**.

1. **Extraction GEE (`CodeGEE.txt`)** : À exécuter dans le *Code Editor* de Google Earth Engine. L'algorithme cible 56 points géographiques répartis mondialement et sélectionnés via un échantillonnage stratifié (forêts, zones urbaines selon les *Local Climate Zones*, milieux arides, deltas, etc.)[cite: 2, 7].
2. **Filtrage spatio-temporel** : Le script apparie les scènes Landsat-8 (L1TP/TOA) et Sentinel-2 (L1C) avec un écart d'acquisition maximal de 10 jours, tout en appliquant des masques nuageux stricts (ex: QA60 et MSK_CLOUD pour Sentinel-2)[cite: 7].
3. **Exportation** : Les intersections géographiques communes sont exportées sur Google Drive avec une projection dynamique UTM[cite: 7].

---

## Pipeline de Traitement Python

Le flux de travail local est divisé en 4 phases distinctes et 8 scripts.

### Phase 1 : Curation et Préparation des Données
*   **`1_patch_extracteur.py`** : Script d'extraction et d'alignement sub-pixel. Les différences de géométrie entre les capteurs sont corrigées par une estimation du décalage (corrélation de phase croisée sur des gradients de Sobel)[cite: 2, 3]. Le script applique un filtrage qualitatif strict en rejetant les tuiles dépassant un seuil de pixels *NoData* et valide la cohérence via les métriques de similarité ZNCC (*Zero-Normalized Cross-Correlation*) et SSIM (*Structural Similarity*)[cite: 2, 3]. Les patchs valides sont divisés en un jeu d'entraînement (80%) et de validation (20%)[cite: 1, 3].
*   **`2_prepare_data.py`** : Les tuiles GeoTIFF sont converties en tableaux Numpy (`.npy`) multi-canaux (RGB + NIR) via des processus parallèles (multiprocessing)[cite: 4]. Cette étape est indispensable pour lever les goulots d'étranglement (I/O bottlenecks) durant l'entraînement GPU[cite: 4].

### Phase 2 : Baseline Mathématique (EDSR)
*   **`3_Entrainement_EDSR.py`** : Entraînement d'un modèle EDSR-Lite (*Enhanced Deep Residual Networks*). L'architecture s'affranchit des couches de *Batch Normalization* pour préserver l'intégrité des valeurs spectrales[cite: 2, 5]. L'optimisation, accélérée par la précision mixte (*AMP*), se fait exclusivement sur la perte L1, maximisant ainsi le rapport signal sur bruit (PSNR) au détriment des détails haute fréquence (résultat lissé)[cite: 2, 5].
*   **`4_Applique_EDSR.py`** : Inférence sur des scènes complètes. L'algorithme déploie une fenêtre glissante (*Sliding Window*) avec un masque de fusion de *Hanning* sur les zones de recouvrement (overlap), empêchant l'apparition de coutures (artefacts de bords) lors de la reconstruction finale du GeoTIFF[cite: 2, 6].

### Phase 3 : Baseline Perceptive (ESRGAN)
*   **`5_Entrainement_ESRGAN.py`** : Entraînement de l'architecture générative adverse. 
    *   **Warmup** : Commence par 30 époques d'échauffement sur la perte L1 pour stabiliser la géométrie et éviter une divergence précoce[cite: 2, 8].
    *   **Phase GAN** : Activation d'un discriminateur *Relativistic Average GAN (RaGAN)* et d'une perte perceptuelle calculée via les cartes de caractéristiques brutes (pré-activation) d'un réseau VGG19[cite: 2, 8]. L'astuce technique consiste à dupliquer la bande infrarouge (NIR) sur 3 canaux pour l'injecter dans le modèle VGG19 pré-entraîné sur des images classiques[cite: 8].
*   **`6_Applique_ESRGAN.py`** : Inférence du générateur ESRGAN pour produire des images aux textures ultra-réalistes, mais spectralement plus instables[cite: 9].

### Phase 4 : L'Architecture Hybride
*   **`7_Entrainement_Hybride.py`** : Cœur scientifique de l'étude. Ce script de *fine-tuning* démarre avec les poids stabilisés du modèle EDSR pré-entraîné[cite: 1, 10]. Il implémente une fonction de perte (Loss) composite novatrice pour équilibrer trois impératifs :
    1.  **L1** : Maintien de la structure spatiale globale[cite: 2, 10].
    2.  **Perte Adversariale (GAN) + VGG** : Synthèse des hautes fréquences et réalisme textural[cite: 2, 10].
    3.  **Spectral Angle Mapper (SAM) Loss** : Cette contrainte agit comme un "verrou physique"[cite: 2]. Elle pénalise toute déviation angulaire entre le vecteur spectral généré et la cible, forçant le GAN à texturer l'image sans en modifier la nature colorimétrique/radiométrique (minimisant les hallucinations)[cite: 2, 10].
*   **`8_Applique_Hybride.py`** : Inférence finale. Les résultats produits (qui affichent le meilleur score global ERGAS de 58.42 et une excellente fidélité NDVI) illustrent une réconciliation pertinente entre acuité visuelle et utilisabilité scientifique des données[cite: 2, 11].

## Résultats & Discussion
Les expériences montrent que l'approche Hybride réduit considérablement les dérives radiométriques de l'ESRGAN standard tout en évitant le flou de l'EDSR[cite: 2]. Il reste néanmoins une sensibilité inhérente aux convolutions (artefacts en damier ou *Checkerboard Artifact*) dans les milieux de très haute fréquence spatiale, comme le tissu urbain complexe[cite: 2].

## Auteur
*   **Mohamed Bouzenad** - *Master of Science in Environmental Science* - [UNIL](https://www.unil.ch/masterenvi)[cite: 2].
