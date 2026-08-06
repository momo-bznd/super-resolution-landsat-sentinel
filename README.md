# super-resolution-landsat-sentinel
# Conception d'une Architecture de Super-Résolution Hybride : Landsat-8 vers Sentinel-2

## À propos de ce projet
Ce dépôt contient le code source et la méthodologie développés dans le cadre du travail de Master de Mohamed Bouzenad, réalisé à la Faculté des géosciences et de l'environnement de l'Université de Lausanne (UNIL) en janvier 2026[cite: 2].

L'objectif de ce projet est de combler l'écart technologique entre les archives historiques Landsat-8 (résolution de 30 mètres) et les standards modernes de Sentinel-2 (résolution de 10 mètres) en explorant le potentiel des réseaux de neurones profonds pour la super-résolution (SISR)[cite: 2]. 

Le projet évalue trois architectures distinctes :
*   **Modèle EDSR** : Focalisé sur la fidélité mathématique et spectrale, indispensable aux calculs d'indices comme le NDVI[cite: 2].
*   **Modèle ESRGAN** : Orienté vers la reconstruction de textures perceptuelles et le réalisme visuel[cite: 2].
*   **Modèle Hybride (EDSR-GAN)** : Une architecture inédite tentant de concilier la justesse géométrique/spectrale de l'EDSR et les détails texturaux de l'ESRGAN grâce à un fine-tuning et une contrainte de régularisation spectrale (SAM)[cite: 2].

## Prérequis et Données
*   **Google Earth Engine (GEE)** : Le fichier `CodeGEE.txt` contient le script utilisé pour l'extraction automatisée des paires d'images Landsat-8 et Sentinel-2 parfaitement synchrones et co-enregistrées.

## Pipeline d'exécution

Le processus complet est divisé en 8 étapes, allant de la préparation des données à l'application des modèles.

### Phase 1 : Préparation des données
*   **Étape 1 : Extraction & Tri** : Ce script crée deux dossiers (`mon_super_dataset` pour l'apprentissage et `mon_super_dataset_val` pour le test) en répartissant les données selon un ratio de 80% pour l'entraînement et 20% pour la validation[cite: 1].
    ```bash
    python 1_patch_extracteur.py --landsat "D:/Donnees/Landsat8" --sentinel "D:/Donnees/Sentinel2" --export "mon_super_dataset" --visu
    ```
*   **Étape 2 : Préparation (Conversion)** : Convertit le dossier d'entraînement en fichiers `.npy` pour accélérer le traitement[cite: 1].
    ```bash
    python 2_prepare_data.py --l8_dir "mon_super_dataset/landsat8" --s2_dir "mon_super_dataset/sentinel2" --out_dir "dataset_final_npy" --patch 32 --workers 8
    ```

### Phase 2 : Modèle EDSR (Baseline spectrale)
*   **Étape 3 : Entraînement EDSR** : Entraîne le modèle sur les données `.npy` pour établir une base géométrique stable[cite: 1].
    ```bash
    python 3_Entrainement_EDSR.py --data_dir "dataset_final_npy" --out_dir "resultats_training" --epochs 50 --batch_size 16 --visu
    ```
*   **Étape 4 : Application EDSR** : Applique le modèle entraîné sur les 20% de données de validation cachées[cite: 1].
    ```bash
    python 4_Applique_EDSR.py --src_dir "mon_super_dataset_val/landsat8" --dst_dir "resultats_finaux_EDSR" --checkpoint "resultats_training/best_model.pth" --tile_size 160 --overlap 32
    ```

### Phase 3 : Modèle ESRGAN (Baseline perceptuelle)
*   **Étape 5 : Entraînement ESRGAN** : Entraînement génératif pour obtenir un meilleur résultat visuel (processus plus long)[cite: 1].
    ```bash
    python 5_Entrainement_ESRGAN.py --data_dir "dataset_final_npy" --out_dir "resultats_training_esrgan" --epochs 100 --warmup 10 --batch 16
    ```
*   **Étape 6 : Application ESRGAN** : Application sur le dossier de validation[cite: 1].
    ```bash
    python 6_Applique_ESRGAN.py --src_dir "mon_super_dataset_val/landsat8" --dst_dir "resultats_finaux_ESRGAN" --checkpoint "resultats_training_esrgan/last_G.pth"
    ```

### Phase 4 : Modèle Hybride
*   **Étape 7 : Entraînement Hybride** : Fine-tuning associant SAM et GAN. Ce processus repart du modèle EDSR (`best_model.pth`) pour gagner en stabilité et en vitesse[cite: 1].
    ```bash
    python 7_Entrainement_Hybride.py --data_dir "dataset_final_npy" --out_dir "resultats_training_hybrid" --pretrained "resultats_training/best_model.pth" --epochs 50 --batch 16 --lambda_sam 0.5
    ```
*   **Étape 8 : Application Hybride** : Évaluation finale du modèle hybride sur les données de validation[cite: 1].
    ```bash
    python 8_Applique_Hybride.py --src_dir "mon_super_dataset_val/landsat8" --dst_dir "resultats_finaux_Hybrid" --checkpoint "resultats_training_hybrid/last_G.pth"
    ```

## Auteur
*   **Mohamed Bouzenad** - *Master of Science in Environmental Science* - [Université de Lausanne (UNIL)](https://www.unil.ch/masterenvi)[cite: 2].
