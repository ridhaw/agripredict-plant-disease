# AgriPredict — détection de maladies des plantes par deep learning

Classification d'images de feuilles en **4 classes** pour aider au diagnostic agricole :

| Classe | Description |
|---|---|
| `carence_azote` | carence en azote |
| `mildiou` | mildiou |
| `oïdium` | oïdium |
| `sain` | plante saine |

## Résultats

| Modèle | Accuracy (test, 226 images) |
|---|---|
| CNN entraîné de zéro (baseline) | 26,5 % |
| **MobileNetV2 fine-tuné (transfer learning)** | **87,6 %** |

Le CNN baseline atteint ~97 % sur l'entraînement mais s'effondre en test : il sur-apprend.
Le transfer learning depuis ImageNet, avec dégel progressif des couches et poids de
classe, règle ce problème.

## Démarche

| Notebook | Étape |
|---|---|
| [`1_eda.ipynb`](notebooks/1_eda.ipynb) | Exploration : distribution des classes et des splits, tailles, formats et qualité des images |
| [`2_preprocessing_cnn_baseline.ipynb`](notebooks/2_preprocessing_cnn_baseline.ipynb) | Redimensionnement 224×224, normalisation, augmentation, CNN baseline |
| [`3_transfer_learning_mobilenetv2.ipynb`](notebooks/3_transfer_learning_mobilenetv2.ipynb) | MobileNetV2 pré-entraîné, fine-tuning, callbacks, recherche d'hyperparamètres (Optuna), évaluation et export du modèle `.keras` |

**Données** : 1 500 images réparties en train (1 050) / validation (224) / test (226),
de 300 à 400 images par classe.

## Stack

Python · TensorFlow / Keras · MobileNetV2 · scikit-learn · Optuna · OpenCV · pandas ·
matplotlib · seaborn · plotly

## Lancer

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Le dataset n'est pas inclus dans le dépôt. Les notebooks l'attendent dans `data/`, à la
racine du projet (`BASE_DIR = Path("..")` depuis `notebooks/`) ; les modèles entraînés
sont exportés dans `backend/models/`.
