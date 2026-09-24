# Robustesse adversariale d'un NIDS-DNN sur NSL-KDD

Code source complet de l'expérimentation présentée en section 6 de l'article
"Robustesse et fiabilité des modèles d'IA pour la détection d'attaques
face aux perturbations adversariales : un survey sur les NIDS-LLM".

## Reproductibilité

Protocole complet répété sur **5 graines aléatoires indépendantes**
(42, 123, 2024, 7, 999) et **5 valeurs d'amplitude de perturbation ε**
(0.01, 0.05, 0.10, 0.15, 0.20), soit 30 entraînements complets
(5 baseline + 25 défendu, un par combinaison seed×epsilon).

## Hyperparamètres (documentés explicitement)

| Paramètre | Valeur |
|---|---|
| Architecture | DNN 41 → 64 (ReLU) → 32 (ReLU) → 2 |
| Optimiseur | Adam, learning rate = 1e-3 |
| Epochs | 15 |
| Batch size | 512 |
| Attaque FGSM | 1 étape, x_adv = x + ε·sign(∇x L) |
| Attaque PGD | 10 itérations, α = (2/15)·ε (donc α=0,02 pour ε=0,15), projection/clipping L∞ dans la boule de rayon ε, pas de random start |
| Variables gelées pendant l'attaque | protocol_type, service, flag (indices 1, 2, 3) |
| Fonction de perte | Entropie croisée |
| Graines testées | [42, 123, 2024, 7, 999] |
| ε testés | [0.01, 0.05, 0.10, 0.15, 0.20] |
| Fit du scaler | StandardScaler ajusté sur train uniquement, appliqué (transform) sur test |
| Encodage catégoriel | OrdinalEncoder, ajusté sur train uniquement ; catégories inconnues du test → code -1 |

## Dataset

NSL-KDD, split train/test natif, sans re-échantillonnage. Les fichiers
`KDDTrain+.txt` et `KDDTest+.txt` sont inclus directement dans `data/`
pour garantir la reproductibilité sans dépendance à un dépôt externe.
Source originale : https://github.com/defcom17/NSL_KDD (licence de
redistribution libre).

## Environnement

Voir `requirements.txt`. Testé avec Python 3.11-3.12, PyTorch 2.14 (CPU).

## Exécution

```bash
pip install -r requirements.txt
python code/run_sensitivity.py
```

## Résultats bruts

Le fichier `code/results_sensitivity.json` contient les résultats agrégés
(moyenne ± écart-type sur les cinq graines) utilisés pour le tableau 4 et
les figures de sensibilité. Le fichier `code/results_sensitivity_raw.json`
contient les observations individuelles par graine, epsilon, configuration
et attaque, générées à chaque exécution.

## Limites connues

Ces attaques sont générées dans l'espace des 38 caractéristiques
continues (les 3 variables catégorielles étant gelées) après extraction
et normalisation ; elles ne garantissent pas la réalisabilité complète
d'un flux réseau physiquement transmissible (voir FENCE, Chernikova &
Oprea 2019/2022 ; PANTS, Jin et al. 2025 ; DeepRed, Hajizadeh et al. 2025).

## Prétraitement et absence de fuite de données

Les encodeurs catégoriels sont ajustés uniquement sur le jeu d'entraînement.
Le jeu de test est ensuite transformé avec ces mêmes paramètres ; aucune
information de test n'est utilisée pour ajuster l'encodage. Le
`StandardScaler` est lui aussi ajusté uniquement sur le train puis appliqué
au test par `transform`.
