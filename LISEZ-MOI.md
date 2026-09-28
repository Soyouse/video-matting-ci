# Détourage vidéo — fond vert sans fond vert

Tu donnes une vidéo où tu parles face caméra et un fond, tu récupères la vidéo détourée sur ce fond, avec le son.

```
python detourer.py "ma_video.mp4" "mon_fond.jpg" "sortie.mp4"
python detourer.py "ma_video.mp4" 00b140 "sortie_vert.mp4"     # fond vert uni, pour finir le montage ailleurs
```

Règle de tournage : **rien de collé à ta silhouette** (casquette accrochée derrière la tête, dossier de chaise) — le
détourage le prendrait pour toi. Et un bon éclairage de face : c'est lui qui fait la qualité des bords.

Zéro friction, l'outil s'occupe de tout : vidéo HDR du téléphone convertie automatiquement, cadence variable des
téléphones ramenée à une cadence fixe, aucune image perdue en fin de vidéo, son conservé, et une vidéo interrompue
reste en `.partiel.mp4` (jamais un fichier à moitié écrit qui a l'air fini).

## Installation (une fois)

Il faut `ffmpeg` dans le PATH, Python 3.11+, puis dans ce dossier :

| Système | Commande |
|---|---|
| Windows (toute carte DirectX 12) | `python -m venv venv && venv\Scripts\pip install -r requirements-windows.txt` |
| macOS (Apple Silicon) | `python3 -m venv venv && venv/bin/pip install -r requirements-macos.txt` |
| Linux + carte Nvidia | `python3 -m venv venv && venv/bin/pip install -r requirements-linux-nvidia.txt` |
| Linux sans Nvidia | non pris en charge sur carte graphique (`requirements-linux.txt` sert à la CI, sur processeur) |

Modèles, dans `modeles/` (non versionnés, trop gros) :
- `rvm_resnet50_fp32.onnx` : https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx
- (mode « fin » ViTMatte retiré le 28/09/2026, archivé : voir `recherche/MODE-FIN.md`)

## État de preuve (28/09/2026)

| Système | Statut |
|---|---|
| Windows 11 + AMD RX 7600 (DirectML) | **PROUVÉ** : 18,4 images/s de bout en bout (plafond de RVM sur cette carte), 14 tests verts, contour identique à la référence |
| macOS Apple Silicon (CoreML) | **PROUVÉ en CI GitHub** (run 36377301755) : 11 tests verts, contour IDENTIQUE à la référence processeur (écart 0,000000), vidéo détourée de bout en bout. Vitesse sur un vrai Mac : **inconnue** (la machine GitHub est virtuelle) |
| Linux + Nvidia (CUDA) | **inconnu** : jamais exécuté sur une carte Nvidia |
| Linux sans Nvidia | **non pris en charge** (refus nommé). WebGPU a été essayé : contour FAUX à 63 %, exclu. Le code lui-même tourne (CI, sur processeur) |
| Windows sur machine GitHub (sans carte graphique) | code prouvé sur processeur en CI (mode réservé à la CI) |

CI (même commande en local et sur GitHub, Mac + Linux + Windows) : `python ci.py`. Tests seuls : `python -m unittest test_detourer` (avec `DETOURAGE_RUSH=<.mp4>` pour les tests sur vrai rush).

## Comment ça marche

1. **RVM** (Robust Video Matting, GPL-3.0) trouve ta silhouette, avec une mémoire d'une image à l'autre.
2. Sur la **bande incertaine** du contour seulement, une sigmoïde normalisée durcit la transparence (main qui bouge
   opaque).
3. Incrustation : `pixel = image × alpha + fond × (1 − alpha)`.
Lecture, calcul sur la carte graphique et écriture tournent en parallèle ; le résultat est identique au pixel près
à la version séquentielle (`--sequentiel`), garanti par test.
