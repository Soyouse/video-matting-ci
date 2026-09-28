# Détourage vidéo — fond vert sans fond vert

Tu donnes une vidéo où tu parles face caméra et un fond, tu récupères la vidéo détourée sur ce fond, avec le son.

```
python detourer.py "ma_video.mp4" "mon_fond.jpg" "sortie.mp4"
python detourer.py "ma_video.mp4" 00b140 "sortie_vert.mp4"     # fond vert uni, pour finir le montage ailleurs
python detourer.py "ma_video.mp4" "mon_fond.jpg" "sortie.mp4" --fin   # bords plus fins, 3 à 4 fois plus lent
```

Règle de tournage : **rien de collé à ta silhouette** (casquette accrochée derrière la tête, dossier de chaise) — le
détourage le prendrait pour toi.

## Installation (une fois)

Il faut `ffmpeg` dans le PATH, Python 3.11+, puis dans ce dossier :

| Système | Commande |
|---|---|
| Windows (toute carte DirectX 12) | `python -m venv venv && venv\Scripts\pip install -r requirements-windows.txt` |
| macOS (Apple Silicon) | `python3 -m venv venv && venv/bin/pip install -r requirements-macos.txt` |
| Linux + carte Nvidia | `python3 -m venv venv && venv/bin/pip install -r requirements-linux-nvidia.txt` |
| Linux, autre carte (Vulkan) | `python3 -m venv venv && venv/bin/pip install -r requirements-linux.txt` |

Modèles, dans `modeles/` (non versionnés, trop gros) :
- `rvm_resnet50_fp32.onnx` : https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx
- `vitmatte_s_960x544.onnx` et `vitmatte_s_544x960.onnx` (seulement pour `--fin`) : fabriqués une fois par
  `python outils_export_vitmatte.py 960 544` puis `... 544 960` (demande `torch` et `transformers`, usine uniquement).

## État de preuve (28/09/2026)

| Système | Statut |
|---|---|
| Windows 11 + AMD RX 7600 (DirectML) | **PROUVÉ** : 18,5 images/s de bout en bout sur un rush 1080×1920 (plafond de RVM sur cette carte), 10 tests verts |
| macOS (CoreML) | **inconnu** : jamais exécuté |
| Linux (CUDA ou WebGPU/Vulkan) | **inconnu** : jamais exécuté |

CI (même commande en local et sur GitHub, Mac + Linux + Windows) : `python ci.py`. Tests seuls : `python -m unittest test_detourer` (avec `DETOURAGE_RUSH=<.mp4>` pour les tests sur vrai rush).

## Comment ça marche

1. **RVM** (Robust Video Matting, GPL-3.0) trouve ta silhouette, avec une mémoire d'une image à l'autre.
2. Sur la **bande incertaine** du contour seulement, une sigmoïde normalisée durcit la transparence (main qui bouge
   opaque). Avec `--fin`, c'est **ViTMatte-S** (MIT) qui recalcule cette bande en regardant l'image.
3. Incrustation : `pixel = image × alpha + fond × (1 − alpha)`.
Lecture, calcul sur la carte graphique et écriture tournent en parallèle ; le résultat est identique au pixel près
à la version séquentielle (`--sequentiel`), garanti par test.
