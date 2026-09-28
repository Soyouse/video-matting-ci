# Mode « fin » (RVM + ViTMatte-S) — RETIRÉ, archivé pour la recherche

Retiré le 28/09/2026 sur décision de l'opérateur, après comparaison sur un rush bien éclairé (LED de face) :
- résultat jugé moins bon que RVM + sigmoïde : mèches « trop passées » ;
- 2,3× plus lent (6,4 contre 15 images/s sur la même vidéo, RX 7600).

## Retrouver le code

- Dernier état complet : étiquette git **`archive-mode-fin-vitmatte`** (`git show archive-mode-fin-vitmatte:detourer.py`).
- Fabrication des modèles ONNX : `recherche/export_vitmatte.py` (usine PyTorch + transformers, une seule fois) :
  `python recherche/export_vitmatte.py 960 544` (portrait) puis `... 544 960` (paysage).
- Modèles déjà fabriqués sur le poste de l'opérateur : `modeles/archive-mode-fin/` (non versionnés).

## Ce qu'on sait du mode (mesures du 28/09/2026)

- Chaîne : trimap tirée de l'alpha RVM (sûr sujet > 242, sûr fond < 13, érodés de 15 px), ViTMatte-S en 544×960
  ne remplace que la bande incertaine, RVM plein format ailleurs.
- ViTMatte : 117 ms/image sur DirectML ; version demi-précision = plantage DirectML.
- Licence : poids Apache 2.0 / code MIT, mais entraînés sur Composition-1k (Adobe, usage recherche) = zone grise.
- Aucun remplaçant plus récent + plus rapide + aussi bon + commercial trouvé (MEMatte 2025 sans licence).
