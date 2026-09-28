"""Calculs purs du détourage (aucune entrée/sortie, aucun savoir sur le système) : testables, identiques sur tout OS.

Toutes les images sont en float32 dans [0, 1], format (hauteur, largeur, canaux) ; alpha en (hauteur, largeur).
"""
import numpy as np
import cv2

SEUIL_BAS, SEUIL_HAUT = 13 / 255, 242 / 255  # mesurés le 28/09/2026 : bande incertaine = alpha RVM entre les deux
EROSION = 15  # px : largeur ajoutée de chaque côté de la bande
PENTE = 12.0  # sigmoïde choisie par l'opérateur le 28/09/2026 (main opaque, ROI jugé supérieur à ViTMatte)


def bande(alpha):
    """Masque booléen de la zone incertaine : tout ce qui n'est pas « sûr sujet » ni « sûr fond » après érosion."""
    k = np.ones((EROSION, EROSION), np.uint8)
    sur_sujet = cv2.erode((alpha > SEUIL_HAUT).astype(np.uint8), k) > 0
    sur_fond = cv2.erode((alpha < SEUIL_BAS).astype(np.uint8), k) > 0
    return ~(sur_sujet | sur_fond)


def sigmoide(alpha, masque):
    """Durcit la transparence DANS la bande seulement ; hors bande, alpha RVM intact. Ne regarde pas l'image :
    elle durcit aussi les erreurs de RVM (un objet collé à la silhouette devient un bord net)."""
    out = alpha.copy()
    out[masque] = _sigmoide_normalisee(alpha[masque])
    return out


def _sigmoide_normalisee(x):
    """Sigmoïde recalée pour que 0 → 0 et 1 → 1 EXACTEMENT (la brute rend 0,0025 et 0,9975 : un fond pur garderait
    0,25 % de la pièce d'origine). Même forme au milieu, et |s(x) − 0,5| ≥ |x − 0,5| partout : elle durcit, jamais
    ne ramollit (vérifié par test)."""
    s = lambda v: 1.0 / (1.0 + np.exp(-PENTE * (v - 0.5)))
    lo, hi = s(0.0), s(1.0)
    return ((s(x) - lo) / (hi - lo)).astype(np.float32)


def incruster(image, alpha, fond):
    """Pixel final = image d'origine × alpha + nouveau fond × (1 − alpha).
    ⚠️ Nettoyage de couleur des bords (Forte & Pitié 2021, fusion floue) ESSAYÉ le 28/09/2026 et REJETÉ : invisible
    sur un rush face caméra même sur fond blanc, et 380 ms/image CPU (6× le détourage). Ne pas le réintroduire sans
    un rush où le liseré se VOIT."""
    a = alpha[..., None]
    return image * a + fond * (1 - a)


def incruster_octets(image, alpha, fond):
    """Même calcul que incruster(), rendu en octets RGB24 : fond + alpha × (image − fond), via OpenCV (multi-cœur).
    Mesuré le 28/09/2026 sur 1080×1920 : 20 ms contre 60 ms en numpy, résultat IDENTIQUE au niveau près (0 pixel
    différent) — le test test_incruster_octets_identique le garde."""
    d = cv2.multiply(cv2.subtract(image, fond), cv2.merge([alpha, alpha, alpha]))
    return cv2.convertScaleAbs(cv2.add(d, fond), alpha=255.0).tobytes()
