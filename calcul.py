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


# --- HDR -> SDR, en calcul pur : identique sur Windows, macOS, Linux ------------------------------------------------
# ⚠️ Né le 28/09/2026 : le ffmpeg de Homebrew (macOS) N'A PAS le filtre zscale (mesuré en CI : « No such filter »),
# alors que l'iPhone filme en HDR par défaut. Ne JAMAIS revenir à un filtre ffmpeg dont la présence dépend du système.
# Constantes : SMPTE ST 2084 (PQ), ARIB STD-B67 / BT.2100 (HLG), BT.2087 (matrice BT.2020 -> BT.709), BT.709 (OETF),
# courbe filmique de John Hable (celle du filtre tonemap=hable de ffmpeg).
_PQ = dict(m1=2610 / 16384, m2=2523 / 4096 * 128, c1=3424 / 4096, c2=2413 / 4096 * 32, c3=2392 / 4096 * 32)
_HLG = dict(a=0.17883277, b=0.28466892, c=0.55991073)
_BT2020_VERS_709 = np.array([[1.6605, -0.5876, -0.0728],
                             [-0.1246, 1.1329, -0.0083],
                             [-0.0182, -0.1006, 1.1187]], np.float32)
BLANC_SDR_NITS = 100.0  # 1,0 en sortie = 100 nits (même référence que zscale npl=100)
CRETE_HLG_NITS = 1000.0


def pq_vers_nits(v):
    """PQ (ST 2084) -> luminance absolue en nits (0 à 10 000)."""
    p = _PQ
    x = np.power(np.clip(v, 0, 1), 1 / p['m2'])
    return 10000.0 * np.power(np.maximum(x - p['c1'], 0) / (p['c2'] - p['c3'] * x), 1 / p['m1'])


def hlg_vers_nits(v):
    """HLG (BT.2100) inverse OETF -> lumière relative [0, 1], ramenée à une crête de 1000 nits."""
    h = _HLG
    v = np.clip(v, 0, 1)
    e = np.where(v <= 0.5, v * v / 3.0, (np.exp((v - h['c']) / h['a']) + h['b']) / 12.0)
    return e * CRETE_HLG_NITS


def _hable(x):
    A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F) - E / F


_LUT = {}


def _tables(transfert):
    """Tables précalculées (une fois par processus) : décodage 16 bits -> linéaire relatif, et OETF BT.709 sur 16 bits.
    Les courbes PQ/HLG/BT.709 agissent canal par canal : une table de 65 536 entrées les remplace EXACTEMENT sur des
    entrées 16 bits. Mesuré le 28/09/2026 : formule directe 322 ms/image en 1080p."""
    if transfert not in _LUT:
        codes = np.arange(65536, dtype=np.float64) / 65535
        nits = pq_vers_nits(codes) if transfert == 'smpte2084' else hlg_vers_nits(codes)
        oetf = np.where(codes < 0.018, 4.5 * codes, 1.099 * np.power(codes, 0.45) - 0.099)
        _LUT[transfert] = ((nits / BLANC_SDR_NITS).astype(np.float32), oetf.astype(np.float32))
    return _LUT[transfert]


def hdr16_vers_sdr(u16, transfert):
    """Version rapide de hdr_vers_sdr pour une image uint16 (h, w, 3) telle que décodée par ffmpeg (rgb48le).
    Même calcul, tables à la place des courbes ; le test test_hdr_rapide_egal_formule borne l'écart à 1/255."""
    decode, oetf = _tables(transfert)
    lin = cv2.transform(decode[u16], _BT2020_VERS_709)
    np.maximum(lin, 0, out=lin)
    crete = CRETE_HLG_NITS / BLANC_SDR_NITS
    sig = lin.max(axis=2)
    gain = np.divide(_hable(sig) / _hable(crete), sig, out=np.zeros_like(sig), where=sig > 1e-6)
    lin *= gain[..., None]
    np.clip(lin, 0, 1, out=lin)
    return oetf[(lin * 65535 + 0.5).astype(np.uint16)]


def hdr_vers_sdr(rgb, transfert):
    """rgb float32 (h, w, 3) dans [0, 1], codé PQ ('smpte2084') ou HLG ('arib-std-b67'), primaires BT.2020
    -> rgb float32 SDR BT.709 dans [0, 1]. Compression Hable appliquée au canal le plus fort (préserve la teinte)."""
    nits = pq_vers_nits(rgb) if transfert == 'smpte2084' else hlg_vers_nits(rgb)
    lin = (nits / BLANC_SDR_NITS).astype(np.float32) @ _BT2020_VERS_709.T
    lin = np.maximum(lin, 0)
    # ⚠️ Crête FIXE (1000 nits, mastering standard des téléphones), JAMAIS le max de l'image : une crête calculée par
    # image ferait « pomper » la luminosité d'une image à l'autre.
    crete = CRETE_HLG_NITS / BLANC_SDR_NITS
    sig = lin.max(axis=2, keepdims=True)
    mappe = _hable(sig) / _hable(crete)
    lin = lin * np.where(sig > 1e-6, mappe / np.maximum(sig, 1e-6), 0)
    lin = np.clip(lin, 0, 1)
    return np.where(lin < 0.018, 4.5 * lin, 1.099 * np.power(lin, 0.45) - 0.099).astype(np.float32)


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
