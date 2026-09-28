"""Détourage vidéo « fond vert sans fond vert » : RVM (contour + mémoire temporelle) puis sigmoïde normalisée sur la
seule bande incertaine, incrustation sur un fond, son d'origine conservé. Windows, macOS, Linux (voir moteur.py).

Usage : python detourer.py <video> <fond> <sortie.mp4> [--sequentiel]
  <fond>        image (jpg/png, recadrée pour couvrir) ou couleur hexadécimale (ex. 00b140, vert d'incrustation).
  --sequentiel  désactive le parallélisme (sert au test d'équivalence : le résultat DOIT être identique au pixel).

⚠️ SYSTÈME OFFICIEL = RVM + sigmoïde, choisi par l'opérateur le 28/09/2026 sur un rush bien éclairé. Le mode --fin
(ViTMatte-S) a été RETIRÉ ce jour-là sur son mandat (mèches jugées « trop passées », 2,3× plus lent) ; son dernier
état est archivé sous l'étiquette git `archive-mode-fin-vitmatte` (voir recherche/MODE-FIN.md). Ne pas le réintroduire
sans nouvelle décision de l'opérateur.
⚠️ Réglages MESURÉS le 28/09/2026 (RX 7600) : RVM downsample 0,25 (0,375 et 0,5 mesurés pires).
⚠️ JAMAIS `-shortest` à l'encodage : mesuré le 28/09/2026 sur un rush de téléphone, 137 images écrites mais 133 dans
le fichier (4,46 s d'image pour 4,57 s de son) — l'option coupe les dernières images quand le son finit un peu avant.
Sans elle : 137/137, 4,567 s d'image pour 4,565 s de son. (Le premier diagnostic, « cadence variable », était FAUX :
un test l'a réfuté en supprimant la conversion de cadence, sans effet sur la synchro.)
Cadence : on décode quand même en cadence FIXE entière (filtre fps, ex. 29,73 -> 30) et on encode à cette cadence,
pour un fichier de sortie à cadence standard et déterministe.
⚠️ Parallélisme = 3 étages (lecture | carte graphique | incrustation + écriture) reliés par des files BORNÉES
(PROFONDEUR images max en mémoire). RVM reste strictement séquentiel : son état récurrent passe d'une image à l'autre.
Licence : RVM GPL-3.0 (usage interne libre).
"""
import os, sys, json, time, queue, threading, subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import numpy as np, cv2
import moteur, calcul

ICI = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.join(ICI, 'modeles')
PROFONDEUR = 8
HDR_COEURS = max(1, min(4, (os.cpu_count() or 2) - 2))  # conversion HDR : laisse 2 cœurs à ffmpeg et à l'incrustation
HDR_AVANCE = 2 * HDR_COEURS  # images HDR en cours de conversion, bornées
SORTIES_RVM = ['pha', 'r1o', 'r2o', 'r3o', 'r4o']  # 'fgr' (couleur plein format, ~25 Mo/image) JAMAIS rapatrié : inutilisé
HDR = {'smpte2084', 'arib-std-b67'}  # PQ (HDR10/HDR10+/Dolby Vision) et HLG : convertis, jamais refusés
FIN = object()


def sonder(video):
    """-> (largeur, hauteur, cadence fixe de sortie, transfert HDR ou None)."""
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height,avg_frame_rate,r_frame_rate,color_transfer:stream_side_data=rotation',
                          '-of', 'json', video], capture_output=True, text=True, check=True).stdout
    st = json.loads(out)['streams'][0]
    w, h = st['width'], st['height']
    rot = next((abs(int(d.get('rotation', 0))) for d in st.get('side_data_list', []) if 'rotation' in d), 0)
    if rot in (90, 270):
        w, h = h, w  # ffmpeg applique la rotation au décodage
    t = st.get('color_transfer')
    return w, h, cadence(st.get('avg_frame_rate'), st.get('r_frame_rate')), (t if t in HDR else None)


def decodeur(video, w, h, fps, transfert):
    """Lance ffmpeg en décodage à cadence fixe et rend (processus, lire) ; lire() -> image float32 SDR ou None.
    Zéro friction exigée par l'opérateur (28/09/2026) : une vidéo HDR n'est PAS refusée. Elle est décodée en 16 bits
    (rgb48) puis convertie en SDR par calcul.hdr_vers_sdr, en Python pur — JAMAIS par un filtre ffmpeg (zscale absent
    du ffmpeg de Homebrew, mesuré en CI macOS). La conversion tourne dans l'étage lecture (parallèle)."""
    fmt, octets = ('rgb48le', 6) if transfert else ('rgb24', 3)
    dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', video, '-vf', f'fps={fps}', '-f', 'rawvideo',
                            '-pix_fmt', fmt, '-'], stdout=subprocess.PIPE)
    taille = w * h * octets

    def brut():
        buf = dec.stdout.read(taille)
        return None if len(buf) < taille else buf

    if not transfert:
        def lire():
            buf = brut()
            return None if buf is None else np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32) / 255
        return dec, lire

    # HDR : conversion (126 ms/image mesurés en 1080p) répartie sur plusieurs cœurs, ORDRE CONSERVÉ (file de futurs
    # dépilée dans l'ordre d'arrivée), profondeur BORNÉE à HDR_AVANCE images en mémoire.
    pool = ThreadPoolExecutor(HDR_COEURS)
    attente = deque()

    def convertir(buf):
        return calcul.hdr16_vers_sdr(np.frombuffer(buf, '<u2').reshape(h, w, 3), transfert)

    def lire():
        while len(attente) < HDR_AVANCE and (buf := brut()) is not None:
            attente.append(pool.submit(convertir, buf))
        if not attente:
            pool.shutdown()
            return None
        return attente.popleft().result()
    return dec, lire


def cadence(moyenne, nominale):
    """Cadence FIXE de sortie : la cadence moyenne réelle arrondie à l'image entière (téléphone ~29,83 -> 30).
    Repli sur la cadence nominale si la moyenne est absente ou nulle ; 30 en dernier recours."""
    for txt in (moyenne, nominale):
        try:
            f = Fraction(txt)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if f > 0:
            return max(1, round(float(f)))
    return 30


def fond(spec, w, h):
    if os.path.isfile(spec):
        im = cv2.imread(spec)
        s = max(w / im.shape[1], h / im.shape[0])
        im = cv2.resize(im, (int(im.shape[1] * s + 0.5), int(im.shape[0] * s + 0.5)), interpolation=cv2.INTER_AREA)
        y0, x0 = (im.shape[0] - h) // 2, (im.shape[1] - w) // 2
        return cv2.cvtColor(im[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    c = spec.lstrip('#')
    return np.full((h, w, 3), [int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


class Detoureur:
    """Étage carte graphique : RVM seul (état récurrent)."""

    def __init__(self):
        self.rvm = moteur.ouvrir(os.path.join(MOD, 'rvm_resnet50_fp32.onnx'))
        self.rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
        self.dr = np.array([0.25], np.float32)

    def __call__(self, img, chw):
        """chw = copie contiguë (1, 3, h, w) de l'image -> alpha RVM (h, w). La bande et la sigmoïde partent dans
        l'étage processeur (finir), pour que l'étage carte graphique ne fasse que du calcul carte graphique."""
        pha, *self.rec = self.rvm.run(SORTIES_RVM, {'src': chw, 'r1i': self.rec[0], 'r2i': self.rec[1],
                                                   'r3i': self.rec[2], 'r4i': self.rec[3], 'downsample_ratio': self.dr})
        return pha[0, 0]


def finir(img, a, bg):
    """Étage processeur : sigmoïde sur la bande incertaine puis incrustation -> octets RGB24."""
    return calcul.incruster_octets(img, calcul.sigmoide(a, calcul.bande(a)).clip(0, 1), bg)


def preparer(img):
    """Copie contiguë (1, 3, h, w) attendue par RVM, faite dans l'étage lecture (hors étage carte graphique)."""
    return img, np.ascontiguousarray(img.transpose(2, 0, 1)[None])


def traiter(lire, ecrire, det, bg, parallele=True):
    """lire() -> image float32 ou None ; ecrire(octets). Retourne le nombre d'images.
    Séquentiel et parallèle DOIVENT produire les mêmes octets (test_detourer.py).
    ⚠️ En cas d'erreur d'un étage, les files sont VIDÉES jusqu'à FIN (jamais abandonnées) : un étage bloqué sur une
    file pleine empêcherait join() de rendre la main — l'outil pendrait au lieu de signaler l'erreur."""
    if not parallele:
        n = 0
        while (img := lire()) is not None:
            ecrire(finir(img, det(*preparer(img)), bg)); n += 1
        return n
    q_in, q_out = queue.Queue(PROFONDEUR), queue.Queue(PROFONDEUR)
    erreurs = []

    def lecteur():
        try:
            while (img := lire()) is not None:
                q_in.put(preparer(img))
        except BaseException as e:
            erreurs.append(e)
        finally:
            q_in.put(FIN)

    def ecrivain():
        try:
            while (x := q_out.get()) is not FIN:
                ecrire(finir(*x, bg))
        except BaseException as e:
            erreurs.append(e)
            while q_out.get() is not FIN:
                pass  # vide la file pour ne pas bloquer l'étage GPU

    tl, te = threading.Thread(target=lecteur, daemon=True), threading.Thread(target=ecrivain, daemon=True)
    tl.start(); te.start()
    n, fin_lue = 0, False
    try:
        while (p := q_in.get()) is not FIN:
            if erreurs:  # l'écrivain a échoué : inutile de calculer le reste
                break
            q_out.put((p[0], det(*p))); n += 1
        else:
            fin_lue = True
    finally:
        q_out.put(FIN)
        if not fin_lue:  # sortie anticipée (erreur ici ou chez l'écrivain) : libérer le lecteur
            while q_in.get() is not FIN:
                pass
        te.join(); tl.join()
    if erreurs:
        raise erreurs[0]
    return n


def main():
    drapeaux = {x for x in sys.argv[1:] if x.startswith('--')}
    args = [x for x in sys.argv[1:] if not x.startswith('--')]
    if len(args) != 3 or drapeaux - {'--sequentiel'}:
        sys.exit(__doc__)
    video, fspec, sortie = args
    w, h, fps, transfert = sonder(video)
    det = Detoureur()
    bg = fond(fspec, w, h)
    # ⚠️ Écriture sous un nom TEMPORAIRE puis renommage : une vidéo interrompue ne doit jamais avoir l'air finie.
    partiel = os.path.splitext(sortie)[0] + '.partiel.mp4'
    dec, lire = decodeur(video, w, h, fps, transfert)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                            '-r', str(fps), '-i', '-', '-i', video, '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264',
                            '-crf', '16', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-c:a', 'copy', partiel],
                           stdin=subprocess.PIPE)
    t0 = time.time()
    n = traiter(lire, enc.stdin.write, det, bg, parallele='--sequentiel' not in drapeaux)
    enc.stdin.close(); enc.wait(); dec.wait()
    if enc.returncode or dec.returncode:
        sys.exit(f'ÉCHEC ffmpeg (décodage {dec.returncode}, encodage {enc.returncode}) ; sortie partielle : {partiel}')
    os.replace(partiel, sortie)
    tt = time.time() - t0
    print(f'FINI : {n} images à {fps} i/s{" (HDR converti)" if transfert else ""} en {tt:.1f} s ({n / tt:.1f} images/s) -> {sortie}')


if __name__ == '__main__':
    main()
