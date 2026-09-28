"""Détourage vidéo « fond vert sans fond vert » : RVM (contour + mémoire temporelle) puis retouche de la seule bande
incertaine, incrustation sur un fond, son d'origine conservé. Windows, macOS, Linux (voir moteur.py).

Usage : python detourer.py <video> <fond> <sortie.mp4> [--fin] [--sequentiel]
  <fond>        image (jpg/png, recadrée pour couvrir) ou couleur hexadécimale (ex. 00b140, vert d'incrustation).
  (par défaut)  retouche = sigmoïde (calcul.py), ~7 ms/image, choix de l'opérateur le 28/09/2026.
  --fin         retouche = ViTMatte-S (carte graphique, ~117 ms/image) : mèches plus naturelles, efface mieux les
                restes d'objets collés à la silhouette.
  --sequentiel  désactive le parallélisme (sert au test d'équivalence : le résultat DOIT être identique au pixel).

⚠️ Réglages MESURÉS le 28/09/2026 (RX 7600, rush face caméra) : RVM downsample 0,25 (0,375 et 0,5 mesurés pires).
⚠️ Parallélisme = 3 étages (lecture | carte graphique | incrustation + écriture) reliés par des files BORNÉES
(PROFONDEUR images max en mémoire : sans borne, une vidéo longue remplirait la RAM). RVM reste strictement séquentiel :
son état récurrent passe d'une image à la suivante.
Licences : RVM GPL-3.0 (usage interne libre) ; ViTMatte MIT (poids entraînés sur un jeu Adobe « recherche »).
"""
import os, sys, json, time, queue, threading, subprocess
import numpy as np, cv2
import moteur, calcul

ICI = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.join(ICI, 'modeles')
PROFONDEUR = 8
SORTIES_RVM = ['pha', 'r1o', 'r2o', 'r3o', 'r4o']  # 'fgr' (couleur plein format, ~25 Mo/image) JAMAIS rapatrié : inutilisé
FIN = object()


def dims(video):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'stream=width,height,r_frame_rate:stream_side_data=rotation', '-of', 'json', video],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)['streams'][0]
    w, h = st['width'], st['height']
    rot = next((abs(int(d.get('rotation', 0))) for d in st.get('side_data_list', []) if 'rotation' in d), 0)
    if rot in (90, 270):
        w, h = h, w  # ffmpeg applique la rotation au décodage
    return w, h, st['r_frame_rate']


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
    """Étage carte graphique : RVM (état récurrent) puis, en mode fin, ViTMatte sur la bande."""

    def __init__(self, w, h, fin):
        self.rvm = moteur.ouvrir(os.path.join(MOD, 'rvm_resnet50_fp32.onnx'))
        self.rec = [np.zeros((1, 1, 1, 1), np.float32)] * 4
        self.dr = np.array([0.25], np.float32)
        self.fin = fin
        if fin:
            self.vm = moteur.ouvrir(os.path.join(MOD, 'vitmatte_s_544x960.onnx' if w >= h else 'vitmatte_s_960x544.onnx'))
            self.vh, self.vw = self.vm.get_inputs()[0].shape[2:]
        self.w, self.h = w, h

    def __call__(self, img, chw):
        """img float32 (h, w, 3) et sa copie contiguë (1, 3, h, w) -> (alpha RVM, alpha ViTMatte ou None, bande ou None).
        En mode par défaut, SEUL RVM tourne ici : la bande et la sigmoïde partent dans l'étage processeur (finir),
        pour que l'étage carte graphique ne fasse que du calcul carte graphique."""
        pha, *self.rec = self.rvm.run(SORTIES_RVM, {'src': chw, 'r1i': self.rec[0], 'r2i': self.rec[1],
                                               'r3i': self.rec[2], 'r4i': self.rec[3], 'downsample_ratio': self.dr})
        a = pha[0, 0]
        if not self.fin:
            return a, None, None
        m = calcul.bande(a)
        tri = np.where(m, 0.5, (a > calcul.SEUIL_HAUT).astype(np.float32)).astype(np.float32)
        small = cv2.resize(img, (self.vw, self.vh), interpolation=cv2.INTER_AREA)
        trs = cv2.resize(tri, (self.vw, self.vh), interpolation=cv2.INTER_NEAREST)
        x = np.concatenate([((small - 0.5) / 0.5).transpose(2, 0, 1), trs[None]], 0)[None].astype(np.float32)
        return a, cv2.resize(self.vm.run(None, {'input': x})[0][0, 0], (self.w, self.h), interpolation=cv2.INTER_LINEAR), m


def finir(img, a, af, m, bg):
    """Étage processeur : retouche de la bande puis incrustation -> octets RGB24."""
    if af is None:
        a = calcul.sigmoide(a, calcul.bande(a))
    else:
        a = a.copy(); a[m] = af[m]
    return calcul.incruster_octets(img, a.clip(0, 1), bg)


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
            p = preparer(img)
            ecrire(finir(img, *det(*p), bg)); n += 1
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
            q_out.put((p[0], *det(*p))); n += 1
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
    if len(args) != 3 or drapeaux - {'--fin', '--sequentiel'}:
        sys.exit(__doc__)
    video, fspec, sortie = args
    w, h, fps = dims(video)
    det = Detoureur(w, h, '--fin' in drapeaux)
    bg = fond(fspec, w, h)
    dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', video, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                            '-r', fps, '-i', '-', '-i', video, '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264',
                            '-crf', '16', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-shortest', sortie],
                           stdin=subprocess.PIPE)
    taille = w * h * 3

    def lire():
        buf = dec.stdout.read(taille)
        return None if len(buf) < taille else np.frombuffer(buf, np.uint8).reshape(h, w, 3).astype(np.float32) / 255

    t0 = time.time()
    n = traiter(lire, enc.stdin.write, det, bg, parallele='--sequentiel' not in drapeaux)
    enc.stdin.close(); enc.wait(); dec.wait()
    if enc.returncode or dec.returncode:
        sys.exit(f'ÉCHEC ffmpeg (décodage {dec.returncode}, encodage {enc.returncode})')
    tt = time.time() - t0
    print(f'FINI : {n} images en {tt:.1f} s ({n / tt:.1f} images/s) -> {sortie}')


if __name__ == '__main__':
    main()
