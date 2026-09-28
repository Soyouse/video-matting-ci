"""Tests : python -m unittest test_detourer (depuis ce dossier, avec le venv).
Les tests sur vidéo tournent sur un VRAI rush (variable DETOURAGE_RUSH = chemin d'un .mp4) et sur la vraie carte
graphique ; les tests « bout en bout » lancent detourer.py lui-même (le point d'entrée réel), sur des clips fabriqués
à partir de ce rush pour reproduire les pièges des téléphones (cadence variable, HDR)."""
import os, json, unittest, subprocess, hashlib, threading, tempfile, sys
import numpy as np
import calcul, detourer

ICI = os.path.dirname(os.path.abspath(__file__))
RUSH = os.environ.get('DETOURAGE_RUSH')


class Calculs(unittest.TestCase):
    def test_bande_entoure_le_contour_et_seulement_lui(self):
        a = np.zeros((200, 200), np.float32); a[:, 100:] = 1.0
        m = calcul.bande(a)
        self.assertTrue(m[:, 95:105].all())
        self.assertFalse(m[:, :80].any()); self.assertFalse(m[:, 120:].any())

    def test_sigmoide_ne_touche_que_la_bande(self):
        a = np.random.default_rng(0).random((50, 50)).astype(np.float32)
        m = np.zeros_like(a, bool); m[10:20] = True
        s = calcul.sigmoide(a, m)
        np.testing.assert_array_equal(s[~m], a[~m])
        self.assertTrue((np.abs(s[m] - 0.5) >= np.abs(a[m] - 0.5) - 1e-6).all())  # durcit, ne ramollit jamais

    def test_sigmoide_garde_les_extremes_exacts(self):
        m = np.ones((1, 3), bool)
        np.testing.assert_allclose(calcul.sigmoide(np.array([[0.0, 0.5, 1.0]], np.float32), m), [[0.0, 0.5, 1.0]], atol=1e-6)

    def test_incruster_formule(self):
        img = np.full((2, 2, 3), 0.8, np.float32); bg = np.full((2, 2, 3), 0.2, np.float32)
        a = np.array([[0, 1], [0.5, 0.25]], np.float32)
        np.testing.assert_allclose(calcul.incruster(img, a, bg)[..., 0], [[0.2, 0.8], [0.5, 0.35]], atol=1e-6)

    def test_incruster_octets_identique(self):
        rng = np.random.default_rng(1)
        img = rng.random((64, 48, 3)).astype(np.float32); bg = rng.random((64, 48, 3)).astype(np.float32)
        a = rng.random((64, 48)).astype(np.float32); a[:5] = 0; a[-5:] = 1
        ref = (calcul.incruster(img, a, bg) * 255 + 0.5).clip(0, 255).astype(np.uint8)
        got = np.frombuffer(calcul.incruster_octets(img, a, bg), np.uint8).reshape(ref.shape)
        self.assertLessEqual(int(np.abs(got.astype(int) - ref.astype(int)).max()), 0)

    def test_cadence(self):
        self.assertEqual(detourer.cadence('163200/5489', '179/6'), 30)  # rush réel : 29,73 i/s moyenne -> 30
        self.assertEqual(detourer.cadence('0/0', '25/1'), 25)  # moyenne absente -> nominale
        self.assertEqual(detourer.cadence(None, None), 30)
        self.assertEqual(detourer.cadence('24000/1001', '24000/1001'), 24)

    def test_filtre_video(self):
        self.assertEqual(detourer.filtre_video(30, False), 'fps=30')
        f = detourer.filtre_video(30, True)
        self.assertTrue(f.startswith('zscale=') and 'tonemap=' in f and f.endswith(',fps=30'))


class Parallelisme(unittest.TestCase):
    # Détoureur factice de même forme que le vrai : Detoureur.__call__(img, chw) -> alpha (h, w).
    DET = staticmethod(lambda img, chw: img[..., 0])
    BG = np.zeros((4, 4, 3), np.float32)

    def test_erreur_de_lecture_remonte_sans_bloquer(self):
        def lire():
            raise IOError('disque')
        with self.assertRaises(IOError):
            detourer.traiter(lire, lambda b: None, self.DET, self.BG)

    def test_erreur_d_ecriture_remonte_sans_bloquer(self):
        it = iter([np.zeros((4, 4, 3), np.float32)] * 50)

        def ecrire(b):
            raise IOError('plein')
        with self.assertRaises(IOError):
            detourer.traiter(lambda: next(it, None), ecrire, self.DET, self.BG)

    def test_erreur_carte_graphique_remonte_sans_bloquer(self):
        # 50 images > PROFONDEUR : le lecteur est bloqué sur une file pleine au moment de l'erreur
        it = iter([np.zeros((4, 4, 3), np.float32)] * 50)

        def det(img, chw):
            raise RuntimeError('carte graphique')
        res = []

        def lancer():
            try:
                detourer.traiter(lambda: next(it, None), lambda b: None, det, self.BG)
            except RuntimeError as e:
                res.append(e)
        t = threading.Thread(target=lancer, daemon=True); t.start()
        # Borne déclarée : « bloqué » contre « lent » est indécidable ; 30 s >> 50 images factices (quelques ms).
        t.join(30)
        self.assertFalse(t.is_alive(), 'traiter() est bloqué : le lecteur n\'a pas été libéré')
        self.assertEqual(len(res), 1)

    @unittest.skipUnless(RUSH, 'DETOURAGE_RUSH absent : pas de rush réel')
    def test_justesse_contre_reference_processeur(self):
        """Le fournisseur choisi par moteur.ouvrir DOIT rendre le même contour que le processeur (référence).
        Né le 28/09/2026 : WebGPU tournait sans erreur et rendait 63 % de pixels faux ; l'équivalence
        séquentiel/parallèle ne pouvait pas le voir (deux résultats faux identiques)."""
        import onnxruntime as ort, moteur
        w, h, _, _ = detourer.sonder(RUSH)
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', RUSH, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt',
                              'rgb24', '-'], capture_output=True, check=True).stdout
        img = np.frombuffer(raw[:w * h * 3], np.uint8).reshape(h, w, 3).astype(np.float32) / 255
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None]); z = [np.zeros((1, 1, 1, 1), np.float32)] * 4
        feed = {'src': x, 'r1i': z[0], 'r2i': z[1], 'r3i': z[2], 'r4i': z[3], 'downsample_ratio': np.array([0.25], np.float32)}
        chemin = os.path.join(detourer.MOD, 'rvm_resnet50_fp32.onnx')
        ref = ort.InferenceSession(chemin, providers=['CPUExecutionProvider']).run(['pha'], feed)[0]
        s = moteur.ouvrir(chemin)
        d = np.abs(s.run(['pha'], feed)[0] - ref)
        print(f'\nJUSTESSE {s.get_providers()[0]} : écart moyen {d.mean():.6f}, pixels > 1 % : {(d > 0.01).mean() * 100:.4f} %')
        self.assertLess(float(d.mean()), 1e-3)
        self.assertLess(float((d > 0.01).mean()), 1e-3)

    @unittest.skipUnless(RUSH, 'DETOURAGE_RUSH absent : pas de rush réel')
    def test_parallele_identique_au_sequentiel_sur_rush_reel(self):
        w, h, _, _ = detourer.sonder(RUSH)
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', RUSH, '-frames:v', '40', '-f', 'rawvideo',
                              '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
        frames = [np.frombuffer(raw[i:i + w * h * 3], np.uint8).reshape(h, w, 3).astype(np.float32) / 255
                  for i in range(0, len(raw), w * h * 3)]
        bg = detourer.fond('1f2a5c', w, h)
        empreintes = []
        for par in (False, True):
            it, sortie = iter(frames), []
            detourer.traiter(lambda: next(it, None), sortie.append, detourer.Detoureur(), bg, par)
            empreintes.append([hashlib.sha256(b).hexdigest() for b in sortie])
        self.assertEqual(len(empreintes[0]), len(frames))
        self.assertEqual(empreintes[0], empreintes[1], 'parallèle ≠ séquentiel')


def _duree(f, flux):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', flux, '-show_entries', 'stream=duration',
                          '-of', 'json', f], capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)['streams'][0]['duration'])


@unittest.skipUnless(RUSH, 'DETOURAGE_RUSH absent : pas de rush réel')
class BoutEnBout(unittest.TestCase):
    """Lance detourer.py (le vrai point d'entrée) sur des clips fabriqués qui reproduisent les pièges des téléphones."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='detourage-test-')

    def _detourer(self, src):
        sortie = os.path.join(self.tmp, 'sortie.mp4')
        r = subprocess.run([sys.executable, os.path.join(ICI, 'detourer.py'), src, '1f2a5c', sortie],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(os.path.exists(sortie))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, 'sortie.partiel.mp4')), 'fichier partiel resté')
        return sortie, r.stdout

    def test_aucune_image_perdue_et_son_aligne(self):
        """Né le 28/09/2026 : sur un rush de téléphone, 137 images écrites mais 133 dans le fichier (`-shortest`
        coupait la fin quand le son finit un peu avant l'image). Clip fabriqué pour reproduire : 3 s d'image, 2,97 s
        de son, et 1 image sur 3 retirée (cadence variable, comme un téléphone)."""
        clip = os.path.join(self.tmp, 'clip.mp4')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', RUSH, '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2.97',
                        '-t', '3', '-vf', 'select=mod(n\\,3)', '-fps_mode', 'vfr', '-map', '0:v', '-map', '1:a',
                        '-c:v', 'libx264', '-c:a', 'aac', clip], check=True)
        sortie, texte = self._detourer(clip)
        ecrites = int(texte.split('FINI : ')[1].split(' images')[0])
        lues = int(json.loads(subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                                              '-show_entries', 'stream=nb_read_frames', '-of', 'json', sortie],
                                             capture_output=True, text=True, check=True).stdout)['streams'][0]['nb_read_frames'])
        dv, da = _duree(sortie, 'v:0'), _duree(sortie, 'a:0')
        print(f'\nFICHIER : {lues}/{ecrites} images, image {dv:.3f} s, son {da:.3f} s')
        self.assertEqual(lues, ecrites, 'des images écrites ont disparu du fichier')
        self.assertLess(abs(dv - da), 0.1, f'image {dv:.3f} s contre son {da:.3f} s')

    def test_hdr_converti_sans_friction(self):
        """Une vidéo HDR (téléphone réglé en HDR) ne doit NI être refusée NI planter, et la conversion doit être
        RÉELLEMENT appliquée : le même contenu marqué HDR ou non doit donner deux images différentes.
        ⚠️ Limite déclarée : on prouve que la conversion s'applique, pas qu'elle est fidèle (pas de vrai rush HDR)."""
        def fabriquer(nom, hdr):
            p = os.path.join(self.tmp, nom)
            vf = ('setparams=color_trc=smpte2084:color_primaries=bt2020:colorspace=bt2020nc,' if hdr else '') + 'format=yuv420p10le'
            tags = ['-color_trc', 'smpte2084', '-color_primaries', 'bt2020', '-colorspace', 'bt2020nc'] if hdr else []
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', RUSH, '-t', '1', '-vf', vf, '-c:v', 'libx264', *tags, p],
                           check=True)
            return p
        hdr, sdr = fabriquer('hdr.mp4', True), fabriquer('sdr.mp4', False)
        w, h, fps, est_hdr = detourer.sonder(hdr)
        self.assertTrue(est_hdr, 'le clip fabriqué devrait être vu comme HDR')
        self.assertFalse(detourer.sonder(sdr)[3])
        # 1) le vrai point d'entrée prend le chemin HDR sans refuser ni planter
        _, texte = self._detourer(hdr)
        self.assertIn('HDR converti', texte)
        # 2) la chaîne de décodage HDR que detourer.py utilise change RÉELLEMENT les pixels (image entière, sans le
        #    fond de remplacement qui dilue l'écart : mesuré 4/255 sur la vidéo d'essai contre 16/255 sur un rush)
        images = []
        for src, marque in ((hdr, True), (sdr, False)):
            raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', src, '-vf', detourer.filtre_video(fps, marque),
                                  '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True,
                                 check=True).stdout
            images.append(np.frombuffer(raw[:w * h * 3], np.uint8).astype(np.float32))
        ecart = float(np.abs(images[0] - images[1]).mean())
        print(f'\nHDR : écart moyen image décodée avec conversion vs sans = {ecart:.1f} / 255')
        self.assertGreater(ecart, 10.0, 'la conversion HDR ne semble pas appliquée')


if __name__ == '__main__':
    unittest.main()
