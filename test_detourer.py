"""Tests : python -m unittest test_detourer (depuis ce dossier, avec le venv).
L'équivalence séquentiel/parallèle tourne sur un VRAI rush (variable DETOURAGE_RUSH = chemin d'un .mp4) et sur la
vraie carte graphique : c'est la chaîne réelle, pas une imitation."""
import os, unittest, subprocess, hashlib, threading
import numpy as np
import calcul, detourer


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


class Parallelisme(unittest.TestCase):
    # Détoureur factice de même forme que le vrai (Detoureur.__call__(img, chw) -> (alpha, None, None)).
    DET = staticmethod(lambda img, chw: (img[..., 0], None, None))
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

    def _equivalence(self, fin):
        video = os.environ['DETOURAGE_RUSH']
        w, h, _ = detourer.dims(video)
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', video, '-frames:v', '40', '-f', 'rawvideo',
                              '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
        frames = [np.frombuffer(raw[i:i + w * h * 3], np.uint8).reshape(h, w, 3).astype(np.float32) / 255
                  for i in range(0, len(raw), w * h * 3)]
        bg = detourer.fond('1f2a5c', w, h)
        empreintes = []
        for par in (False, True):
            it, sortie = iter(frames), []
            detourer.traiter(lambda: next(it, None), sortie.append, detourer.Detoureur(w, h, fin), bg, par)
            empreintes.append([hashlib.sha256(b).hexdigest() for b in sortie])
        self.assertEqual(len(empreintes[0]), len(frames))
        self.assertEqual(empreintes[0], empreintes[1], f'fin={fin} : parallèle ≠ séquentiel')

    @unittest.skipUnless(os.environ.get('DETOURAGE_RUSH'), 'DETOURAGE_RUSH absent : pas de rush réel')
    def test_justesse_contre_reference_processeur(self):
        """Le fournisseur choisi par moteur.ouvrir DOIT rendre le même contour que le processeur (référence).
        Né le 28/09/2026 : WebGPU tournait sans erreur et rendait 63 % de pixels faux ; l'équivalence
        séquentiel/parallèle ne pouvait pas le voir (deux résultats faux identiques)."""
        import onnxruntime as ort, moteur
        video = os.environ['DETOURAGE_RUSH']
        w, h, _ = detourer.dims(video)
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', video, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt',
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

    @unittest.skipUnless(os.environ.get('DETOURAGE_RUSH'), 'DETOURAGE_RUSH absent : pas de rush réel')
    def test_parallele_identique_au_sequentiel_sur_rush_reel(self):
        self._equivalence(False)

    @unittest.skipUnless(os.environ.get('DETOURAGE_RUSH') and os.path.exists(
        os.path.join(detourer.MOD, 'vitmatte_s_960x544.onnx')), 'rush ou modèles ViTMatte absents : mode --fin non testé')
    def test_parallele_identique_au_sequentiel_mode_fin(self):
        self._equivalence(True)


if __name__ == '__main__':
    unittest.main()
