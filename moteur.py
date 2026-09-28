"""Choix de la carte graphique : SEUL endroit du projet qui sait quel système tourne (règle cross-OS du parc).

Sources officielles lues le 28/09/2026 (onnxruntime.ai) :
- Windows : paquet `onnxruntime-directml`, fournisseur DmlExecutionProvider ; la doc EXIGE enable_mem_pattern=False
  et execution_mode=ORT_SEQUENTIAL. DirectML = Windows uniquement.
- macOS : le paquet standard `onnxruntime` (arm64) inclut CoreMLExecutionProvider ; ModelFormat=MLProgram (macOS 12+),
  MLComputeUnits=ALL. Les opérations non gérées par CoreML retombent sur le processeur (partition partielle, normal).
- Linux : CUDAExecutionProvider si `onnxruntime-gpu` + carte Nvidia. Linux sans Nvidia = NON PRIS EN CHARGE (refus nommé).

🛑 WebGPU (plugin `onnxruntime-ep-webgpu` 0.4.0) EXCLU, mesuré le 28/09/2026 sur RX 7600 et sur Linux/lavapipe :
RVM plante (« Shape mismatch attempting to re-use buffer ») ; avec enable_mem_reuse=False il tourne en 55 ms mais rend
un contour FAUX (63 % des pixels à plus de 5 % d'écart de la référence processeur), SANS AUCUNE ERREUR. Ne jamais le
réintroduire sans que test_justesse_contre_reference_processeur passe dessus.
⚠️ Un fournisseur n'est admis que s'il rend le MÊME contour que le processeur (test de justesse) : « ça tourne » ne
prouve rien. DirectML mesuré exact (écart 0,000000) le 28/09/2026.

⚠️ NE JAMAIS laisser un calcul partir sur le processeur en silence : si le premier fournisseur de la session n'est pas
une carte graphique, ARRÊT immédiat avec un message qui dit pourquoi (l'opérateur juge un repli CPU comme une panne).
"""
import sys
import onnxruntime as ort

GPU = ('DmlExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider')


def _options():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return so


def session(chemin):
    """Session ONNX Runtime sur la carte graphique de la machine, ou arrêt nommé."""
    dispo = set(ort.get_available_providers())
    so = _options()
    if sys.platform == 'win32' and 'DmlExecutionProvider' in dispo:
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        s = ort.InferenceSession(chemin, so, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
    elif sys.platform == 'darwin' and 'CoreMLExecutionProvider' in dispo:
        s = ort.InferenceSession(chemin, so, providers=[
            ('CoreMLExecutionProvider', {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'ALL'}), 'CPUExecutionProvider'])
    elif 'CUDAExecutionProvider' in dispo:
        s = ort.InferenceSession(chemin, so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    else:
        sys.exit(f'ARRÊT : aucune carte graphique prise en charge sur ce système ({sys.platform}) : Windows (DirectML), '
                 f'macOS (CoreML) ou Linux + Nvidia (CUDA). Voir LISEZ-MOI.md.')
    premier = s.get_providers()[0]
    if premier not in GPU:
        sys.exit(f'ARRÊT : {chemin} retombe sur le processeur ({premier}) au lieu de la carte graphique.')
    return s


def ouvrir(chemin):
    """Point d'entrée unique pour le reste du projet : carte graphique obligatoire, sauf si DETOURAGE_CI=1 (posé
    UNIQUEMENT par ci.py, jamais dans l'environnement de l'opérateur)."""
    import os
    return session_ci(chemin) if os.environ.get('DETOURAGE_CI') == '1' else session(chemin)


def session_ci(chemin):
    """RÉSERVÉ À ci.py sur les machines GitHub, qui n'ont PAS de carte graphique (mesuré 28/09/2026 : DirectML
    « No devices detected »). Essaie la carte graphique ; à défaut, processeur AVEC le nom du fournisseur affiché,
    pour prouver que le CODE tourne sur ce système. ⚠️ Ne JAMAIS appeler depuis detourer.py : chez l'opérateur, un
    repli processeur reste une panne."""
    try:
        return session(chemin)
    except SystemExit as e:
        print(f'CI SANS CARTE GRAPHIQUE ({e}) : exécution sur processeur pour prouver le code', flush=True)
        return ort.InferenceSession(chemin, _options(), providers=['CPUExecutionProvider'])
