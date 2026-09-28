"""Choix de la carte graphique : SEUL endroit du projet qui sait quel système tourne (règle cross-OS du parc).

Sources officielles lues le 28/09/2026 (onnxruntime.ai) :
- Windows : paquet `onnxruntime-directml`, fournisseur DmlExecutionProvider ; la doc EXIGE enable_mem_pattern=False
  et execution_mode=ORT_SEQUENTIAL. DirectML = Windows uniquement.
- macOS : le paquet standard `onnxruntime` (arm64) inclut CoreMLExecutionProvider ; ModelFormat=MLProgram (macOS 12+),
  MLComputeUnits=ALL. Les opérations non gérées par CoreML retombent sur le processeur (partition partielle, normal).
- Linux : CUDAExecutionProvider si `onnxruntime-gpu` + carte Nvidia ; sinon WebGPU via le plugin `onnxruntime-ep-webgpu`
  (Vulkan). WebGPU mesuré 2× plus lent que DirectML sur une RX 7600 : jamais préféré là où DirectML existe.

⚠️ NE JAMAIS laisser un calcul partir sur le processeur en silence : si le premier fournisseur de la session n'est pas
une carte graphique, ARRÊT immédiat avec un message qui dit pourquoi (l'opérateur juge un repli CPU comme une panne).
État de preuve : Windows + RX 7600 PROUVÉ par exécution (28/09/2026) ; macOS et Linux = INCONNU tant que non exécutés.
"""
import sys
import onnxruntime as ort

GPU = ('DmlExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider', 'WebGpuExecutionProvider')


def _options():
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return so


def _webgpu(so):
    """Enregistre le plugin WebGPU (Linux sans Nvidia). Retourne False s'il est absent."""
    try:
        import onnxruntime_ep_webgpu as wep
    except ImportError:
        return False
    try:
        ort.register_execution_provider_library('webgpu', wep.get_library_path())
    except Exception:
        pass  # déjà enregistré dans ce processus
    dev = next((d for d in ort.get_ep_devices() if d.ep_name == wep.get_ep_name()), None)
    if dev is None:
        return False
    so.add_provider_for_devices([dev], {})
    return True


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
    elif _webgpu(so):
        s = ort.InferenceSession(chemin, sess_options=so)
    else:
        sys.exit(f'ARRÊT : aucune carte graphique utilisable sur ce système ({sys.platform}). Installe le paquet '
                 f'du requirements correspondant (voir LISEZ-MOI.md).')
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
