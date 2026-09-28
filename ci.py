"""La CI est CETTE commande, identique en local et sur GitHub (le workflow ne fait que l'appeler) :
    python ci.py
1) télécharge ce qui manque (modèle RVM officiel, vidéo d'essai publique avec une personne) dans .ci/ et modeles/ ;
2) lance TOUS les tests, dont l'équivalence séquentiel/parallèle sur la vidéo d'essai ;
3) détoure la vidéo d'essai de bout en bout et vérifie la sortie (flux vidéo, nombre d'images) ;
4) affiche le système, la carte graphique réellement utilisée et la vitesse.
Code de sortie ≠ 0 au moindre échec. Le mode --fin (ViTMatte) n'est testé que si ses modèles sont présents."""
import os, sys, json, subprocess, urllib.request, platform

ICI = os.path.dirname(os.path.abspath(__file__))
RVM = ('modeles/rvm_resnet50_fp32.onnx',
       'https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx')
ESSAI = ('.ci/essai.mp4', 'https://raw.githubusercontent.com/pq-yang/MatAnyone2/main/inputs/video/test-sample2.mp4')


def telecharger(rel, url):
    p = os.path.join(ICI, rel)
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        print(f'téléchargement {url}', flush=True)
        urllib.request.urlretrieve(url, p + '.part'); os.replace(p + '.part', p)
    return p


def main():
    rvm, essai = telecharger(*RVM), telecharger(*ESSAI)
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        # Machines GitHub sans carte graphique : repli processeur AUTORISÉ ici seulement (voir moteur.session_ci).
        # En local, jamais : la CI locale prouve aussi que la carte graphique est bien utilisée.
        os.environ['DETOURAGE_CI'] = '1'
    import moteur
    s = moteur.ouvrir(rvm)
    print(f'SYSTÈME {platform.system()} {platform.machine()} | fournisseur : {s.get_providers()[0]}', flush=True)
    del s
    env = dict(os.environ, DETOURAGE_RUSH=essai, PYTHONUTF8='1')
    r = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_detourer'], cwd=ICI, env=env)
    if r.returncode:
        sys.exit('ÉCHEC : tests')
    sortie = os.path.join(ICI, '.ci', 'sortie.mp4')
    r = subprocess.run([sys.executable, 'detourer.py', essai, '1f2a5c', sortie], cwd=ICI, env=env)
    if r.returncode:
        sys.exit('ÉCHEC : détourage de bout en bout')
    info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries',
                                      'stream=width,height,nb_read_frames', '-of', 'json', sortie],
                                     capture_output=True, text=True, check=True).stdout)['streams'][0]
    if int(info['nb_read_frames']) < 50:
        sys.exit(f'ÉCHEC : sortie trop courte {info}')
    print(f'CI VERTE : sortie {info["width"]}x{info["height"]}, {info["nb_read_frames"]} images')


if __name__ == '__main__':
    main()
