# bm25s importe JAX quand il est installé (c'est le cas sur Colab). Or JAX réserve 75 % de la VRAM
# du GPU dès son import : il ne reste presque plus rien pour le générateur, d'où des erreurs
# « CUDA out of memory » sur un T4. On force donc JAX sur le CPU et on désactive sa préallocation.
# Ces variables doivent être posées avant le premier import de JAX, d'où leur place ici : la
# baseline importe bm25s avant groundtruth. setdefault laisse la main à qui les aurait déjà définies.
import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
