"""Configuração do pytest: coloca ``src/`` no caminho de importação.

Permite que ``import data_structures`` (e demais módulos de src/) funcione ao
rodar ``python -m pytest`` a partir da raiz do repositório.
"""
import os
import sys

_RAIZ = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_RAIZ, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
