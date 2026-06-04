"""
performance_monitor.py
======================

Instrumentação de desempenho dos algoritmos. Para cada tamanho de instância
N ∈ {5, 8, 10, 12, 20, 50, 100} mede:

  * **tempo**   — ``time.perf_counter`` em milissegundos (ms);
  * **memória** — pico via ``tracemalloc`` em megabytes (MB);
  * **operações elementares** — contadores instrumentados em cada algoritmo
    (chamadas recursivas na força bruta; pushes/pops do heap no guloso).

Compara **Força Bruta** (MST exaustiva) vs **Guloso** (Prim). A força bruta
só roda enquanto N <= LIMITE_FORCA_BRUTA (=12); acima disso registra apenas o
guloso e marca a força bruta como "inviável". Calcula o **gap de otimalidade**
(custo_guloso / custo_otimo - 1), que deve ser 0 para a MST de Prim (Prim é
exato), servindo de validação cruzada.

Os resultados são salvos em ``data/processed/performance.json``.
"""

from __future__ import annotations

import json
import os
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from brute_force import LIMITE_FORCA_BRUTA, todas_arvores_geradoras
from data_structures import Grafo, Municipio
from greedy import prim_mst

_AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ_REPO = os.path.abspath(os.path.join(_AQUI, ".."))
DIR_PROCESSED = os.path.join(RAIZ_REPO, "data", "processed")

TAMANHOS_PADRAO: List[int] = [5, 8, 10, 12, 20, 50, 100]


@dataclass
class MedidaAlgoritmo:
    algoritmo: str
    n: int
    tempo_ms: float
    memoria_mb: float
    operacoes: int
    custo: float
    viavel: bool = True


@dataclass
class LinhaComparativa:
    n: int
    forca_bruta: Optional[MedidaAlgoritmo]
    guloso: MedidaAlgoritmo
    gap_otimalidade: Optional[float]  # guloso/otimo - 1 ; None se FB inviável


def _amostra_grafo(municipios: List[Municipio], n: int, k: int = 3) -> Grafo:
    """Constrói um grafo com os ``n`` primeiros municípios (k-vizinhos)."""
    from loader import construir_grafo
    sel = municipios[:n]
    k_eff = min(k, max(1, len(sel) - 1))
    return construir_grafo(sel, f"perf_N{n}", k_eff)


def _medir(func, *args) -> Dict[str, object]:
    """Executa ``func(*args)`` medindo tempo (ms), pico de memória (MB)."""
    tracemalloc.start()
    inicio = time.perf_counter()
    resultado = func(*args)
    fim = time.perf_counter()
    _atual, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "resultado": resultado,
        "tempo_ms": (fim - inicio) * 1000.0,
        "memoria_mb": pico / (1024 * 1024),
    }


def medir_para_n(municipios: List[Municipio], n: int) -> LinhaComparativa:
    """Roda Prim e (se viável) Força Bruta para um grafo de tamanho ``n``."""
    grafo = _amostra_grafo(municipios, n)

    # --- Guloso (Prim) ---
    m_guloso = _medir(prim_mst, grafo)
    res_guloso = m_guloso["resultado"]
    medida_guloso = MedidaAlgoritmo(
        algoritmo="Guloso (Prim)",
        n=grafo.num_vertices(),
        tempo_ms=round(m_guloso["tempo_ms"], 4),
        memoria_mb=round(m_guloso["memoria_mb"], 6),
        operacoes=res_guloso.operacoes,
        custo=round(res_guloso.custo_total, 6),
    )

    # --- Força Bruta (só se N <= limite) ---
    medida_fb: Optional[MedidaAlgoritmo] = None
    gap: Optional[float] = None
    if grafo.num_vertices() <= LIMITE_FORCA_BRUTA:
        m_fb = _medir(todas_arvores_geradoras, grafo)
        res_fb = m_fb["resultado"]
        medida_fb = MedidaAlgoritmo(
            algoritmo="Forca Bruta (MST exaustiva)",
            n=grafo.num_vertices(),
            tempo_ms=round(m_fb["tempo_ms"], 4),
            memoria_mb=round(m_fb["memoria_mb"], 6),
            operacoes=res_fb.chamadas_recursivas,
            custo=round(res_fb.custo_total, 6),
        )
        if res_fb.custo_total > 0:
            gap = res_guloso.custo_total / res_fb.custo_total - 1.0
            # zera ruído de ponto flutuante: Prim é exato para a MST
            if abs(gap) < 1e-6:
                gap = 0.0
            else:
                gap = round(gap, 8)
        else:
            gap = 0.0
    else:
        medida_fb = MedidaAlgoritmo(
            algoritmo="Forca Bruta (MST exaustiva)",
            n=grafo.num_vertices(),
            tempo_ms=float("nan"),
            memoria_mb=float("nan"),
            operacoes=-1,
            custo=float("nan"),
            viavel=False,
        )

    return LinhaComparativa(
        n=grafo.num_vertices(),
        forca_bruta=medida_fb,
        guloso=medida_guloso,
        gap_otimalidade=gap,
    )


def rodar_benchmark(
    municipios: List[Municipio],
    tamanhos: Optional[List[int]] = None,
    salvar: bool = True,
) -> List[LinhaComparativa]:
    """Roda o benchmark completo e (opcionalmente) salva o JSON."""
    tamanhos = tamanhos or TAMANHOS_PADRAO
    disponiveis = len(municipios)
    linhas: List[LinhaComparativa] = []
    for n in tamanhos:
        if n > disponiveis:
            continue
        linhas.append(medir_para_n(municipios, n))

    if salvar:
        os.makedirs(DIR_PROCESSED, exist_ok=True)
        caminho = os.path.join(DIR_PROCESSED, "performance.json")

        def _clean(med: Optional[MedidaAlgoritmo]):
            if med is None:
                return None
            d = asdict(med)
            for k in ("tempo_ms", "memoria_mb", "custo"):
                if isinstance(d[k], float) and d[k] != d[k]:  # NaN
                    d[k] = None
            return d

        payload = [
            {
                "n": ln.n,
                "forca_bruta": _clean(ln.forca_bruta),
                "guloso": _clean(ln.guloso),
                "gap_otimalidade": ln.gap_otimalidade,
            }
            for ln in linhas
        ]
        with open(caminho, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    return linhas


def imprimir_tabela(linhas: List[LinhaComparativa]) -> None:
    """Imprime a tabela comparativa no console."""
    print(f"{'N':>4} | {'FB tempo(ms)':>13} | {'FB ops':>10} | "
          f"{'Guloso(ms)':>11} | {'Guloso ops':>11} | {'gap':>8}")
    print("-" * 72)
    for ln in linhas:
        fb = ln.forca_bruta
        fb_t = f"{fb.tempo_ms:.3f}" if (fb and fb.viavel) else "inviavel"
        fb_o = f"{fb.operacoes}" if (fb and fb.viavel) else "-"
        gap = "exato(0)" if ln.gap_otimalidade == 0 else (
            f"{ln.gap_otimalidade}" if ln.gap_otimalidade is not None else "-"
        )
        print(f"{ln.n:>4} | {fb_t:>13} | {fb_o:>10} | "
              f"{ln.guloso.tempo_ms:>11.3f} | {ln.guloso.operacoes:>11} | {gap:>8}")


__all__ = [
    "TAMANHOS_PADRAO",
    "MedidaAlgoritmo",
    "LinhaComparativa",
    "medir_para_n",
    "rodar_benchmark",
    "imprimir_tabela",
    "DIR_PROCESSED",
]
