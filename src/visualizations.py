"""
visualizations.py
=================

Geração das figuras do relatório. O networkx + matplotlib são usados **apenas
para desenhar** (os algoritmos foram implementados do zero em ``greedy.py`` e
``brute_force.py``).

Figuras geradas em ``report/figuras/``:
  * ``grafo_<cenario>_mst.png`` — municípios (posicionados pelo centroide real)
    com as arestas da MST de Prim destacadas em vermelho.
  * ``bst_<cenario>.png``       — diagrama da BST (10–15 nós) com os índices de
    risco como chaves.
  * ``tempo_vs_n.png``          — tempo de execução × N (Força Bruta vs Guloso),
    escala log no eixo do tempo.
  * ``gap_otimalidade.png``     — gap de otimalidade do guloso vs N.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # backend não-interativo (sem display)
import matplotlib.pyplot as plt
import networkx as nx

from data_structures import BinarySearchTree, Grafo, Node
from greedy import ResultadoMST
from performance_monitor import LinhaComparativa

_AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ_REPO = os.path.abspath(os.path.join(_AQUI, ".."))
DIR_FIGURAS = os.path.join(RAIZ_REPO, "report", "figuras")

# paleta de cores por bioma
CORES_BIOMA = {
    "Amazônia": "#2e7d32",
    "Cerrado": "#f9a825",
    "Caatinga": "#bf6c2c",
    "Mata Atlântica": "#1b5e20",
    "Pantanal": "#00838f",
    "Pampa": "#9e9d24",
}


def _garantir_dir() -> None:
    os.makedirs(DIR_FIGURAS, exist_ok=True)


# ===========================================================================
# 1. Grafo de municípios com a MST destacada
# ===========================================================================
def desenhar_grafo_mst(grafo: Grafo, mst: ResultadoMST, titulo: str,
                       nome_arquivo: str) -> str:
    """Desenha o grafo (posições = centroides reais) com a MST em vermelho."""
    _garantir_dir()
    G = nx.Graph()
    pos: Dict[int, Tuple[float, float]] = {}
    cores: List[str] = []
    tamanhos: List[float] = []
    rotulos: Dict[int, str] = {}

    for vid, m in grafo.vertices.items():
        G.add_node(vid)
        pos[vid] = (m.lon, m.lat)  # x=lon, y=lat
        cores.append(CORES_BIOMA.get(m.bioma, "#607d8b"))
        tamanhos.append(200 + 900 * m.risco)
        rotulos[vid] = m.nome[:10]

    for (u, v, p) in grafo.arestas():
        G.add_edge(u, v, weight=p)

    arestas_mst = {(min(u, v), max(u, v)) for (u, v, _p) in mst.arestas}

    fig, ax = plt.subplots(figsize=(11, 9))
    # arestas comuns (cinza claro)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cfd8dc", width=1.0)
    # arestas da MST (vermelho, grossas)
    mst_edges = [(u, v) for (u, v) in G.edges() if (min(u, v), max(u, v)) in arestas_mst]
    nx.draw_networkx_edges(G, pos, edgelist=mst_edges, ax=ax,
                           edge_color="#d32f2f", width=2.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cores,
                           node_size=tamanhos, edgecolors="#263238", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, labels=rotulos, ax=ax, font_size=7)

    ax.set_title(f"{titulo}\nMST de Prim (vermelho) — custo total "
                 f"{mst.custo_total:.1f} h | tamanho do nó ∝ índice de risco",
                 fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle=":", alpha=0.4)

    # legenda de biomas presentes
    biomas_presentes = {m.bioma for m in grafo.vertices.values()}
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=b,
                   markerfacecolor=CORES_BIOMA.get(b, "#607d8b"), markersize=10)
        for b in sorted(biomas_presentes)
    ]
    ax.legend(handles=handles, title="Bioma", loc="best", fontsize=8)

    caminho = os.path.join(DIR_FIGURAS, nome_arquivo)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


# ===========================================================================
# 2. Diagrama da BST
# ===========================================================================
def _posicoes_bst(no: Optional[Node], x: float, y: float, dx: float,
                  G: nx.DiGraph, pos: Dict[str, Tuple[float, float]],
                  rotulos: Dict[str, str], pai: Optional[str] = None) -> None:
    if no is None:
        return
    chave_id = f"{id(no)}"
    G.add_node(chave_id)
    pos[chave_id] = (x, y)
    nome = str(no.payload.get("nome", ""))[:8]
    rotulos[chave_id] = f"{no.chave:.2f}\n{nome}"
    if pai is not None:
        G.add_edge(pai, chave_id)
    _posicoes_bst(no.esquerda, x - dx, y - 1, dx / 1.8, G, pos, rotulos, chave_id)
    _posicoes_bst(no.direita, x + dx, y - 1, dx / 1.8, G, pos, rotulos, chave_id)


def desenhar_bst(bst: BinarySearchTree, titulo: str, nome_arquivo: str) -> str:
    """Desenha a BST como árvore hierárquica com chave (risco) e nome."""
    _garantir_dir()
    G = nx.DiGraph()
    pos: Dict[str, Tuple[float, float]] = {}
    rotulos: Dict[str, str] = {}
    _posicoes_bst(bst.raiz, 0.0, 0.0, 4.0, G, pos, rotulos)

    fig, ax = plt.subplots(figsize=(13, 7))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#455a64",
                           arrows=True, arrowsize=12, width=1.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#90caf9",
                           node_size=1700, edgecolors="#0d47a1", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, labels=rotulos, ax=ax, font_size=7)
    ax.set_title(f"{titulo}\nBST ordenada por índice de risco | "
                 f"altura={bst.altura()} | n={len(bst)}", fontsize=12)
    ax.axis("off")
    caminho = os.path.join(DIR_FIGURAS, nome_arquivo)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


# ===========================================================================
# 3. Tempo × N (Força Bruta vs Guloso)
# ===========================================================================
def desenhar_tempo_vs_n(linhas: List[LinhaComparativa],
                        nome_arquivo: str = "tempo_vs_n.png") -> str:
    _garantir_dir()
    ns_fb, t_fb = [], []
    ns_g, t_g = [], []
    for ln in linhas:
        ns_g.append(ln.n)
        t_g.append(max(ln.guloso.tempo_ms, 1e-4))
        if ln.forca_bruta and ln.forca_bruta.viavel:
            ns_fb.append(ln.n)
            t_fb.append(max(ln.forca_bruta.tempo_ms, 1e-4))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ns_fb, t_fb, "o-", color="#d32f2f", label="Força Bruta (exaustiva)")
    ax.plot(ns_g, t_g, "s-", color="#2e7d32", label="Guloso (Prim)")
    ax.set_yscale("log")
    ax.set_xlabel("N (nº de municípios / vértices)")
    ax.set_ylabel("Tempo de execução (ms, escala log)")
    ax.set_title("Custo computacional: Força Bruta vs Guloso\n"
                 "explosão combinatória da força bruta a partir de N≈12")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    caminho = os.path.join(DIR_FIGURAS, nome_arquivo)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


# ===========================================================================
# 4. Gap de otimalidade
# ===========================================================================
def desenhar_gap(linhas: List[LinhaComparativa],
                 nome_arquivo: str = "gap_otimalidade.png") -> str:
    _garantir_dir()
    ns, gaps = [], []
    for ln in linhas:
        if ln.gap_otimalidade is not None:
            ns.append(ln.n)
            gaps.append(ln.gap_otimalidade * 100.0)  # em %

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar([str(n) for n in ns], gaps, color="#1565c0")
    ax.axhline(0, color="#37474f", linewidth=1)
    ax.set_xlabel("N (vértices, onde a Força Bruta é viável)")
    ax.set_ylabel("Gap de otimalidade do Guloso (%)")
    ax.set_title("Gap de otimalidade: custo(Prim)/custo(ótimo) − 1\n"
                 "Prim é exato para a MST → gap = 0% (validação cruzada)")
    # limites com folga para mostrar barras de altura 0
    topo = max(gaps + [0.1])
    ax.set_ylim(min(gaps + [-0.1]) - 0.1, topo + 0.5)
    for i, g in enumerate(gaps):
        ax.text(i, g + 0.05, f"{g:.2f}%", ha="center", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    caminho = os.path.join(DIR_FIGURAS, nome_arquivo)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


__all__ = [
    "DIR_FIGURAS",
    "desenhar_grafo_mst",
    "desenhar_bst",
    "desenhar_tempo_vs_n",
    "desenhar_gap",
]
