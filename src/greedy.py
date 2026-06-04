"""
greedy.py
=========

Algoritmos **gulosos** implementados do zero com ``heapq`` (fila de
prioridade). NÃO usam o networkx para o cálculo — o networkx só aparece em
``visualizations.py`` para desenhar.

1. ``prim_mst(grafo)``       — Árvore Geradora Mínima (MST) pelo algoritmo de
   Prim. Conecta todos os municípios minimizando o tempo total de
   deslocamento — útil para planejar a malha de bases/rotas de brigadas.

2. ``dijkstra(grafo, hub)``  — caminhos mínimos (em horas) a partir de um hub
   (base operacional) para todos os demais municípios. Reconstrói o caminho.

Ambos usam o **heap binário** do ``heapq`` como fila de prioridade
(min-heap). Complexidade: O(E log V).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from data_structures import Grafo


# ===========================================================================
# Resultados
# ===========================================================================
@dataclass
class ResultadoMST:
    arestas: List[Tuple[int, int, float]]  # (u, v, peso_horas) da MST
    custo_total: float
    operacoes: int                          # operações elementares (pushes/pops do heap)


@dataclass
class ResultadoDijkstra:
    origem: int
    distancias: Dict[int, float]            # hub -> horas até cada vértice
    predecessores: Dict[int, Optional[int]]
    operacoes: int

    def caminho_ate(self, destino: int) -> List[int]:
        """Reconstrói o caminho hub→destino a partir dos predecessores."""
        if destino not in self.distancias or self.distancias[destino] == float("inf"):
            return []
        caminho: List[int] = []
        atual: Optional[int] = destino
        while atual is not None:
            caminho.append(atual)
            atual = self.predecessores.get(atual)
        caminho.reverse()
        return caminho


# ===========================================================================
# 1. Algoritmo de Prim (MST) — heapq
# ===========================================================================
def prim_mst(grafo: Grafo, raiz: Optional[int] = None) -> ResultadoMST:
    """Árvore Geradora Mínima pelo algoritmo de Prim com min-heap.

    Estratégia gulosa: a cada passo, adiciona à árvore a aresta de menor peso
    que conecta um vértice já incluído a um vértice ainda fora. A escolha do
    mínimo é feita por um **heap** de candidatos ``(peso, de, para)``.

    Pré-condição: grafo conexo (caso contrário devolve a MST da componente da
    raiz). Retorna as arestas da árvore, o custo total e o nº de operações.
    """
    if grafo.num_vertices() == 0:
        return ResultadoMST(arestas=[], custo_total=0.0, operacoes=0)

    if raiz is None:
        raiz = grafo.ids()[0]

    na_arvore: set[int] = set()
    arestas_mst: List[Tuple[int, int, float]] = []
    custo_total = 0.0
    operacoes = 0

    # heap de candidatos: (peso, vertice_origem, vertice_destino)
    heap: List[Tuple[float, int, int]] = []
    na_arvore.add(raiz)
    for (viz, peso) in grafo.vizinhos(raiz):
        heapq.heappush(heap, (peso, raiz, viz))
        operacoes += 1

    while heap and len(na_arvore) < grafo.num_vertices():
        peso, de, para = heapq.heappop(heap)
        operacoes += 1
        if para in na_arvore:
            continue  # aresta obsoleta (destino já coberto)
        # adiciona aresta à MST
        na_arvore.add(para)
        arestas_mst.append((de, para, peso))
        custo_total += peso
        # relaxa: empilha as novas arestas de fronteira
        for (viz, p) in grafo.vizinhos(para):
            if viz not in na_arvore:
                heapq.heappush(heap, (p, para, viz))
                operacoes += 1

    return ResultadoMST(
        arestas=arestas_mst,
        custo_total=round(custo_total, 6),
        operacoes=operacoes,
    )


# ===========================================================================
# 2. Algoritmo de Dijkstra (caminho mínimo) — heapq
# ===========================================================================
def dijkstra(grafo: Grafo, origem: int) -> ResultadoDijkstra:
    """Caminhos mínimos (em horas) a partir de ``origem`` (hub) para todos.

    Usa um min-heap de ``(distancia_acumulada, vertice)``. Pesos não-negativos
    (tempos de viagem), portanto a escolha gulosa do vértice mais próximo ainda
    não finalizado é ótima. Retorna distâncias, predecessores (para reconstruir
    caminhos) e o nº de operações.
    """
    INF = float("inf")
    distancias: Dict[int, float] = {vid: INF for vid in grafo.ids()}
    predecessores: Dict[int, Optional[int]] = {vid: None for vid in grafo.ids()}
    finalizados: set[int] = set()
    operacoes = 0

    if origem not in distancias:
        return ResultadoDijkstra(origem, distancias, predecessores, 0)

    distancias[origem] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, origem)]

    while heap:
        d, u = heapq.heappop(heap)
        operacoes += 1
        if u in finalizados:
            continue
        finalizados.add(u)
        for (viz, peso) in grafo.vizinhos(u):
            if viz in finalizados:
                continue
            nova = d + peso
            if nova < distancias[viz]:
                distancias[viz] = nova
                predecessores[viz] = u
                heapq.heappush(heap, (nova, viz))
                operacoes += 1

    return ResultadoDijkstra(
        origem=origem,
        distancias={k: round(v, 6) if v != INF else v for k, v in distancias.items()},
        predecessores=predecessores,
        operacoes=operacoes,
    )


def hub_otimo(grafo: Grafo) -> int:
    """Heurística: escolhe como hub o município de **maior índice de risco**.

    Em triagem de incêndios, faz sentido posicionar a base no foco mais crítico.
    """
    return max(grafo.vertices.values(), key=lambda m: m.risco).id_municipio


__all__ = [
    "ResultadoMST",
    "ResultadoDijkstra",
    "prim_mst",
    "dijkstra",
    "hub_otimo",
]
