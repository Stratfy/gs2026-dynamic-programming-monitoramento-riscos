"""
brute_force.py
==============

Enumeração exaustiva (recursão + backtracking) usada como **oráculo de
validação** dos algoritmos gulosos. Aplicável apenas a instâncias pequenas
(N <= 12), pois o espaço de busca cresce de forma combinatória.

Fornece duas famílias de enumeração:

1. ``todos_os_caminhos(grafo, origem, destino)`` — enumera *todos* os caminhos
   simples (sem repetir vértice) entre origem e destino, devolvendo o de menor
   custo (soma dos pesos). Oráculo do **Dijkstra** (caminho mínimo).

2. ``todas_arvores_geradoras(grafo)`` — enumera *todas* as árvores geradoras
   por backtracking sobre o conjunto de arestas, devolvendo a de menor peso
   total. Oráculo do **Prim** (MST — árvore geradora mínima).

Ambas mantêm contadores de chamadas recursivas e de soluções avaliadas, que
alimentam a análise de complexidade empírica (``performance_monitor.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from data_structures import Grafo

LIMITE_FORCA_BRUTA: int = 12  # nº máximo de vértices recomendado


# ===========================================================================
# Resultados (dataclasses para clareza no relatório)
# ===========================================================================
@dataclass
class ResultadoCaminho:
    melhor_caminho: List[int]
    melhor_custo: float
    caminhos_avaliados: int
    chamadas_recursivas: int


@dataclass
class ResultadoMST:
    arestas: List[Tuple[int, int, float]]
    custo_total: float
    arvores_avaliadas: int
    chamadas_recursivas: int


# ===========================================================================
# 1. Todos os caminhos simples (oráculo do Dijkstra)
# ===========================================================================
def todos_os_caminhos(grafo: Grafo, origem: int, destino: int) -> ResultadoCaminho:
    """Enumera todos os caminhos simples origem→destino (backtracking).

    Retorna o caminho de menor custo. Levanta ``ValueError`` se a instância
    exceder ``LIMITE_FORCA_BRUTA`` (proteção contra explosão combinatória).
    """
    if grafo.num_vertices() > LIMITE_FORCA_BRUTA:
        raise ValueError(
            f"Força bruta limitada a N<={LIMITE_FORCA_BRUTA} vértices "
            f"(recebido {grafo.num_vertices()})."
        )

    contadores = {"caminhos": 0, "chamadas": 0}
    melhor: Dict[str, object] = {"caminho": None, "custo": float("inf")}
    visitados: Set[int] = set()

    def backtrack(atual: int, caminho: List[int], custo: float) -> None:
        contadores["chamadas"] += 1
        if atual == destino:
            contadores["caminhos"] += 1
            if custo < melhor["custo"]:
                melhor["custo"] = custo
                melhor["caminho"] = list(caminho)  # cópia da list
            return
        for (viz, peso) in grafo.vizinhos(atual):
            if viz in visitados:
                continue
            # Poda: nunca vale a pena seguir se já passamos do melhor custo
            if custo + peso >= melhor["custo"]:
                continue
            visitados.add(viz)
            caminho.append(viz)
            backtrack(viz, caminho, custo + peso)
            caminho.pop()
            visitados.remove(viz)

    visitados.add(origem)
    backtrack(origem, [origem], 0.0)

    return ResultadoCaminho(
        melhor_caminho=melhor["caminho"] or [],
        melhor_custo=(melhor["custo"] if melhor["caminho"] else float("inf")),
        caminhos_avaliados=contadores["caminhos"],
        chamadas_recursivas=contadores["chamadas"],
    )


# ===========================================================================
# 2. Todas as árvores geradoras (oráculo do Prim / MST)
# ===========================================================================
def todas_arvores_geradoras(grafo: Grafo) -> ResultadoMST:
    """Enumera todas as árvores geradoras por backtracking sobre as arestas.

    Estratégia: ordena as arestas e decide incluir/excluir cada uma. Uma seleção
    é árvore geradora se tem exatamente |V|-1 arestas, conecta todos os vértices
    e é acíclica (verificado via union-find incremental). Retorna a de menor
    custo total. Restrito a N <= ``LIMITE_FORCA_BRUTA``.
    """
    n = grafo.num_vertices()
    if n > LIMITE_FORCA_BRUTA:
        raise ValueError(
            f"Força bruta limitada a N<={LIMITE_FORCA_BRUTA} vértices (recebido {n})."
        )

    arestas: List[Tuple[int, int, float]] = grafo.arestas()
    ids: List[int] = grafo.ids()
    indice: Dict[int, int] = {vid: i for i, vid in enumerate(ids)}
    m = len(arestas)

    contadores = {"arvores": 0, "chamadas": 0}
    melhor: Dict[str, object] = {"arestas": None, "custo": float("inf")}

    def find(pai: List[int], x: int) -> int:
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def union(pai: List[int], a: int, b: int) -> bool:
        ra, rb = find(pai, a), find(pai, b)
        if ra == rb:
            return False  # criaria ciclo
        pai[ra] = rb
        return True

    def backtrack(i: int, selecao: List[Tuple[int, int, float]], custo: float) -> None:
        contadores["chamadas"] += 1
        # Poda: se já temos uma árvore (n-1 arestas), avalia
        if len(selecao) == n - 1:
            # verifica conectividade via union-find
            pai = list(range(n))
            ok = True
            for (u, v, _w) in selecao:
                if not union(pai, indice[u], indice[v]):
                    ok = False
                    break
            if ok and len({find(pai, indice[v]) for v in ids}) == 1:
                contadores["arvores"] += 1
                if custo < melhor["custo"]:
                    melhor["custo"] = custo
                    melhor["arestas"] = list(selecao)
            return
        if i >= m:
            return
        # Poda por custo e por viabilidade (arestas restantes insuficientes)
        if custo >= melhor["custo"]:
            return
        if (n - 1 - len(selecao)) > (m - i):
            return
        # Ramo 1: inclui a aresta i
        selecao.append(arestas[i])
        backtrack(i + 1, selecao, custo + arestas[i][2])
        selecao.pop()
        # Ramo 2: exclui a aresta i
        backtrack(i + 1, selecao, custo)

    backtrack(0, [], 0.0)

    return ResultadoMST(
        arestas=melhor["arestas"] or [],
        custo_total=(melhor["custo"] if melhor["arestas"] else float("inf")),
        arvores_avaliadas=contadores["arvores"],
        chamadas_recursivas=contadores["chamadas"],
    )


__all__ = [
    "LIMITE_FORCA_BRUTA",
    "ResultadoCaminho",
    "ResultadoMST",
    "todos_os_caminhos",
    "todas_arvores_geradoras",
]
