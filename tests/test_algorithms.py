"""
test_algorithms.py
==================

Suíte de testes (pytest) que valida as estruturas e algoritmos da disciplina:

  (a) a BST mantém a propriedade de ordenação e o percurso in-order é crescente;
  (b) a busca por intervalo [r_min, r_max] devolve exatamente as chaves corretas;
  (c) Força Bruta == Guloso (Prim) no custo da MST em instâncias pequenas;
  (d) Dijkstra está correto em um grafo conhecido (distâncias calculadas à mão).

Também valida remoção na BST, conectividade do grafo e o oráculo de caminho
mínimo (Força Bruta == Dijkstra).
"""

import math
import os
import random
import sys

import pytest

# torna os módulos de src/ importáveis
_AQUI = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(_AQUI, "..", "src"))
sys.path.insert(0, SRC)

from brute_force import todas_arvores_geradoras, todos_os_caminhos  # noqa: E402
from data_structures import (  # noqa: E402
    BinarySearchTree,
    Grafo,
    Municipio,
    haversine_km,
    indice_risco,
)
from greedy import dijkstra, prim_mst  # noqa: E402
from loader import carregar_municipios, cenario_matopiba, subgrafo_forca_bruta  # noqa: E402


# ===========================================================================
# Fixtures / helpers
# ===========================================================================
def _municipio(i: int, lat: float, lon: float, risco: float = 0.5,
               focos: int = 1000) -> Municipio:
    return Municipio(
        id_municipio=i, nome=f"M{i}", uf="XX", bioma="Cerrado",
        n_focos=focos, risco_fogo_medio=risco, frp_medio=10.0,
        dias_sem_chuva_medio=5.0, lat=lat, lon=lon,
    )


def _grafo_conhecido() -> Grafo:
    """Grafo pequeno com pesos explícitos para validar Dijkstra/Prim à mão.

        A --1-- B --2-- C
        |               |
        4               1
        |               |
        D ------ 7 ----- E   (D-E peso 7), B-D peso 5

    Vértices: 1=A 2=B 3=C 4=D 5=E
    """
    g = Grafo("conhecido")
    for i in range(1, 6):
        g.add_vertice(_municipio(i, lat=-10.0 - i, lon=-45.0 - i))
    g.add_aresta(1, 2, 1.0)  # A-B
    g.add_aresta(2, 3, 2.0)  # B-C
    g.add_aresta(1, 4, 4.0)  # A-D
    g.add_aresta(3, 5, 1.0)  # C-E
    g.add_aresta(4, 5, 7.0)  # D-E
    g.add_aresta(2, 4, 5.0)  # B-D
    return g


@pytest.fixture(scope="module")
def grafo_real_pequeno() -> Grafo:
    """Subgrafo real (MATOPIBA) com N<=10 para os testes de oráculo."""
    municipios = carregar_municipios()
    g = cenario_matopiba(municipios, n_nos=18, k_vizinhos=3)
    return subgrafo_forca_bruta(g, n_max=9)


# ===========================================================================
# (a) BST mantém propriedade e in_order é ordenado
# ===========================================================================
def test_bst_in_order_ordenado():
    bst = BinarySearchTree()
    valores = [0.5, 0.2, 0.8, 0.1, 0.3, 0.7, 0.9, 0.25, 0.6]
    for v in valores:
        bst.inserir(v, {"nome": f"m{v}"})
    chaves = [no.chave for no in bst.percurso_in_order()]
    assert chaves == sorted(valores)
    assert len(bst) == len(valores)


def test_bst_propriedade_recursiva():
    """Para todo nó: esquerda < nó <= direita (verificação recursiva)."""
    bst = BinarySearchTree()
    random.seed(7)
    for _ in range(200):
        bst.inserir(round(random.random(), 4))

    def valida(no, lo, hi):
        if no is None:
            return True
        if not (lo <= no.chave <= hi):
            return False
        return (valida(no.esquerda, lo, no.chave)
                and valida(no.direita, no.chave, hi))

    assert valida(bst.raiz, float("-inf"), float("inf"))
    assert bst.altura() >= math.floor(math.log2(len(bst)))  # altura mínima teórica


def test_bst_busca_exata_e_altura():
    bst = BinarySearchTree()
    assert bst.altura() == -1  # árvore vazia
    bst.inserir(0.5)
    assert bst.altura() == 0   # só raiz
    bst.inserir(0.3)
    bst.inserir(0.7)
    assert bst.altura() == 1
    assert bst.buscar(0.3) is not None
    assert bst.buscar(0.99) is None


# ===========================================================================
# (b) busca por intervalo correta
# ===========================================================================
def test_bst_busca_intervalo():
    bst = BinarySearchTree()
    valores = [0.10, 0.22, 0.35, 0.41, 0.55, 0.63, 0.70, 0.82, 0.91, 0.95]
    for v in valores:
        bst.inserir(v)
    achados = [no.chave for no in bst.buscar_intervalo(0.35, 0.70)]
    esperado = [v for v in valores if 0.35 <= v <= 0.70]
    assert achados == sorted(esperado)
    # intervalo fora de tudo
    assert bst.buscar_intervalo(0.96, 1.0) == []
    # intervalo cobrindo tudo
    assert len(bst.buscar_intervalo(0.0, 1.0)) == len(valores)


def test_bst_busca_intervalo_aleatoria():
    bst = BinarySearchTree()
    random.seed(11)
    valores = [round(random.random(), 5) for _ in range(300)]
    for v in valores:
        bst.inserir(v)
    r_min, r_max = 0.3, 0.6
    achados = sorted(no.chave for no in bst.buscar_intervalo(r_min, r_max))
    esperado = sorted(v for v in valores if r_min <= v <= r_max)
    assert achados == esperado


def test_bst_remocao_mantem_ordem():
    bst = BinarySearchTree()
    valores = [0.5, 0.2, 0.8, 0.1, 0.3, 0.7, 0.9]
    for v in valores:
        bst.inserir(v)
    assert bst.remover(0.5) is True   # remove raiz (dois filhos)
    assert bst.buscar(0.5) is None
    assert len(bst) == len(valores) - 1
    chaves = [no.chave for no in bst.percurso_in_order()]
    assert chaves == sorted(chaves)   # continua ordenada
    assert bst.remover(0.123) is False  # inexistente


# ===========================================================================
# (c) Força Bruta == Guloso (Prim) no custo da MST
# ===========================================================================
def test_mst_forca_bruta_igual_prim_grafo_conhecido():
    g = _grafo_conhecido()
    fb = todas_arvores_geradoras(g)
    prim = prim_mst(g)
    assert math.isclose(fb.custo_total, prim.custo_total, abs_tol=1e-9)
    # MST conhecida: A-B(1)+B-C(2)+C-E(1)+A-D(4) = 8.0
    assert math.isclose(prim.custo_total, 8.0, abs_tol=1e-9)
    assert fb.arvores_avaliadas > 0


def test_mst_forca_bruta_igual_prim_dados_reais(grafo_real_pequeno):
    fb = todas_arvores_geradoras(grafo_real_pequeno)
    prim = prim_mst(grafo_real_pequeno)
    assert math.isclose(fb.custo_total, prim.custo_total, rel_tol=1e-9, abs_tol=1e-6)


def test_mst_arestas_formam_arvore():
    g = _grafo_conhecido()
    prim = prim_mst(g)
    # uma MST tem exatamente |V|-1 arestas
    assert len(prim.arestas) == g.num_vertices() - 1


# ===========================================================================
# (d) Dijkstra correto em grafo conhecido
# ===========================================================================
def test_dijkstra_grafo_conhecido():
    g = _grafo_conhecido()
    res = dijkstra(g, origem=1)  # a partir de A
    # distâncias calculadas à mão:
    # A=0; B=1; C=3 (A-B-C); E=4 (A-B-C-E); D=4 (A-D direto, pois A-B-D=6)
    assert math.isclose(res.distancias[1], 0.0)
    assert math.isclose(res.distancias[2], 1.0)
    assert math.isclose(res.distancias[3], 3.0)
    assert math.isclose(res.distancias[5], 4.0)
    assert math.isclose(res.distancias[4], 4.0)
    # caminho A→E reconstruído
    assert res.caminho_ate(5) == [1, 2, 3, 5]


def test_dijkstra_igual_forca_bruta_caminho():
    g = _grafo_conhecido()
    fb = todos_os_caminhos(g, origem=1, destino=5)
    dij = dijkstra(g, origem=1)
    assert math.isclose(fb.melhor_custo, dij.distancias[5], abs_tol=1e-9)
    assert fb.melhor_caminho == dij.caminho_ate(5)


def test_dijkstra_dados_reais_coincide_forca_bruta(grafo_real_pequeno):
    ids = grafo_real_pequeno.ids()
    o, d = ids[0], ids[-1]
    fb = todos_os_caminhos(grafo_real_pequeno, o, d)
    dij = dijkstra(grafo_real_pequeno, o)
    assert math.isclose(fb.melhor_custo, dij.distancias[d], rel_tol=1e-9, abs_tol=1e-6)


# ===========================================================================
# Testes auxiliares: domínio (haversine, índice de risco, conectividade)
# ===========================================================================
def test_haversine_distancia_conhecida():
    # Brasília (-15.78, -47.93) → São Paulo (-23.55, -46.63) ≈ 873 km
    d = haversine_km(-15.78, -47.93, -23.55, -46.63)
    assert 850 < d < 900


def test_indice_risco_monotonia_e_limites():
    assert indice_risco(0.0, 0) == 0.0
    assert 0.0 <= indice_risco(1.0, 50000) <= 1.0
    # mais focos e mais risco ⇒ índice maior
    assert indice_risco(0.9, 40000) > indice_risco(0.3, 100)


def test_grafo_conexo_e_kvizinhos():
    municipios = carregar_municipios()
    g = cenario_matopiba(municipios, n_nos=20, k_vizinhos=3)
    assert g.num_vertices() == 20
    assert g.eh_conexo()
    # cada vértice tem pelo menos 1 vizinho
    assert all(len(g.vizinhos(v)) >= 1 for v in g.ids())
