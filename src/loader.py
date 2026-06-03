"""
loader.py
=========

Carregamento dos **dados reais** (focos de calor agregados por município —
INPE/BDQueimadas 2025) e construção dos cenários de análise.

Lê ``data/raw/focos_municipios_agg.csv`` usando apenas a biblioteca padrão
(``csv``), sem depender de pandas. Constrói objetos ``Municipio`` e monta o
``Grafo`` de proximidade (k-vizinhos mais próximos) para cada cenário.
"""

from __future__ import annotations

import csv
import os
from typing import Callable, Dict, List, Optional

from data_structures import Grafo, Municipio

# Caminho do CSV relativo à raiz do repositório
_AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ_REPO = os.path.abspath(os.path.join(_AQUI, ".."))
CSV_PADRAO = os.path.join(RAIZ_REPO, "data", "raw", "focos_municipios_agg.csv")


def _to_float(valor: str, padrao: float = 0.0) -> float:
    valor = (valor or "").strip()
    if valor == "":
        return padrao
    try:
        return float(valor)
    except ValueError:
        return padrao


def _to_int(valor: str, padrao: int = 0) -> int:
    try:
        return int(float(valor))
    except (ValueError, TypeError):
        return padrao


def carregar_municipios(caminho_csv: str = CSV_PADRAO) -> List[Municipio]:
    """Lê o CSV agregado e devolve uma lista de ``Municipio``.

    Linhas sem centroide são descartadas (não podem virar vértices geográficos).
    O ``risco_fogo_medio`` vazio é tratado como 0.0 (município sem leitura
    confiável de risco no período).
    """
    municipios: List[Municipio] = []
    with open(caminho_csv, "r", encoding="utf-8", newline="") as fh:
        leitor = csv.DictReader(fh)
        for linha in leitor:
            lat = linha.get("centroide_lat", "").strip()
            lon = linha.get("centroide_lon", "").strip()
            if lat == "" or lon == "":
                continue
            municipios.append(
                Municipio(
                    id_municipio=_to_int(linha["id_municipio"]),
                    nome=linha["nome_municipio"].strip(),
                    uf=linha["sigla_uf"].strip(),
                    bioma=linha["bioma_predominante"].strip(),
                    n_focos=_to_int(linha["n_focos"]),
                    risco_fogo_medio=_to_float(linha["risco_fogo_medio"]),
                    frp_medio=_to_float(linha["frp_medio"]),
                    dias_sem_chuva_medio=_to_float(linha["dias_sem_chuva_medio"]),
                    lat=float(lat),
                    lon=float(lon),
                )
            )
    return municipios


def selecionar_topo(
    municipios: List[Municipio],
    filtro: Callable[[Municipio], bool],
    limite: int,
    chave=lambda m: m.n_focos,
) -> List[Municipio]:
    """Filtra e devolve os ``limite`` municípios de maior ``chave``."""
    selecionados = [m for m in municipios if filtro(m)]
    selecionados.sort(key=chave, reverse=True)
    return selecionados[:limite]


def construir_grafo(
    municipios: List[Municipio], nome: str, k_vizinhos: int = 3
) -> Grafo:
    """Monta um ``Grafo`` conexo conectando cada vértice a k vizinhos próximos."""
    grafo = Grafo(nome=nome)
    for m in municipios:
        grafo.add_vertice(m)
    grafo.conectar_k_vizinhos(k=k_vizinhos)
    return grafo


# ---------------------------------------------------------------------------
# Cenários exigidos no enunciado
# ---------------------------------------------------------------------------
UFS_MATOPIBA = {"MA", "TO", "PI", "BA"}


def cenario_matopiba(
    municipios: List[Municipio], n_nos: int = 24, k_vizinhos: int = 3
) -> Grafo:
    """Cenário B — Triagem de risco no MATOPIBA (MA, TO, PI, BA).

    Seleciona os ``n_nos`` municípios com mais focos na fronteira agropecuária
    do MATOPIBA (maior pressão de fogo no Cerrado em 2025).
    """
    sel = selecionar_topo(
        municipios, lambda m: m.uf in UFS_MATOPIBA, n_nos
    )
    return construir_grafo(sel, "Cenario_B_MATOPIBA", k_vizinhos)


def cenario_amazonia(
    municipios: List[Municipio], n_nos: int = 24, k_vizinhos: int = 3
) -> Grafo:
    """Cenário D — Rotas de brigadas na Amazônia (bioma Amazônia).

    Seleciona os ``n_nos`` municípios do bioma Amazônia com mais focos
    (arco do desmatamento — Pará, Mato Grosso, Rondônia, Amazonas).
    """
    sel = selecionar_topo(
        municipios, lambda m: m.bioma == "Amazônia", n_nos
    )
    return construir_grafo(sel, "Cenario_D_Amazonia", k_vizinhos)


def subgrafo_forca_bruta(grafo: Grafo, n_max: int = 10) -> Grafo:
    """Extrai um subgrafo com os ``n_max`` vértices de maior risco (N<=12).

    Reconstrói as arestas por k-vizinhos para garantir conectividade no
    subgrafo (usado pela força bruta como instância pequena/oráculo).
    """
    top = sorted(grafo.vertices.values(), key=lambda m: m.risco, reverse=True)[:n_max]
    k = min(3, max(1, len(top) - 1))
    return construir_grafo(top, f"{grafo.nome}_FB{n_max}", k)


__all__ = [
    "CSV_PADRAO",
    "carregar_municipios",
    "selecionar_topo",
    "construir_grafo",
    "cenario_matopiba",
    "cenario_amazonia",
    "subgrafo_forca_bruta",
    "UFS_MATOPIBA",
]
