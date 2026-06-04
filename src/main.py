"""
main.py
=======

Orquestrador do pipeline da disciplina **Dynamic Programming** (Estruturas de
Dados e Algoritmos) — Stratfy / FIAP Global Solution 2026.

Etapas:
  1. Carrega os dados reais (focos de calor agregados por município, INPE 2025).
  2. Monta 2 cenários: B (MATOPIBA) e D (Amazônia).
  3. Para cada cenário:
       - constrói a BST por índice de risco e demonstra busca por intervalo;
       - roda o Guloso (Prim para MST e Dijkstra a partir do hub de maior risco);
       - extrai um subgrafo N<=12 e roda a Força Bruta como oráculo;
       - serializa grafo + MST em data/processed/*.json.
  4. Roda o benchmark de desempenho (N = 5,8,10,12,20,50,100).
  5. Gera as figuras em report/figuras/.

Executável: ``python src/main.py`` (a partir da raiz do repositório).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

# garante import dos módulos irmãos quando rodado de qualquer cwd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# garante saída UTF-8 no console (Windows usa cp1252 por padrão e quebraria
# com caracteres como ∈, →, índice de risco etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):  # pragma: no cover
    pass

from brute_force import todas_arvores_geradoras, todos_os_caminhos
from data_structures import BinarySearchTree, Grafo
from greedy import dijkstra, hub_otimo, prim_mst
from loader import (
    carregar_municipios,
    cenario_amazonia,
    cenario_matopiba,
    subgrafo_forca_bruta,
)
from performance_monitor import DIR_PROCESSED, imprimir_tabela, rodar_benchmark
from visualizations import (
    desenhar_bst,
    desenhar_gap,
    desenhar_grafo_mst,
    desenhar_tempo_vs_n,
)


def _linha(titulo: str) -> None:
    print("\n" + "=" * 74)
    print(titulo)
    print("=" * 74)


def construir_bst_do_grafo(grafo: Grafo) -> BinarySearchTree:
    """Insere todos os municípios do grafo em uma BST chaveada pelo risco."""
    bst = BinarySearchTree()
    for m in grafo.vertices.values():
        bst.inserir(
            chave=m.risco,
            payload={"id": m.id_municipio, "nome": m.nome, "uf": m.uf,
                     "bioma": m.bioma, "n_focos": m.n_focos},
        )
    return bst


def processar_cenario(nome_curto: str, grafo: Grafo) -> Dict[str, object]:
    """Roda BST + Guloso + Força Bruta para um cenário e serializa resultados."""
    _linha(f"CENÁRIO: {grafo.nome}  ({grafo.num_vertices()} municípios, "
           f"{grafo.num_arestas()} arestas, conexo={grafo.eh_conexo()})")

    # ---- BST + busca por intervalo (municípios de risco alto) ----
    bst = construir_bst_do_grafo(grafo)
    in_order = bst.percurso_in_order()
    print(f"BST: n={len(bst)} | altura={bst.altura()} | "
          f"risco mín={in_order[0].chave:.3f} máx={in_order[-1].chave:.3f}")
    criticos = bst.buscar_intervalo(0.70, 1.0)
    print(f"Municípios em estado CRÍTICO (índice de risco ∈ [0.70, 1.00]): "
          f"{len(criticos)}")
    for no in criticos[:5]:
        print(f"   - {no.payload['nome']:<22} ({no.payload['uf']}) "
              f"risco={no.chave:.3f} focos={no.payload['n_focos']}")

    # ---- Guloso: Prim (MST) ----
    mst = prim_mst(grafo)
    print(f"\nPrim (MST): {len(mst.arestas)} arestas | custo total "
          f"{mst.custo_total:.2f} h | {mst.operacoes} operações de heap")

    # ---- Guloso: Dijkstra a partir do hub de maior risco ----
    hub = hub_otimo(grafo)
    nome_hub = grafo.vertices[hub].nome
    dij = dijkstra(grafo, hub)
    alcancaveis = [d for d in dij.distancias.values() if d != float("inf")]
    mais_distante = max(dij.distancias, key=lambda k: dij.distancias[k]
                        if dij.distancias[k] != float("inf") else -1)
    print(f"Dijkstra (hub = {nome_hub}, maior risco): alcança "
          f"{len(alcancaveis)} municípios | tempo máx até um vértice "
          f"{dij.distancias[mais_distante]:.2f} h")
    caminho = dij.caminho_ate(mais_distante)
    nomes_caminho = " → ".join(grafo.vertices[v].nome for v in caminho)
    print(f"   Rota mais longa do hub: {nomes_caminho}")

    # ---- Força Bruta: subgrafo N<=10 como oráculo ----
    sub = subgrafo_forca_bruta(grafo, n_max=10)
    fb = todas_arvores_geradoras(sub)
    mst_sub = prim_mst(sub)
    coincide = abs(fb.custo_total - mst_sub.custo_total) < 1e-6
    print(f"\nForça Bruta (subgrafo N={sub.num_vertices()}): "
          f"avaliou {fb.arvores_avaliadas} árvores geradoras em "
          f"{fb.chamadas_recursivas} chamadas recursivas")
    print(f"   custo ótimo (FB) = {fb.custo_total:.4f} h | "
          f"custo Prim = {mst_sub.custo_total:.4f} h | "
          f"{'COINCIDEM (Prim é exato)' if coincide else 'DIVERGEM!'}")

    # caminho mínimo por força bruta vs Dijkstra (validação extra)
    ids_sub = sub.ids()
    o, d = ids_sub[0], ids_sub[-1]
    fb_cam = todos_os_caminhos(sub, o, d)
    dij_sub = dijkstra(sub, o)
    coincide_cam = abs(fb_cam.melhor_custo - dij_sub.distancias[d]) < 1e-6
    print(f"   caminho mín {sub.vertices[o].nome}→{sub.vertices[d].nome}: "
          f"FB={fb_cam.melhor_custo:.4f} h | Dijkstra={dij_sub.distancias[d]:.4f} h | "
          f"{'OK' if coincide_cam else 'DIVERGEM!'}")

    # ---- serialização ----
    os.makedirs(DIR_PROCESSED, exist_ok=True)
    payload_grafo = grafo.to_dict()
    payload_grafo["mst"] = {
        "arestas": [{"u": u, "v": v, "peso_horas": round(p, 4)}
                    for (u, v, p) in mst.arestas],
        "custo_total_horas": mst.custo_total,
    }
    payload_grafo["hub_dijkstra"] = {
        "id": hub, "nome": nome_hub,
        "distancias_horas": {str(k): v for k, v in dij.distancias.items()},
    }
    caminho_json = os.path.join(DIR_PROCESSED, f"grafo_{nome_curto}.json")
    with open(caminho_json, "w", encoding="utf-8") as fh:
        json.dump(payload_grafo, fh, ensure_ascii=False, indent=2)
    print(f"\n[salvo] {caminho_json}")

    # serializa a BST (percurso in-order) também
    caminho_bst = os.path.join(DIR_PROCESSED, f"bst_{nome_curto}.json")
    with open(caminho_bst, "w", encoding="utf-8") as fh:
        json.dump({
            "altura": bst.altura(),
            "n": len(bst),
            "in_order": [{"risco": round(no.chave, 4), **no.payload}
                         for no in in_order],
        }, fh, ensure_ascii=False, indent=2)
    print(f"[salvo] {caminho_bst}")

    # ---- figuras ----
    fig_mst = desenhar_grafo_mst(grafo, mst,
                                 titulo=grafo.nome.replace("_", " "),
                                 nome_arquivo=f"grafo_{nome_curto}_mst.png")
    print(f"[figura] {fig_mst}")

    # BST reduzida (10-15 nós de maior risco) para legibilidade do diagrama.
    # A ordem de inserção é embaralhada (seed fixa) — inserir já ordenado por
    # risco degeneraria a BST em uma lista encadeada (altura = n-1). Embaralhar
    # produz uma árvore representativa e ilustra que a forma da BST depende da
    # ordem de inserção.
    import random as _rnd
    top_munis = sorted(grafo.vertices.values(), key=lambda m: m.risco,
                       reverse=True)[:13]
    ordem = list(top_munis)
    _rnd.Random(42).shuffle(ordem)
    bst_fig = BinarySearchTree()
    for m in ordem:
        bst_fig.inserir(m.risco, {"nome": m.nome, "uf": m.uf})
    fig_bst = desenhar_bst(bst_fig, titulo=f"BST — {grafo.nome.replace('_', ' ')}",
                           nome_arquivo=f"bst_{nome_curto}.png")
    print(f"[figura] {fig_bst}")

    return {
        "cenario": grafo.nome,
        "n_vertices": grafo.num_vertices(),
        "n_arestas": grafo.num_arestas(),
        "mst_custo": mst.custo_total,
        "fb_coincide": coincide,
        "caminho_coincide": coincide_cam,
    }


def main() -> None:
    _linha("STRATFY — FIAP Global Solution 2026 | Dynamic Programming")
    print("Sistema de Detecção de Incêndios Florestais — Monitoramento de Riscos")
    print("Dados reais: INPE/BDQueimadas 2025 (3.466.399 focos agregados).")

    municipios = carregar_municipios()
    print(f"\nMunicípios carregados (com centroide): {len(municipios)}")

    resumos: List[Dict[str, object]] = []

    # Cenário B — MATOPIBA
    g_mato = cenario_matopiba(municipios, n_nos=24, k_vizinhos=3)
    resumos.append(processar_cenario("cenarioB_matopiba", g_mato))

    # Cenário D — Amazônia
    g_amaz = cenario_amazonia(municipios, n_nos=24, k_vizinhos=3)
    resumos.append(processar_cenario("cenarioD_amazonia", g_amaz))

    # ---- Benchmark de desempenho ----
    _linha("BENCHMARK DE DESEMPENHO (N = 5, 8, 10, 12, 20, 50, 100)")
    # usa o conjunto MATOPIBA (995 municípios) como base de amostragem por N
    base = sorted(
        [m for m in municipios if m.uf in {"MA", "TO", "PI", "BA"}],
        key=lambda m: m.n_focos, reverse=True,
    )
    linhas = rodar_benchmark(base, salvar=True)
    imprimir_tabela(linhas)

    fig_tempo = desenhar_tempo_vs_n(linhas)
    fig_gap = desenhar_gap(linhas)
    print(f"\n[figura] {fig_tempo}")
    print(f"[figura] {fig_gap}")

    # ---- resumo geral ----
    resumo_geral = {
        "projeto": "Stratfy - Dynamic Programming - GS2026",
        "tema": "Sistema de Deteccao de Incendios Florestais",
        "cenarios": resumos,
        "validacao_forca_bruta": all(r["fb_coincide"] and r["caminho_coincide"]
                                     for r in resumos),
    }
    caminho_resumo = os.path.join(DIR_PROCESSED, "resumo_execucao.json")
    with open(caminho_resumo, "w", encoding="utf-8") as fh:
        json.dump(resumo_geral, fh, ensure_ascii=False, indent=2)

    _linha("CONCLUÍDO")
    print(f"[salvo] {caminho_resumo}")
    print("Validação cruzada Força Bruta == Guloso em todos os cenários: "
          f"{resumo_geral['validacao_forca_bruta']}")
    print("Artefatos: data/processed/*.json e report/figuras/*.png")


if __name__ == "__main__":
    main()
