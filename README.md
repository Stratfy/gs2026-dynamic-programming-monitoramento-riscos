# Monitoramento de Riscos Ambientais com Árvores, Grafos e Algoritmos

### Dynamic Programming (Estruturas de Dados e Algoritmos) — FIAP Global Solution 2026

Entregável do grupo **Stratfy** (Turma 2ESPH, Engenharia de Software) para a Global
Solution 2026 (tema macro: **Economia Espacial**). Tema unificador do grupo: **Sistema de
Detecção de Incêndios Florestais** — focos de calor por satélite + clima + desmatamento no
Brasil (2025).

Este módulo aplica **estruturas de dados** (Árvore Binária de Busca e Grafo) e **algoritmos**
(força bruta exaustiva, guloso de Prim e Dijkstra) ao problema real de **triagem de risco** e
**roteamento de brigadas** de combate a incêndios, usando dados reais do INPE/BDQueimadas
2025 (3.466.399 focos agregados em 5.512 municípios).

---

## Integrantes

| Integrante | RM |
|---|---|
| Anthony Sforzin | RM562096 |
| Luigi Mendes Cabrini | RM563552 |
| Rogério Cruz Arroyo | RM563517 |
| Bruno Koeke | RM561309 |

---

## Descrição

Cada **município** é um vértice de um grafo, posicionado pelo seu **centroide geográfico
real** (lat/lon). O **índice de risco** do vértice combina o `risco_fogo_medio` do INPE
(∈ [0,1]) e o nº de focos (em escala logarítmica):

```
indice_risco = 0.6 · clamp(risco_fogo_medio, 0, 1) + 0.4 · min(1, log1p(n_focos) / log1p(50000))
```

Cada **aresta** liga municípios próximos; o **peso** é o tempo estimado de deslocamento em
horas, obtido da **distância geodésica (haversine)** entre os centroides reais dividida pela
velocidade média de uma brigada (60 km/h):

```
peso_horas = distancia_haversine_km / 60
```

O grafo é construído por **k-vizinhos mais próximos** (k=3), resultando em um grafo
**esparso e conexo** — exatamente o cenário em que a *lista de adjacência* supera a *matriz*.

### Cenários (dados reais)

- **Cenário B — Triagem de risco no MATOPIBA**: os 24 municípios com mais focos em
  **MA, TO, PI e BA** (fronteira agropecuária do Cerrado, maior pressão de fogo em 2025).
- **Cenário D — Rotas de brigadas na Amazônia**: os 24 municípios do **bioma Amazônia** com
  mais focos (arco do desmatamento — PA, MT, RO, AM).

Para a **força bruta** usa-se um subgrafo com N ≤ 12 (instância pequena, oráculo de validação).

---

## Estrutura do projeto e descrição de cada módulo

```
dynamic-programming/
├── README.md                       Este arquivo
├── requirements.txt                Dependências (networkx, matplotlib, seaborn, pytest)
├── conftest.py                     Coloca src/ no PYTHONPATH para o pytest
├── data/
│   ├── raw/
│   │   └── focos_municipios_agg.csv   Dados reais (5.512 municípios, INPE 2025)
│   └── processed/                  Grafos/árvores serializados (JSON) — gerados pelo build
├── src/
│   ├── data_structures.py          Estruturas do ZERO: Node, BinarySearchTree, Grafo;
│   │                               uso explícito de list/tuple/dict/set/heap; haversine e
│   │                               índice de risco.
│   ├── loader.py                   Carga do CSV real e construção dos cenários (k-vizinhos).
│   ├── brute_force.py              Enumeração exaustiva (recursão + backtracking) p/ N<=12:
│   │                               todas as árvores geradoras e todos os caminhos; contadores
│   │                               de chamadas recursivas e soluções avaliadas. Oráculo.
│   ├── greedy.py                   Prim (MST) e Dijkstra do ZERO com heapq (sem networkx).
│   ├── performance_monitor.py      Tempo (perf_counter, ms), memória (tracemalloc, MB) e
│   │                               operações elementares p/ N=5,8,10,12,20,50,100; gap.
│   ├── visualizations.py           networkx+matplotlib só p/ desenhar: grafo+MST, BST,
│   │                               tempo×N e gap de otimalidade. Salva PNGs.
│   └── main.py                     Orquestra todo o pipeline.
├── notebooks/
│   └── analise_resultados.ipynb    Análise interativa + ESCALA DE DECISÃO (4 níveis).
├── tests/
│   └── test_algorithms.py          pytest: BST, busca por intervalo, Prim==Força Bruta,
│                                   Dijkstra correto.
└── report/
    ├── relatorio_final.md          Relatório técnico (7 seções).
    └── figuras/                    PNGs gerados pelo build.
```

### Estruturas de dados utilizadas (qual / onde / por quê)

| Estrutura | Onde | Por quê |
|---|---|---|
| `list` | listas de adjacência, percursos, caminhos | sequência ordenada e mutável de vizinhos/nós |
| `tuple` | aresta `(vizinho, peso)`, coordenada `(lat, lon)` | par imutável, hashable, leve |
| `dict` | `Grafo.adjacencia` (vértice→vizinhos), `Grafo.vertices`, distâncias do Dijkstra | acesso O(1) por id do município |
| `set` | `Grafo._arestas` (dedup), `na_arvore` (Prim), `finalizados` (Dijkstra) | pertencimento O(1), sem duplicatas |
| `heap` (`heapq`) | fila de prioridade no Prim, no Dijkstra e na seleção de k-vizinhos | extrair o mínimo em O(log n) |
| `Node` / `BinarySearchTree` | triagem por índice de risco | inserir/buscar/intervalo em O(altura) |
| `Grafo` (lista de adjacência) | malha de municípios | grafo esparso |

---

## Instruções de execução

Pré-requisitos: **Python 3.11+** (testado em 3.13). A partir da raiz deste repositório:

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Rodar a suíte de testes (todos devem passar)
python -m pytest -q

# 3. Rodar o pipeline completo (gera data/processed/*.json e report/figuras/*.png)
python src/main.py

# 4. (Opcional) Abrir o notebook de análise interativa
jupyter notebook notebooks/analise_resultados.ipynb
```

> O CSV de dados reais já está em `data/raw/focos_municipios_agg.csv`. Os algoritmos
> (BST, Grafo, Prim, Dijkstra, força bruta) são implementados **do zero** usando apenas a
> biblioteca padrão; `networkx`/`matplotlib` servem **apenas para desenhar** as figuras.

### Saídas esperadas

- `data/processed/grafo_cenarioB_matopiba.json`, `grafo_cenarioD_amazonia.json` — grafos +
  MST + distâncias do Dijkstra.
- `data/processed/bst_*.json` — percurso in-order das BSTs.
- `data/processed/performance.json`, `resumo_execucao.json` — benchmark e validação.
- `report/figuras/*.png` — 6 figuras (grafo+MST de cada cenário, BST de cada cenário,
  tempo×N e gap de otimalidade).

---

## Resultados em destaque (execução real)

- **Validação cruzada**: em todos os cenários, o custo da MST do **Prim coincide
  exatamente** com o ótimo global da **força bruta** (gap = 0%) — Prim é exato.
- **Cenário B (MATOPIBA)**: 24 municípios, 47 arestas, MST = **49,5 h** de deslocamento total.
- **Cenário D (Amazônia)**: 24 municípios, 46 arestas, MST = **83,1 h**.
- **Explosão combinatória**: a força bruta passa de 219 chamadas recursivas (N=5) para
  **685.505** (N=12), enquanto o guloso permanece em dezenas de operações.

---

## Conexão com os ODS

ODS **2** (segurança alimentar / fronteira agropecuária), ODS **9** (infraestrutura e
inovação), ODS **11** (comunidades resilientes) e ODS **13** (ação climática).
