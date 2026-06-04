# Relatório Técnico — Monitoramento de Riscos Ambientais com Árvores, Grafos e Algoritmos

**Disciplina:** Dynamic Programming (Estruturas de Dados e Algoritmos)
**Projeto:** Sistema de Detecção de Incêndios Florestais — FIAP Global Solution 2026
**Grupo:** Stratfy (Turma 2ESPH, Engenharia de Software)

---

## 1. Identificação (RA / RM — Nome)

| RM | Nome |
|---|---|
| RM562096 | Anthony Sforzin |
| RM563552 | Luigi Mendes Cabrini |
| RM563517 | Rogério Cruz Arroyo |
| RM561309 | Bruno Koeke |

**Tema unificador do grupo:** Sistema de Detecção de Incêndios Florestais (focos de calor por
satélite + clima + desmatamento, Brasil 2025). **Fonte de dados:** INPE/BDQueimadas 2025 —
3.466.399 focos, agregados em 5.512 municípios com centroide geográfico real, índice de risco
de fogo médio, FRP médio e dias sem chuva. A sazonalidade confirma a estação seca como crítica
(pico em **set/2025: 833.039 focos**; out: 823.767; ago: 594.309) e o **Cerrado** como bioma
mais afetado (1.784.865 focos), seguido da **Amazônia** (975.655).

---

## 2. Modelagem (Grafo + BST)

### 2.1 Grafo de municípios

- **Vértice** = município. Atributos reais: nº de focos, `risco_fogo_medio` (∈ [0,1], modelo
  meteorológico do INPE), FRP, dias sem chuva e **centroide (lat/lon)**.
- **Índice de risco do vértice** (combinação normalizada, ∈ [0,1]):

  ```
  risco_norm   = clamp(risco_fogo_medio, 0, 1)
  focos_norm   = min(1, log1p(n_focos) / log1p(50000))
  indice_risco = 0.6 · risco_norm + 0.4 · focos_norm
  ```

  O peso 0,6 prioriza o preditor meteorológico de ignição; o peso 0,4 captura a intensidade
  histórica observada em escala logarítmica (a distribuição de focos é fortemente assimétrica:
  de poucos focos até 42.141 em Colniza/MT).

- **Aresta** = proximidade entre municípios. **Peso = horas de deslocamento** =
  `haversine_km(centroides) / 60` (velocidade média de brigada 60 km/h). O grafo é montado por
  **k-vizinhos mais próximos (k=3)**, garantindo um grafo **esparso e conexo** (uma rotina de
  conectividade liga eventuais componentes pela aresta mais curta entre elas).

### 2.2 Árvore Binária de Busca (BST)

A BST é chaveada pelo **índice de risco** e implementada **do zero** (`Node` +
`BinarySearchTree`): inserir, buscar, **buscar_intervalo(r_min, r_max)** (range query com
poda), percurso in-order, altura e remover (sucessor in-order para nós com dois filhos). Serve
de **mecanismo de triagem**: a busca por intervalo isola em O(log n + k) os municípios em uma
faixa de risco (ex.: ∈ [0,90; 1,00] = estado crítico).

### 2.3 Lista de adjacência vs matriz de adjacência

Adotou-se **lista de adjacência** (`dict` de `list` de `tuple`). Justificativa: o grafo de
proximidade é **esparso** — com k=3, |E| = O(k·|V|) ≈ 3|V|, muito menor que |V|². Comparação:

| Critério | Lista de adjacência | Matriz de adjacência |
|---|---|---|
| Memória | **O(V + E)** | O(V²) |
| Iterar vizinhos de u | **O(grau(u))** | O(V) |
| Testar aresta (u,v) | O(grau(u)) | **O(1)** |
| Adequação (grafo esparso) | **Excelente** | Desperdiça memória |

Como Prim e Dijkstra **iteram sobre os vizinhos** (operação dominante) e o grafo é esparso, a
lista de adjacência minimiza memória e tempo. A matriz só venceria em grafos densos com muitas
consultas pontuais de existência de aresta — não é o caso.

---

## 3. Análise de complexidade teórica

| Operação | Estrutura/Algoritmo | Complexidade | Observação |
|---|---|---|---|
| Inserir / buscar | BST | O(h), h = altura | O(log n) se balanceada; O(n) no pior caso (inserção ordenada) |
| Busca por intervalo | BST | O(h + k) | k = nº de resultados; poda dos ramos fora do intervalo |
| In-order / altura | BST | O(n) | percorre todos os nós |
| MST | **Prim** (heap binário) | **O(E log V)** | cada aresta gera no máx. 1 push/pop no heap |
| Caminho mínimo | **Dijkstra** (heap binário) | **O(E log V)** | pesos não-negativos (tempos) |
| Todas as árvores geradoras | **Força Bruta** | **O(2^E)** (limitado a N≤12) | nº de árvores geradoras ≤ n^(n−2) (Cayley) |
| Todos os caminhos simples | **Força Bruta** | **O(V!)** no pior caso | backtracking com poda por custo |

A força bruta é **exponencial/fatorial** e só é viável para instâncias minúsculas; os
algoritmos gulosos são **polinomiais log-lineares** e escalam para milhares de vértices.

---

## 4. Resultados e figuras

Execução real (`python src/main.py`), dados reais de 2025:

| Cenário | Vértices | Arestas | Conexo | MST (Prim) | Validação FB |
|---|---|---|---|---|---|
| **B — MATOPIBA** | 24 | 47 | sim | **49,5 h** | Prim == ótimo (gap 0%) |
| **D — Amazônia** | 24 | 46 | sim | **83,1 h** | Prim == ótimo (gap 0%) |

- **Triagem (BST)**: no Cenário B, todos os 24 municípios estão em risco ≥ 0,80 (Cerrado em
  plena seca); o índice máximo é **Mirador/MA (0,971)**, escolhido como hub do Dijkstra. A rota
  mais longa a partir do hub leva **18,68 h** (Mirador → … → Formoso do Araguaia).
- **Roteamento (Dijkstra)**: no Cenário D, o hub é **Colniza/MT (0,883)** e a rota mais longa
  atinge **30,38 h** (Colniza → Apuí → Jacareacanga → … → Paragominas), refletindo as enormes
  distâncias da Amazônia frente ao Cerrado.

**Benchmark de desempenho** (`data/processed/performance.json`):

| N | Força Bruta — tempo / chamadas recursivas | Guloso — tempo / operações | Gap |
|---|---|---|---|
| 5 | 0,27 ms / 219 | 0,04 ms / 13 | 0% |
| 8 | 6,0 ms / 5.475 | 0,02 ms / 21 | 0% |
| 10 | 28,8 ms / 33.835 | 0,03 ms / 28 | 0% |
| 12 | 552,7 ms / 685.505 | 0,06 ms / 37 | 0% |
| 20 | inviável | 0,07 ms / 74 | — |
| 50 | inviável | 0,14 ms / 185 | — |
| 100 | inviável | 0,34 ms / 370 | — |

**Figuras** (em `report/figuras/`):
- `grafo_cenarioB_matopiba_mst.png` / `grafo_cenarioD_amazonia_mst.png` — municípios nos
  centroides reais com a MST de Prim destacada em vermelho (tamanho do nó ∝ índice de risco).
- `bst_cenarioB_matopiba.png` / `bst_cenarioD_amazonia.png` — diagrama da BST (13 nós) com
  índices de risco como chaves.
- `tempo_vs_n.png` — tempo × N (Força Bruta vs Guloso, escala log): explosão combinatória da FB.
- `gap_otimalidade.png` — gap de otimalidade do guloso = 0% em todo N viável.

---

## 5. Escala de decisão (com gap)

A escolha do algoritmo segue uma **escala de 4 níveis** que troca *garantia de otimalidade* por
*escalabilidade*, conforme o tamanho do problema (N = nº de municípios a coordenar):

| Nível | Faixa de N | Algoritmo | Qualidade | Custo computacional | Uso típico |
|---|---|---|---|---|---|
| **1 — Exato/Auditoria** | N ≤ 12 | Força Bruta (backtracking) | **Ótimo global** | Alto (≈0,5 s e 6×10⁵ chamadas em N=12) | Auditar o guloso; decisões críticas minúsculas |
| **2 — Guloso exato** | 12 < N ≤ 10³ | Prim + Dijkstra (heap) | **Ótimo (gap 0%)** | Baixo: O(E log V), ms | Planejamento operacional diário (caso padrão) |
| **3 — Guloso escalável** | 10³ < N ≤ 10⁵ | Prim/Dijkstra esparso | Ótimo (exige grafo k-vizinhos) | Médio (E = O(k·V)) | Cobertura estadual/regional |
| **4 — Heurístico** | N > 10⁵ | Clustering + Prim local / A* | Sub-ótimo controlado | Limitado por tempo real | Cobertura nacional em tempo real |

**Leitura do gap:** como Prim e Dijkstra são **exatos**, o gap de otimalidade medido contra a
força bruta é **0% em todo N viável** (níveis 2 e 3 dominam a FB em qualidade *e* custo). Por
isso o guloso é a escolha operacional padrão e a força bruta fica restrita à auditoria (nível 1).
O nível 4 só se justifica quando o tamanho do problema inviabiliza manter E = O(k·V) em tempo
real — aí aceita-se um gap > 0% em troca de resposta imediata.

---

## 6. Conclusão e conexão com os ODS

As estruturas e algoritmos clássicos resolvem com elegância dois subproblemas reais do combate
a incêndios: **(i) triagem** — a BST prioriza municípios por faixa de risco em O(log n + k); e
**(ii) roteamento** — Prim define a malha mínima de deslocamento entre bases e Dijkstra a rota
mais rápida a partir de um hub, ambos **exatos** e de custo O(E log V). A força bruta, embora
inviável em escala (685.505 chamadas já em N=12), cumpriu seu papel de **oráculo**, comprovando
que o guloso atinge o ótimo (gap 0%). Os dados reais de 2025 evidenciam por que o MATOPIBA e o
arco amazônico concentram a demanda: risco de fogo elevado, longas distâncias e pico sazonal na
seca.

**ODS atendidos:** **ODS 2** (segurança alimentar — proteção da fronteira agropecuária do
MATOPIBA contra queimadas), **ODS 9** (infraestrutura e inovação — sistema de apoio à decisão
baseado em dados de satélite), **ODS 11** (comunidades resilientes — priorização e proteção de
municípios) e **ODS 13** (ação climática — resposta rápida a focos de calor e mitigação de
emissões por queimadas).

---

## 7. Referências

- INPE — Programa Queimadas / BDQueimadas. Dados de focos de calor 2025. <https://terrabrasilis.dpi.inpe.br/queimadas/>
- INPE — PRODES / TerraBrasilis. Desmatamento por bioma. <https://terrabrasilis.dpi.inpe.br/>
- IBGE — API de Localidades (malha municipal e centroides). <https://servicodados.ibge.gov.br/api/docs/localidades>
- CORMEN, T. H. et al. *Introduction to Algorithms*. 4ª ed. MIT Press, 2022 (Prim, Dijkstra, BST).
- HAGBERG, A.; SCHULT, D.; SWART, P. *Exploring Network Structure, Dynamics, and Function using NetworkX*. SciPy, 2008.
- ONU — Objetivos de Desenvolvimento Sustentável (ODS 2, 9, 11, 13). <https://brasil.un.org/pt-br/sdgs>
