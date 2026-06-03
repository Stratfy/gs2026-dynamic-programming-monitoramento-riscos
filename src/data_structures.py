"""
data_structures.py
==================

Estruturas de dados implementadas **do zero** para o Sistema de Detecção de
Incêndios Florestais (FIAP Global Solution 2026 — Stratfy).

Este módulo concentra:

1. Uso explícito e documentado das estruturas nativas do Python exigidas pelo
   enunciado: ``list``, ``tuple``, ``dict``, ``set`` e ``heap`` (``heapq``).
2. ``Node``  — nó de uma Árvore Binária de Busca (BST).
3. ``BinarySearchTree`` — BST construída do zero (sem bibliotecas externas de
   árvore) com: inserir, buscar, buscar_intervalo(r_min, r_max),
   percurso_in_order, altura e remover.
4. ``Grafo`` — grafo ponderado representado por **lista de adjacência**
   (``dict`` de ``list`` de ``tuple``) com ``add_vertice`` / ``add_aresta``.

Modelagem do domínio
--------------------
Cada **vértice** é um município brasileiro com focos de calor agregados (dados
reais do INPE/BDQueimadas 2025). O **índice de risco** do vértice combina o
``risco_fogo_medio`` (modelo do INPE, ∈ [0,1]) e o nº de focos do município
(normalizado em escala log), conforme a fórmula documentada em
``indice_risco()``.

Cada **aresta** liga dois municípios; o **peso** é o tempo estimado de
deslocamento em horas = distância_geodésica_km / 60 (velocidade média de
60 km/h para brigadas em campo). A distância usa a fórmula de *haversine*
sobre os centroides reais (lat/lon) dos municípios.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constantes de domínio
# ---------------------------------------------------------------------------
RAIO_TERRA_KM: float = 6371.0088          # raio médio da Terra (haversine)
VELOCIDADE_BRIGADA_KMH: float = 60.0      # velocidade média de deslocamento
N_FOCOS_REFERENCIA: float = 50000.0       # escala para normalizar nº de focos


# ===========================================================================
# 1. FÓRMULAS DE DOMÍNIO  (distância geodésica e índice de risco)
# ===========================================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância geodésica em km entre dois pontos (fórmula de haversine).

    Recebe coordenadas em graus decimais. Usada para gerar o peso (horas) das
    arestas entre centroides reais de municípios.
    """
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return RAIO_TERRA_KM * c


def horas_viagem(distancia_km: float, velocidade_kmh: float = VELOCIDADE_BRIGADA_KMH) -> float:
    """Converte distância (km) em tempo de viagem (horas) — peso da aresta."""
    return distancia_km / velocidade_kmh


def indice_risco(risco_fogo_medio: float, n_focos: int) -> float:
    """Índice de risco normalizado do vértice (município), ∈ [0, 1].

    Fórmula documentada (combinação ponderada de dois sinais reais do INPE):

        risco_norm   = clamp(risco_fogo_medio, 0, 1)               # já ∈ [0,1]
        focos_norm   = min(1.0, log1p(n_focos) / log1p(REF))        # escala log
        indice_risco = 0.6 * risco_norm + 0.4 * focos_norm

    Justificativa dos pesos:
      * 0.6 para ``risco_fogo_medio``: é o modelo meteorológico do INPE
        (umidade, dias sem chuva, vegetação) — preditor direto de ignição.
      * 0.4 para o volume de focos (em escala logarítmica, pois a distribuição
        é fortemente assimétrica): captura a intensidade histórica observada.
      * REF = ``N_FOCOS_REFERENCIA`` (50.000) satura o componente de volume nos
        municípios extremos (ex.: Colniza/MT com 42.141 focos).
    """
    risco_norm = max(0.0, min(1.0, float(risco_fogo_medio)))
    focos_norm = min(1.0, math.log1p(max(0, n_focos)) / math.log1p(N_FOCOS_REFERENCIA))
    return round(0.6 * risco_norm + 0.4 * focos_norm, 6)


# ===========================================================================
# 2. ENTIDADE DE DOMÍNIO  (Município) — usa tuple/namedtuple-like via dataclass
# ===========================================================================
@dataclass
class Municipio:
    """Vértice do grafo: um município com seus atributos reais agregados."""
    id_municipio: int
    nome: str
    uf: str
    bioma: str
    n_focos: int
    risco_fogo_medio: float
    frp_medio: float
    dias_sem_chuva_medio: float
    lat: float
    lon: float

    @property
    def risco(self) -> float:
        """Índice de risco combinado (∈ [0,1]) deste município."""
        return indice_risco(self.risco_fogo_medio, self.n_focos)

    @property
    def coordenadas(self) -> Tuple[float, float]:
        """Centroide como **tuple** imutável (lat, lon)."""
        return (self.lat, self.lon)


# ===========================================================================
# 3. ÁRVORE BINÁRIA DE BUSCA (BST)  — implementada do zero
# ===========================================================================
@dataclass
class Node:
    """Nó da BST.

    A ordenação da árvore é feita pela ``chave`` (índice de risco do município).
    ``payload`` é um ``dict`` com metadados (nome, UF, id) — útil para relatório.
    """
    chave: float
    payload: Dict[str, object] = field(default_factory=dict)
    esquerda: Optional["Node"] = None
    direita: Optional["Node"] = None

    def __repr__(self) -> str:  # pragma: no cover - apenas depuração
        nome = self.payload.get("nome", "?")
        return f"Node(chave={self.chave:.3f}, mun={nome!r})"


class BinarySearchTree:
    """Árvore Binária de Busca ordenada pelo índice de risco.

    Suporta chaves repetidas (vão para a subárvore direita), pois municípios
    distintos podem ter o mesmo índice de risco. Implementada inteiramente do
    zero — nenhuma biblioteca externa de árvore é usada.
    """

    def __init__(self) -> None:
        self.raiz: Optional[Node] = None
        self._tamanho: int = 0

    # ----- inserção --------------------------------------------------------
    def inserir(self, chave: float, payload: Optional[Dict[str, object]] = None) -> None:
        """Insere uma nova chave mantendo a propriedade da BST."""
        novo = Node(chave=float(chave), payload=payload or {})
        if self.raiz is None:
            self.raiz = novo
        else:
            self._inserir_rec(self.raiz, novo)
        self._tamanho += 1

    def _inserir_rec(self, atual: Node, novo: Node) -> None:
        if novo.chave < atual.chave:
            if atual.esquerda is None:
                atual.esquerda = novo
            else:
                self._inserir_rec(atual.esquerda, novo)
        else:  # >= vai para a direita (permite duplicatas)
            if atual.direita is None:
                atual.direita = novo
            else:
                self._inserir_rec(atual.direita, novo)

    # ----- busca exata -----------------------------------------------------
    def buscar(self, chave: float) -> Optional[Node]:
        """Retorna o primeiro nó com a chave dada, ou ``None``."""
        atual = self.raiz
        while atual is not None:
            if math.isclose(chave, atual.chave, abs_tol=1e-9):
                return atual
            atual = atual.esquerda if chave < atual.chave else atual.direita
        return None

    # ----- busca por intervalo (range query) -------------------------------
    def buscar_intervalo(self, r_min: float, r_max: float) -> List[Node]:
        """Retorna **list** de nós com chave ∈ [r_min, r_max], em ordem crescente.

        Faz poda da árvore: só desce à esquerda se ``r_min < chave`` e à direita
        se ``chave < r_max`` (in-order com pruning) — eficiente em árvores
        balanceadas (O(log n + k), k = nº de resultados).
        """
        resultado: List[Node] = []
        self._buscar_intervalo_rec(self.raiz, r_min, r_max, resultado)
        return resultado

    def _buscar_intervalo_rec(
        self, no: Optional[Node], r_min: float, r_max: float, acc: List[Node]
    ) -> None:
        if no is None:
            return
        if r_min < no.chave:
            self._buscar_intervalo_rec(no.esquerda, r_min, r_max, acc)
        if r_min <= no.chave <= r_max:
            acc.append(no)
        if no.chave < r_max:
            self._buscar_intervalo_rec(no.direita, r_min, r_max, acc)

    # ----- percurso in-order ----------------------------------------------
    def percurso_in_order(self) -> List[Node]:
        """Percurso in-order: devolve **list** de nós em ordem crescente de chave."""
        acc: List[Node] = []
        self._in_order_rec(self.raiz, acc)
        return acc

    def _in_order_rec(self, no: Optional[Node], acc: List[Node]) -> None:
        if no is None:
            return
        self._in_order_rec(no.esquerda, acc)
        acc.append(no)
        self._in_order_rec(no.direita, acc)

    # ----- altura ----------------------------------------------------------
    def altura(self) -> int:
        """Altura da árvore (nº de arestas no caminho raiz→folha mais profunda).

        Convenção: árvore vazia = -1; árvore só com raiz = 0.
        """
        return self._altura_rec(self.raiz)

    def _altura_rec(self, no: Optional[Node]) -> int:
        if no is None:
            return -1
        return 1 + max(self._altura_rec(no.esquerda), self._altura_rec(no.direita))

    # ----- remoção ---------------------------------------------------------
    def remover(self, chave: float) -> bool:
        """Remove a primeira ocorrência da chave. Retorna ``True`` se removeu."""
        self.raiz, removeu = self._remover_rec(self.raiz, chave)
        if removeu:
            self._tamanho -= 1
        return removeu

    def _remover_rec(self, no: Optional[Node], chave: float) -> Tuple[Optional[Node], bool]:
        if no is None:
            return None, False
        if math.isclose(chave, no.chave, abs_tol=1e-9):
            # Caso 1 e 2: zero ou um filho
            if no.esquerda is None:
                return no.direita, True
            if no.direita is None:
                return no.esquerda, True
            # Caso 3: dois filhos → substitui pelo sucessor in-order (menor da direita)
            sucessor = self._minimo(no.direita)
            no.chave = sucessor.chave
            no.payload = sucessor.payload
            no.direita, _ = self._remover_rec(no.direita, sucessor.chave)
            return no, True
        if chave < no.chave:
            no.esquerda, removeu = self._remover_rec(no.esquerda, chave)
        else:
            no.direita, removeu = self._remover_rec(no.direita, chave)
        return no, removeu

    @staticmethod
    def _minimo(no: Node) -> Node:
        while no.esquerda is not None:
            no = no.esquerda
        return no

    # ----- utilitários -----------------------------------------------------
    def __len__(self) -> int:
        return self._tamanho

    def esta_balanceada(self) -> bool:
        """Heurística: True se altura <= 2*log2(n+1) (apenas informativo)."""
        if self._tamanho <= 1:
            return True
        return self.altura() <= 2 * math.log2(self._tamanho + 1)


# ===========================================================================
# 4. GRAFO  — lista de adjacência (dict de listas)
# ===========================================================================
class Grafo:
    """Grafo ponderado **não-direcionado** em lista de adjacência.

    Estruturas internas (uso explícito das exigidas no enunciado):
      * ``self.adjacencia`` — **dict**: vértice_id -> **list** de **tuple**
        ``(vizinho_id, peso_horas)``  → a própria lista de adjacência.
      * ``self.vertices``   — **dict**: vértice_id -> objeto ``Municipio``.
      * ``self._arestas``   — **set** de **tuple** ``(min_id, max_id)`` para
        evitar arestas duplicadas (deduplicação O(1)).

    Lista de adjacência foi escolhida (em vez de matriz) porque o grafo de
    proximidade entre municípios é **esparso**: cada município conecta-se
    apenas a seus k vizinhos mais próximos, então |E| = O(k·|V|) ≪ |V|².
    """

    def __init__(self, nome: str = "grafo") -> None:
        self.nome = nome
        self.adjacencia: Dict[int, List[Tuple[int, float]]] = {}
        self.vertices: Dict[int, Municipio] = {}
        self._arestas: Set[Tuple[int, int]] = set()

    # ----- construção ------------------------------------------------------
    def add_vertice(self, municipio: Municipio) -> None:
        """Adiciona um vértice (município). Idempotente."""
        if municipio.id_municipio not in self.vertices:
            self.vertices[municipio.id_municipio] = municipio
            self.adjacencia[municipio.id_municipio] = []

    def add_aresta(self, u: int, v: int, peso: Optional[float] = None) -> float:
        """Adiciona aresta não-direcionada u—v.

        Se ``peso`` for ``None``, calcula automaticamente o peso em horas a
        partir da distância geodésica entre os centroides reais dos vértices.
        Retorna o peso usado. Ignora laços e arestas duplicadas.
        """
        if u == v or u not in self.vertices or v not in self.vertices:
            return 0.0
        chave = (min(u, v), max(u, v))
        if chave in self._arestas:
            return 0.0
        if peso is None:
            mu, mv = self.vertices[u], self.vertices[v]
            peso = horas_viagem(haversine_km(mu.lat, mu.lon, mv.lat, mv.lon))
        peso = float(peso)
        self.adjacencia[u].append((v, peso))
        self.adjacencia[v].append((u, peso))
        self._arestas.add(chave)
        return peso

    # ----- consultas -------------------------------------------------------
    def vizinhos(self, u: int) -> List[Tuple[int, float]]:
        """Lista de adjacência de u: list de tuple (vizinho, peso)."""
        return self.adjacencia.get(u, [])

    def num_vertices(self) -> int:
        return len(self.vertices)

    def num_arestas(self) -> int:
        return len(self._arestas)

    def ids(self) -> List[int]:
        """Lista (ordenada) dos ids dos vértices."""
        return sorted(self.vertices.keys())

    def arestas(self) -> List[Tuple[int, int, float]]:
        """Devolve list de tuple (u, v, peso) — cada aresta uma única vez."""
        out: List[Tuple[int, int, float]] = []
        for (u, v) in sorted(self._arestas):
            # recupera o peso da lista de adjacência
            peso = next(p for (w, p) in self.adjacencia[u] if w == v)
            out.append((u, v, peso))
        return out

    def conectar_k_vizinhos(self, k: int = 3) -> None:
        """Constrói arestas conectando cada vértice aos seus ``k`` mais próximos.

        Usa um **heap** (heapq) por vértice para extrair os k menores pesos —
        demonstra o uso de heap exigido no enunciado. Garante grafo conexo ao
        adicionar uma aresta de "espinha" entre vértices consecutivos quando
        necessário.
        """
        ids = self.ids()
        for u in ids:
            mu = self.vertices[u]
            # heap de (distancia, v) para todos os outros vértices
            monte: List[Tuple[float, int]] = []
            for v in ids:
                if v == u:
                    continue
                mv = self.vertices[v]
                d = haversine_km(mu.lat, mu.lon, mv.lat, mv.lon)
                heapq.heappush(monte, (d, v))
            # extrai os k menores do heap
            for _ in range(min(k, len(monte))):
                _, v = heapq.heappop(monte)
                self.add_aresta(u, v)
        # garante conectividade: liga componentes via "espinha" ordenada por risco
        self._garantir_conexo()

    def _garantir_conexo(self) -> None:
        """Liga componentes desconexas adicionando arestas de menor distância."""
        comps = self.componentes_conexas()
        while len(comps) > 1:
            # liga a componente 0 à componente 1 pela aresta mais curta entre elas
            c0, c1 = comps[0], comps[1]
            melhor: Optional[Tuple[float, int, int]] = None
            for u in c0:
                mu = self.vertices[u]
                for v in c1:
                    mv = self.vertices[v]
                    d = haversine_km(mu.lat, mu.lon, mv.lat, mv.lon)
                    if melhor is None or d < melhor[0]:
                        melhor = (d, u, v)
            if melhor is None:
                break
            self.add_aresta(melhor[1], melhor[2])
            comps = self.componentes_conexas()

    def componentes_conexas(self) -> List[Set[int]]:
        """Componentes conexas via BFS (usa set de visitados e list como fila)."""
        visitados: Set[int] = set()
        comps: List[Set[int]] = []
        for inicio in self.ids():
            if inicio in visitados:
                continue
            comp: Set[int] = set()
            fila: List[int] = [inicio]
            visitados.add(inicio)
            while fila:
                atual = fila.pop()
                comp.add(atual)
                for (viz, _) in self.adjacencia[atual]:
                    if viz not in visitados:
                        visitados.add(viz)
                        fila.append(viz)
            comps.append(comp)
        return comps

    def eh_conexo(self) -> bool:
        return len(self.componentes_conexas()) <= 1

    # ----- serialização ----------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        """Serializa o grafo para um dict JSON-friendly (data/processed)."""
        return {
            "nome": self.nome,
            "vertices": [
                {
                    "id": m.id_municipio,
                    "nome": m.nome,
                    "uf": m.uf,
                    "bioma": m.bioma,
                    "n_focos": m.n_focos,
                    "risco_fogo_medio": m.risco_fogo_medio,
                    "frp_medio": m.frp_medio,
                    "dias_sem_chuva_medio": m.dias_sem_chuva_medio,
                    "lat": m.lat,
                    "lon": m.lon,
                    "indice_risco": m.risco,
                }
                for m in self.vertices.values()
            ],
            "arestas": [
                {"u": u, "v": v, "peso_horas": round(p, 4)}
                for (u, v, p) in self.arestas()
            ],
        }


__all__ = [
    "haversine_km",
    "horas_viagem",
    "indice_risco",
    "Municipio",
    "Node",
    "BinarySearchTree",
    "Grafo",
    "RAIO_TERRA_KM",
    "VELOCIDADE_BRIGADA_KMH",
    "N_FOCOS_REFERENCIA",
]
