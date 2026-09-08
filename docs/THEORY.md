# RAG Topológico: A Teoria das Bacias de Atração no BasinRAG

Este documento delineia a fundamentação matemática subjacente ao **BasinRAG**, elucidando as propriedades de partições estruturais em grafos funcionais quando comparadas a grafos de co-ocorrência estocásticos.

## 1. Dinâmica de Conectividade em Grafos Baseados em Co-ocorrência

Sistemas de RAG baseados em grafos de conhecimento frequentemente mapeiam relações a partir de extração de entidades e co-ocorrências locais. Em corpora densos, essa modelagem encontra desafios de distribuição de grau em nós centrais, fenômeno conhecido na teoria de redes complexas como **"Hub Congestion"** ou **"Hub Collapse"**.

### 1.1 Concentração em Hubs Hiperconectados

Em procedimentos de extração automatizada sobre grandes corpora, conceitos genéricos, identificadores utilitários ou termos de alta frequência tendem a acumular um volume desproporcional de arestas de co-ocorrência, formando *hubs hiperconectados*.

Quando métodos de difusão ou caminhadas aleatórias (como variantes de PageRank) operam sobre tais estruturas sem restrição de fluxo, a distribuição de probabilidade estacionária pode convergir predominantemente para esses hubs. Isso atenua o sinal dos nós mais específicos (cauda longa), diluindo o contexto em direção a nós de alta centralidade estatística.

### 1.2 Dispersão em Caminhadas Multi-hop (*Noise Drift*)

Em grafos densamente conectados, a expansão de vizinhança por múltiplos saltos (*multi-hop*) tende a atravessar pontes criadas pelos nós centrais, dispersando a busca para fragmentos contextualmente desvinculados do objetivo da consulta (*Multi-hop Noise Drift*). Esse efeito pode comprometer a precisão da recuperação ao integrar trechos semanticamente heterogêneos.

## 2. Dinâmica de Bacias de Atração

O BasinRAG mitiga a dispersão contextual modelando o espaço de busca através de partições estruturais orientadas pela própria topologia do documento original.

Ao invés de derivar a topologia estocasticamente pelo vocabulário, definimos um espaço mapeado onde a estrutura de leitura gera a métrica fundamental de ancoramento semântico. Cada sub-árvore da documentação flui logicamente para suas conclusões ou raízes conceituais. 

Nós denotamos a estrutura de **Bacias de Atração** onde todo segmento menor inevitavelmente se funde sob a âncora do seu ancestral estrutural (o Sumidouro, $A_i$). 
Desta forma, a contração para atratores confina o raio de expansão da pesquisa. Quando uma busca intra-bacia aciona as vizinhanças, ela é restrita ao limite $B(A_i)$. O confinamento determinístico preserva a lógica do autor, assegurando isolamento semântico transversal. A busca não se dissipa aleatoriamente por tokens de alta centralidade léxica.

## 3. Árvores $\rho$ e Decaimento Exponencial de Hops

Como não dependemos de extração probabilística de grafos densos, a árvore de predecessores intrafamiliar $\rho$ nos entrega uma coordenada exata de quão profundo é o nó pesquisado perante seu centro contextual.

Introduzimos a função de profundidade topológica $h(v)$, representando a contagem de saltos necessários na árvore $\rho$ para atingir o atrator a partir do fragmento $v$.

Para a agregação de score durante a fase Híbrida do BasinRAG, aplicamos o prior de suavização exponencial topológica:

$$ S_{\text{topológico}} = \exp(-h(v) \cdot \lambda) $$

**Justificativa Matemática:** A constante de decaimento $\lambda$ regula a taxa de penalização semântica por desvio da âncora central. Fragmentos dispersos no limiar de uma seção longa $h \gg 1$ requerem uma confiança vetorial crua $S_{\text{RRF}}$ substancialmente superior para superarem o prior da raiz $A_i$, onde $h(A_i) = 0$. Esse prior age como um regularizador bayesiano local contra fragmentação contextual e previne o Multi-hop Noise Drift.

## 4. Comparação Formal de Complexidade

Na literatura de recuperação estruturada em grafos, podemos analisar o comportamento assintótico de modelos probabilísticos comunitários em contraste com grafos funcionais determinísticos:

**Modelos Baseados em Clustering e Centralidade em Grafos de Co-ocorrência**
Para identificar agrupamentos comunitários, tais abordagens utilizam rotinas de particionamento hierárquico (Leiden ou Louvain) ou cálculo de centralidade de autovetor.
O limite inferior para essas rotinas em grafos reais é tipicamente $O(|E| \log |V|)$, podendo alcançar $O(|V|^2)$ em redes densas, gerando custo de construção proporcionalmente elevado em grandes corpora.

**BasinRAG (Grafo Funcional Estrito)**
A geração da subestrutura baseia-se num Grafo Funcional $\phi: V \to V \cup \{\emptyset\}$. A geração da partição das bacias de atração restringe-se inteiramente ao processamento topológico seqüencial. Por conseguinte:
* O número de arestas restringe-se estritamente a $|E| \le |V|$.
* A construção requer um mapeamento direto de predecessores lineares. 

Resultando em complexidade temporal de construção $O(N)$ em relação ao volume de ingestão, com garantia determinística de partição e preservação do fluxo local com baixa latência.