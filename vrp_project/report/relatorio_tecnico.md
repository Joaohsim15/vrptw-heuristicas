# Relatório Técnico (Primeira Versão)
## Roteamento de Veículos Multiobjetivo: VRPTW com GA, NSGA-II e diferencial de warm-start

**Disciplina:** INF0415 — Heurísticas e Modelagem Multiobjetivo · UFG · 2026/1
**Equipe:** Henrique Matheus Mendonça de Miranda, Eduardo Dias Peixoto, Vitor Luiz Caldas de Sousa,
Antonio Carlos de Barcelos Fernandes, João Henrique Félix Simielli

---

## 1. Introdução

Este relatório documenta a evolução do projeto de roteamento de veículos com janelas de
tempo (VRPTW) em sua formulação multiobjetivo, conforme a proposta inicial do grupo (Problema
P1). Cobre os três checkpoints implementados até aqui: (i) modelagem completa + baseline + GA
single-objective; (ii) versão multiobjetivo com NSGA-II; (iii) versão integrada com o diferencial
de **warm-start**. É a primeira versão do documento — focada em registrar formulação, método e
resultados preliminares; discussão mais profunda de trade-offs entra na versão final.

## 2. Formulação do problema

**Entrada:** depósito único, frota de veículos idênticos de capacidade *Q*, e um conjunto de
clientes com demanda, coordenadas e janela de atendimento `[e_i, l_i]`.

**Decisão / representação:** ao invés de variáveis binárias `x_ijk` explícitas, a solução é
codificada como um **giant-tour** — uma permutação dos clientes, sem marcadores de rota. Um
procedimento de *split* guloso decodifica essa permutação em rotas viáveis: percorre a
permutação inserindo clientes na rota atual e abre uma nova rota (a partir do depósito) sempre
que o próximo cliente violaria a capacidade do veículo ou sua janela de tempo. Essa escolha
evita ter que evoluir explicitamente o número de rotas, simplificando os operadores genéticos.

**Restrições garantidas pela representação/decodificação:**
- cada cliente é visitado exatamente uma vez (é uma permutação);
- a carga acumulada de cada rota nunca excede *Q*;
- toda rota inicia e termina no depósito;
- chegadas antes de `e_i` esperam; chegadas após `l_i` são tratadas como infactíveis e forçam
  a abertura de uma nova rota (penalização "estrutural", em vez de penalização na função objetivo).

**Objetivos:**
- **f1** — distância total percorrida pela frota;
- **f2** — número de veículos (rotas) utilizados;
- **f3** — desbalanceamento de carga entre veículos (desvio-padrão da ocupação), diferencial
  ético (D6); já implementado e disponível, mas mantido desativado por padrão nesta fase,
  conforme o escopo definido na proposta (foco em f1 e f2).

## 3. Metodologia

### 3.1 Instância

Solomon **C101**, restrita aos **25 primeiros clientes** (`data/C101.txt`). Essa redução segue
a própria proposta do grupo, que cita instâncias de 25–100 clientes como viáveis em laptop
comum dentro do orçamento de tempo das entregas.

### 3.2 Baseline — Clarke & Wright (savings)

Implementação clássica do algoritmo de savings: cada cliente começa em sua própria rota; pares
de rotas são fundidos em ordem decrescente de economia (`savings = d(0,i) + d(0,j) − d(i,j)`),
desde que a fusão preserve viabilidade de capacidade e de janela de tempo. **Limitação conhecida
e documentada:** esta implementação tenta a fusão apenas em uma orientação fixa das rotas
(extremidade final de uma rota com a extremidade inicial da outra), sem testar a rota invertida.
Isso é uma simplificação intencional do algoritmo completo e explica por que o baseline obtido
(10 veículos, distância 556,5) é sensivelmente mais fraco do que as metaheurísticas — é um ponto
de partida simples e conservador, não o estado da arte do Clarke-Wright.

### 3.3 Metaheurística single-objective — Algoritmo Genético (DEAP)

- Representação: giant-tour (permutação).
- Fitness (minimização): `f1 + 50 × f2` — f2 entra como penalização linear, como definido na
  proposta para o Checkpoint 1.
- Operadores: crossover OX (`cxOrdered`), mutação por troca de índices (`mutShuffleIndexes`,
  indpb=0.05), seleção por torneio (k=3), elitismo (melhor indivíduo sempre preservado).
- Parâmetros: população=120, 150 gerações, cxpb=0.8, mutpb=0.2.
- Execução com **5 sementes** (1 a 5) para análise de variância.

### 3.4 Versão multiobjetivo — NSGA-II (pymoo)

Mesma representação (permutação); operadores de permutação nativos do pymoo (Order Crossover +
Inversion Mutation). Otimiza (f1, f2) simultaneamente, sem agregação em escalar. Usado tanto para
a fronteira de Pareto inicial (Checkpoint 2, 30 gerações) quanto para a versão integrada
(Checkpoint 3, 80 gerações).

### 3.5 Diferencial — Warm-start (D2)

Em vez de inicializar a população do NSGA-II 100% aleatoriamente, **25% dos indivíduos** da
população inicial são "semeados" com soluções já conhecidas como boas — a rota do baseline
Clarke-Wright e a melhor solução do GA single-objective (melhor entre as 5 sementes) — e o
restante (75%) continua aleatório, preservando diversidade genética. A justificativa é simples
de explicar: começar a busca evolutiva perto de soluções razoáveis (em vez de partir do zero)
costuma acelerar fortemente a convergência, e é isso que os experimentos confirmam.

## 4. Resultados

### 4.1 Checkpoint 1 — Baseline vs. GA (5 sementes)

| Método                       | f1 (distância) | f2 (veículos) |
|-------------------------------|----------------:|---------------:|
| Baseline Clarke-Wright        |          556,51 |             10 |
| GA — semente 1                |          343,57 |              4 |
| GA — semente 2                |          303,02 |              4 |
| GA — semente 3                |          309,50 |              5 |
| GA — semente 4                |          307,89 |              5 |
| GA — semente 5 (melhor)       |          266,08 |              4 |
| **GA — média (5 sementes)**   |  **306,01 ± 24,62** |  **4,4 ± 0,49** |

O GA supera o baseline em todas as sementes, tanto em distância (~45% menor, em média) quanto
em número de veículos. A variância entre sementes é moderada (desvio ≈ 8% da média em f1),
indicando alguma sensibilidade à inicialização aleatória — esperado em GA sem reinício múltiplo.

Figuras: `results/cp1_baseline_routes.png`, `results/cp1_ga_convergence.png`,
`results/cp1_ga_best_routes.png`.

### 4.2 Checkpoint 2 — Fronteira de Pareto inicial (NSGA-II, 30 gerações)

| f1 (distância) | f2 (veículos) |
|----------------:|---------------:|
|          483,67 |              9 |
|          494,24 |              8 |
|          524,03 |              7 |

Com apenas 30 gerações, a fronteira do NSGA-II ainda está dominada pelas soluções do GA
single-objective (que rodou 150 gerações) — todas as soluções do GA têm f1 menor para f2
comparável ou menor. Isso é esperado: é uma fronteira **inicial**, não convergida, e ilustra o
trade-off direto entre tempo de busca e qualidade da fronteira (ver Seção 5).

Figura: `results/cp2_pareto_comparison.png`.

### 4.3 Checkpoint 3 — Efeito do warm-start (NSGA-II, 80 gerações)

| Configuração                     | f1 (geração 0) | f1 (geração 80) |
|------------------------------------|----------------:|------------------:|
| NSGA-II aleatório (sem diferencial)|           686,27 |            424,20 |
| NSGA-II + warm-start (D2)          |           266,08 |            218,09 |

Com warm-start, a população já nasce competitiva (266 vs. 686 de distância na geração 0) e
termina as 80 gerações quase **2× melhor** que a versão aleatória (218 vs. 424) — superando
inclusive o melhor resultado do GA single-objective do Checkpoint 1 (266,08), agora também com
f2 = 4 veículos. A versão aleatória, mesmo após 80 gerações, ainda não alcança a qualidade do
baseline combinado com GA.

Figuras: `results/cp3_warmstart_convergence.png`, `results/cp3_pareto_warmstart.png`.

## 5. Discussão de trade-offs (preliminar)

- **Distância vs. número de veículos:** reduzir f1 tende a concentrar clientes em menos rotas
  (sobrecarregando veículos até o limite de capacidade/tempo), enquanto reduzir f2 normalmente
  alonga as rotas restantes — o conflito previsto na proposta se confirma nos dados (compare,
  por ex., as soluções do GA com f2=4 vs. f2=5 no Checkpoint 1).
- **Tempo de busca vs. qualidade da fronteira:** o NSGA-II com 30 gerações (Checkpoint 2) é
  claramente inferior ao GA com 150 gerações; isso não é uma falha do método multiobjetivo, mas
  reflete o orçamento de gerações exigido para cada checkpoint. A versão de 80 gerações com
  warm-start (Checkpoint 3) já reverte essa desvantagem.
- **Custo de implementação vs. ganho:** o warm-start foi o diferencial mais simples de
  implementar entre os disponíveis (hibridização, HPO, XAI, ética) e, ainda assim, produziu o
  maior salto de qualidade observado no projeto até agora — um bom indício de que a fronteira de
  Pareto final (Checkpoint de entrega final) deve se beneficiar de combiná-lo com a hibridização
  memética (SA + 2-opt) já planejada na proposta.

## 6. Limitações e próximos passos

- Clarke-Wright simplificado (sem testar rota invertida na fusão) — próxima versão pode
  implementar a fusão bidirecional completa para um baseline mais forte.
- f3 (equilíbrio de carga, diferencial ético) está implementado mas não usado nos experimentos
  reportados — pode ser ativado (`use_f3_ethics=True`) para a entrega final, gerando uma
  fronteira de Pareto 3D.
- Métricas de qualidade multiobjetivo (hypervolume, IGD) ainda não foram calculadas; ficam
  para a versão final, junto com a hibridização SA+2-opt e a comparação estatística mais
  rigorosa (testes de significância entre sementes).

## 7. Referências

DEB, K. et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Trans.
Evolutionary Computation*, v. 6, n. 2, p. 182–197, 2002.

SOLOMON, M. M. Algorithms for the vehicle routing and scheduling problems with time window
constraints. *Operations Research*, v. 35, n. 2, p. 254–265, 1987.

CLARKE, G.; WRIGHT, J. W. Scheduling of vehicles from a central depot to a number of delivery
points. *Operations Research*, v. 12, n. 4, p. 568–581, 1964.

TALBI, E.-G. *Metaheuristics: From Design to Implementation*. Hoboken: Wiley, 2009.

EIBEN, A. E.; SMITH, J. E. *Introduction to Evolutionary Computing*. 2. ed. Berlin: Springer, 2015.
