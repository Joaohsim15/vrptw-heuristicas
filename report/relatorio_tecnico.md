# Relatório Técnico Final
## Roteamento de Veículos Multiobjetivo: VRPTW com GA, NSGA-II e diferencial de warm-start

**Disciplina:** INF0415 — Heurísticas e Modelagem Multiobjetivo · UFG · 2026/1
**Equipe:** Henrique Matheus Mendonça de Miranda, Eduardo Dias Peixoto, Vitor Luiz Caldas de Sousa,
Antonio Carlos de Barcelos Fernandes, João Henrique Félix Simielli

---

## 1. Introdução

Este relatório documenta o projeto de roteamento de veículos com janelas de tempo (VRPTW) em
sua formulação multiobjetivo, conforme a proposta inicial do grupo (Problema P1). O projeto
foi desenvolvido em três checkpoints progressivos: (i) modelagem completa + baseline + GA
single-objective; (ii) versão multiobjetivo com NSGA-II; (iii) versão integrada com o diferencial
de warm-start, cujos resultados são consolidados e expandidos nesta entrega final com o cálculo
do **Hypervolume** como métrica de qualidade da fronteira de Pareto.

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
  ético (D6); implementado e disponível em `src/nsga2.py` (`use_f3_ethics=True`), mantido
  desativado por padrão conforme o escopo definido na proposta (foco em f1 e f2).

## 3. Metodologia

### 3.1 Instância

Solomon **C101**, restrita aos **25 primeiros clientes** (`data/C101.txt`). Essa redução segue
a própria proposta do grupo, que cita instâncias de 25–100 clientes como viáveis em laptop
comum dentro do orçamento de tempo das entregas. Os parâmetros da instância são: capacidade de
veículo Q = 200, janelas de tempo com horizonte total de 1236 unidades de tempo.

### 3.2 Baseline — Clarke & Wright (savings)

Implementação clássica do algoritmo de savings (`src/baseline.py`): cada cliente começa em sua
própria rota; pares de rotas são fundidos em ordem decrescente de economia
(`savings = d(0,i) + d(0,j) − d(i,j)`), desde que a fusão preserve viabilidade de capacidade
e de janela de tempo.

**Limitação conhecida e documentada:** esta implementação tenta a fusão apenas em uma
orientação fixa das rotas (extremidade final de uma rota com a extremidade inicial da outra),
sem testar a rota invertida. Isso é uma simplificação intencional do algoritmo completo e explica
por que o baseline obtido é sensivelmente mais fraco do que as metaheurísticas — é um ponto de
partida simples e conservador, não o estado da arte do Clarke-Wright.

### 3.3 Metaheurística single-objective — Algoritmo Genético (DEAP)

- **Representação:** giant-tour (permutação).
- **Fitness (minimização):** `f1 + 50 × f2` — f2 entra como penalização linear, conforme
  definido na proposta para o Checkpoint 1.
- **Operadores:** crossover OX (`cxOrdered`), mutação por troca de índices
  (`mutShuffleIndexes`, indpb=0.05), seleção por torneio (k=3), elitismo (melhor indivíduo
  sempre preservado).
- **Parâmetros:** população=120, 150 gerações, cxpb=0.8, mutpb=0.2.
- **Execução:** 5 sementes (1 a 5) para análise de variância.

### 3.4 Versão multiobjetivo — NSGA-II (pymoo)

Mesma representação (permutação); operadores de permutação nativos do pymoo (Order Crossover +
Inversion Mutation). Otimiza (f1, f2) simultaneamente, sem agregação em escalar. A fronteira de
Pareto resultante representa o conjunto de soluções não-dominadas — soluções para as quais não
existe outra que seja melhor em ambos os objetivos ao mesmo tempo.

Parâmetros: população=120, crossover OX, mutação por inversão, eliminação de duplicatas.

### 3.5 Diferencial — Warm-start (D2)

Em vez de inicializar a população do NSGA-II 100% aleatoriamente, **25% dos indivíduos** da
população inicial são "semeados" com soluções já conhecidas como boas — a rota do baseline
Clarke-Wright e a melhor solução do GA single-objective (seed 5) — e o restante (75%) continua
aleatório, preservando diversidade genética. A justificativa é que começar a busca evolutiva
perto de soluções razoáveis costuma acelerar fortemente a convergência, confirmado pelos
experimentos.

### 3.6 Métrica de qualidade — Hypervolume (HV)

O **Hypervolume** é a métrica padrão para avaliar fronteiras de Pareto multiobjetivo. Dado um
ponto de referência *r* (escolhido como pior que todas as soluções observadas), o HV mede o
volume (área em 2D) do espaço de objetivos que é dominado pela fronteira e delimitado por *r*.
Quanto **maior** o HV, melhor a fronteira — ela domina mais da região de interesse, capturando
tanto a convergência (soluções próximas ao ótimo) quanto a diversidade (soluções bem distribuídas).

Ponto de referência utilizado: **r = (750, 12)** — claramente pior que qualquer solução
observada nos experimentos (f1 máximo observado ≈ 686; f2 máximo observado = 10).

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
indicando alguma sensibilidade à inicialização aleatória.

Figuras: `results/final_baseline_routes.png`, `results/final_ga_convergence.png`,
`results/final_ga_best_routes.png`.

### 4.2 Checkpoint 2 — Fronteira de Pareto inicial (NSGA-II, 30 gerações)

| f1 (distância) | f2 (veículos) |
|----------------:|---------------:|
|          483,67 |              9 |
|          494,24 |              8 |
|          524,03 |              7 |

**Hypervolume (CP2, 30 gerações, sem warm-start): 1.280,72**

Com apenas 30 gerações, a fronteira do NSGA-II ainda está dominada pelas soluções do GA
single-objective (que rodou 150 gerações). Isso não é uma falha do método multiobjetivo, mas
reflete o orçamento de gerações exigido para cada checkpoint.

Figura: `results/cp2_pareto_comparison.png`.

### 4.3 Checkpoint 3 — Efeito do warm-start (NSGA-II, 80 gerações)

| Configuração                     | f1 (geração 0) | f1 (geração 80) |
|------------------------------------|----------------:|------------------:|
| NSGA-II aleatório (sem diferencial)|           686,27 |            424,20 |
| NSGA-II + warm-start (D2)          |           266,08 |            218,09 |

Com warm-start, a população já nasce competitiva (f1=266 na geração 0, vs. 686 sem warm-start)
e termina as 80 gerações quase **2× melhor** em f1. Estes valores são do registro histórico do
Checkpoint 3 (`scripts/checkpoint3_pre_final.py`); por ser estocástico, o NSGA-II pode produzir
resultados ligeiramente distintos em execuções independentes — a análise consolidada e
reproduzível da entrega final está na Seção 4.4.
Figuras: `results/cp3_warmstart_convergence.png`.

### 4.4 Entrega Final — Comparação consolidada e Hypervolume

Esta entrega executa a pipeline completa em `scripts/checkpoint_final.py` e calcula o
Hypervolume para todas as frentes geradas, permitindo comparação objetiva.

| Configuração                             | HV (ref. = [750, 12]) | f1 mín | f2 mín |
|------------------------------------------|----------------------:|-------:|-------:|
| NSGA-II, 30 gen, sem warm-start (CP2)    |               1.280,7 |  483,7 |      7 |
| NSGA-II, 80 gen, sem warm-start          |               1.901,8 |  420,5 |      6 |
| **NSGA-II, 80 gen, COM warm-start (D2)** |           **3.871,4** |**266,1**|    **4**|

O warm-start mais que **dobra o Hypervolume** (+103,6%) em relação à versão aleatória com o
mesmo número de gerações. A fronteira final com warm-start inclui soluções com f1=266 e f2=4,
superando inclusive o melhor resultado do GA single-objective (que também atingiu f1=266, mas
via otimização escalar, sem garantia de diversidade na fronteira).

Figura: `results/final_pareto_comparison.png`, `results/final_convergence.png`.

## 5. Discussão de trade-offs

- **Distância vs. número de veículos:** reduzir f1 tende a concentrar clientes em menos rotas
  (sobrecarregando veículos até o limite de capacidade/tempo), enquanto reduzir f2 normalmente
  alonga as rotas restantes. O conflito previsto na proposta se confirma nos dados: soluções com
  f2=4 têm f1 maior que soluções "teóricas" sem restrição de veículos, e vice-versa.

- **Tempo de busca vs. qualidade da fronteira:** o NSGA-II com 30 gerações (CP2, HV=1.280) é
  claramente inferior ao de 80 gerações (CP3, HV=1.902). O warm-start amplifica esse ganho de
  forma não-linear: com o mesmo orçamento de 80 gerações, o HV passa de 1.902 para 3.871 —
  um salto de 103% apenas por melhorar a inicialização.

- **Custo de implementação vs. ganho:** o warm-start foi o diferencial com melhor
  custo-benefício dentre os disponíveis. A implementação exigiu ~15 linhas de código extra
  (`SeededPermutationSampling` em `src/nsga2.py`) e nenhuma modificação nos operadores
  evolutivos, produzindo o maior salto de qualidade do projeto.

- **Single-objective vs. multiobjetivo:** o GA single-objective (f1 + 50×f2) reporta apenas
  um ponto — a melhor solução escalar encontrada (f1=266, f2=4). O NSGA-II, mesmo na versão sem
  warm-start, produz uma fronteira com soluções alternativas como (f1=420, f2=6), que o GA
  escalar nunca encontraria por estar focado em uma única combinação de objetivos. Com warm-start,
  o NSGA-II iguala o melhor f1 do GA (266) com a mesma f2=4, maximizando o hypervolume — o que
  indica que a fronteira domina a maior região possível do espaço de objetivos dentro do orçamento
  de gerações disponível.

## 6. Limitações conhecidas

- **Clarke-Wright simplificado:** a fusão de rotas é tentada apenas em uma orientação fixa.
  A versão completa (com rota invertida) produziria um baseline mais forte, mas a limitação
  está documentada e é intencional para manter o comparativo simples.
- **f3 disponível mas não utilizado:** o objetivo de desbalanceamento de carga (diferencial
  ético D6) está implementado em `src/instance.py` e pode ser ativado em `src/nsga2.py`
  (`use_f3_ethics=True`), gerando uma fronteira de Pareto 3D, mas foi mantido desativado para
  preservar o escopo de (f1, f2) da proposta.
- **Uma única instância:** todos os experimentos usam C101 (25 clientes). Generalização para
  outras instâncias Solomon (R, RC, maiores) não foi testada.
- **Sem hibridização memética:** a combinação NSGA-II + SA/2-opt mencionada na proposta não
  foi implementada; os experimentos indicam que o warm-start já produziu ganhos suficientes
  dentro do orçamento de tempo do projeto.

## 7. Referências

DEB, K. et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Trans.
Evolutionary Computation*, v. 6, n. 2, p. 182–197, 2002.

SOLOMON, M. M. Algorithms for the vehicle routing and scheduling problems with time window
constraints. *Operations Research*, v. 35, n. 2, p. 254–265, 1987.

CLARKE, G.; WRIGHT, J. W. Scheduling of vehicles from a central depot to a number of delivery
points. *Operations Research*, v. 12, n. 4, p. 568–581, 1964.

ZITZLER, E.; THIELE, L. Multiobjective evolutionary algorithms: a comparative case study and
the strength Pareto approach. *IEEE Trans. Evolutionary Computation*, v. 3, n. 4, p. 257–271,
1999.

TALBI, E.-G. *Metaheuristics: From Design to Implementation*. Hoboken: Wiley, 2009.

EIBEN, A. E.; SMITH, J. E. *Introduction to Evolutionary Computing*. 2. ed. Berlin: Springer, 2015.
