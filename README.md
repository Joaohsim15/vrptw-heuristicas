# VRPTW Multiobjetivo — INF0415 (Heurísticas e Modelagem Multiobjetivo)

Roteamento de Veículos com Janelas de Tempo (VRPTW), formulação multiobjetivo,
conforme a `Proposta_VRP.pdf` da equipe (Problema P1).

Instância usada: **Solomon C101**, reduzida aos **25 primeiros clientes**
(`data/C101.txt`), conforme sugerido na proposta para manter o custo
computacional dentro do orçamento de tempo das entregas.

## Estrutura

```
src/
  instance.py    -> leitura da instância Solomon + modelagem (variáveis, restrições,
                    decodificação giant-tour -> rotas, objetivos f1/f2/f3)
  baseline.py    -> heurística clássica Clarke & Wright (savings)         [R4]
  ga.py          -> Algoritmo Genético single-objective (DEAP)           [R2 - Checkpoint 1]
  nsga2.py       -> NSGA-II multiobjetivo (pymoo), com suporte a warm-start [R2/R3 - Checkpoint 2 e 3]
  utils.py       -> funções de plot (rotas, convergência, fronteira de Pareto)

scripts/
  checkpoint1_modelagem.py     -> entrega "Projeto final - modelagem completa"
  checkpoint2_multiobjetivo.py -> entrega "Projeto final - Versão multiobjetivo"
  checkpoint3_pre_final.py     -> entrega "Projeto final - Pré-final" (+ diferencial)

data/
  C101.txt       -> instância Solomon original (formato .txt clássico)

results/
  cp1_*, cp2_*, cp3_*  -> CSVs e figuras gerados por cada checkpoint
  summary.json         -> resumo numérico de todos os resultados

report/
  relatorio_tecnico.md -> relatório técnico (primeira versão) - Checkpoint 3
```

## Como rodar

```bash
pip install -r requirements.txt

python3 scripts/checkpoint1_modelagem.py      # baseline + GA (5 sementes)
python3 scripts/checkpoint2_multiobjetivo.py  # NSGA-II, fronteira de Pareto inicial
python3 scripts/checkpoint3_pre_final.py      # versão integrada + diferencial warm-start
```

Cada script é independente e salva seus artefatos em `results/`. Os parâmetros
(seeds, tamanho de população, nº de gerações) ficam no topo de cada arquivo.

## Resumo dos resultados (instância C101, 25 clientes)

| Método                                   | f1 (distância) | f2 (veículos) |
|-------------------------------------------|----------------:|---------------:|
| Baseline Clarke-Wright                   |          556.51 |             10 |
| GA single-objective (média de 5 sementes)|          306.01 |            4.4 |
| GA single-objective (melhor semente)     |          266.08 |              4 |
| NSGA-II, fronteira inicial (30 gerações) |     483.7–524.0 |          7 a 9 |
| NSGA-II + warm-start (80 gerações)       |          218.09 |              4 |

## Diferencial (Checkpoint 3)

**Warm-start (D2)**: em vez de começar o NSGA-II com população 100% aleatória,
25% dos indivíduos iniciais são "semeados" com a rota do baseline Clarke-Wright
e com a melhor solução do GA single-objective. Isso acelera fortemente a
convergência (ver `results/cp3_warmstart_convergence.png` e o relatório técnico).

## Limitações conhecidas (documentadas no relatório)

- A fusão de rotas no Clarke-Wright só é tentada em uma orientação fixa
  (não testa a rota invertida), o que deixa o baseline mais fraco do que a
  versão "completa" do algoritmo — é um baseline simples e intencionalmente
  conservador, não o estado da arte.
- f3 (desbalanceamento de carga, diferencial ético D6) já está implementado em
  `instance.py` e disponível em `nsga2.py` (`use_f3_ethics=True`), mas por
  padrão os checkpoints usam (f1, f2), conforme a proposta original.
