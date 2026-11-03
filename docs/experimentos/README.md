# Como avaliar desempenho e custo

[Projeto](../../README.md) · [Teoria](../teoria/README.md) · [Relatório](../relatorio/README.md)

O enunciado pede uma comparação de habilidade preditiva e custo computacional com uma floresta ortogonal. Esta página propõe um protocolo reproduzível. **Ainda não há um estudo comparativo completo ou pontuações oficiais do Kaggle documentados no projeto.** O script de exemplo usa dados sintéticos e serve para verificar o fluxo de medição.

## Perguntas que o experimento deve responder

- Qual acurácia cada modelo obtém sobre as mesmas amostras de teste?
- Quanto tempo leva para treinar e para predizer o mesmo conjunto?
- Como os resultados variam com número de árvores, amostras e atributos?
- Há ganho preditivo suficiente para justificar o custo adicional?

## Comparação justa

| Fator | Padrão de `RF` | Padrão de `oRF` | Consequência |
|---|---:|---:|---|
| Árvores `k` | 10 | 250 | Tempo e estabilidade de votos diferem |
| Amostragem `p` | 0,60 | 0,63 | Quantidade de sorteios de treino difere |
| Atributos `s` | 0,10 | 0,27 | Dimensão de entrada das árvores difere |
| Profundidade | 100 | 10 | Capacidade e custo das árvores diferem |
| Direções por nó | Atributos disponíveis | Até três direções de SVD | Estratégia de busca difere |

Executar os dois padrões mede o desempenho de **duas configurações completas**. Não isola o efeito de usar hiperplanos oblíquos. Igualar `k`, `p` e `s` ajuda, mas não elimina a diferença de profundidade: os construtores das florestas não expõem esse parâmetro.

Para um estudo controlado da geometria, uma alternativa já suportada é comparar `DecisionTreeClassifier(max_depth=h)` e `ObliqueDecisionTreeClassifier(max_depth=h)` sobre os mesmos dados. Isso é uma comparação de **árvores individuais**, não de florestas. Uma comparação de florestas com profundidade controlada exigiria uma extensão do código ou um experimento próprio que explicite essa alteração.

## Protocolo recomendado

1. Defina o conjunto e registre tamanho, número de atributos, classes e tratamento de dados.
2. Reserve uma avaliação final antes da escolha de hiperparâmetros. Para classificação com classes suficientes, use divisão estratificada.
3. Ajuste pré-processamentos somente no treino de cada divisão. A PCA dos nós já é ajustada dentro do `fit`.
4. Selecione configurações usando validação interna, sem consultar repetidamente a avaliação final.
5. Execute cada configuração em várias sementes e registre resultados por execução, não só o melhor.
6. Meça `fit` e `predict` separadamente, com o mesmo hardware, dados e condições de execução.
7. Apresente média e dispersão de acurácia e tempo. Interprete diferenças considerando as configurações e a variabilidade.

O teste sem rótulos fornecido para submissão não permite calcular acurácia local. Separe uma validação a partir de `X_train` e `y_train`. A métrica oficial da competição deve ser consultada na página do Kaggle; a acurácia é uma medida ilustrativa explicitamente contemplada pelo enunciado.

## Medição executável

```bash
python docs/experimentos/comparacao_sintetica.py
```

O [script](comparacao_sintetica.py) usa o mesmo conjunto sintético e a mesma divisão estratificada para os modelos, com `k=10`, `p=0.7` e `s=0.5`. Reinicia a semente antes de cada treino. No código atual, isso alinha os sorteios de linhas e colunas, pois as árvores não fazem sorteios adicionais. As profundidades permanecem 100 e 10.

A saída informa modelo, configuração, acurácia, segundos de treinamento e segundos de predição. **É uma execução exploratória com uma semente; não sustenta uma conclusão geral sobre qual modelo é melhor.** Repita o protocolo em diferentes bases e sementes para produzir resultados de relatório. Tempos dependem da máquina e da carga concorrente.

## Medidas

$$
\operatorname{acurácia}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\{\widehat{y}_i=y_i\}.
$$

O tempo de treinamento é a diferença de `time.perf_counter()` ao redor de `fit`; o de inferência, ao redor de `predict`. Para comparar lotes de tamanhos distintos, pode-se reportar tempo por amostra, informando também o tamanho do lote. Classes desbalanceadas podem exigir métricas complementares, como acurácia balanceada e matriz de confusão, escolhidas conforme a tarefa.

## Modelo de tabela para resultados reais

Preencha somente após executar e guardar os resultados. “A medir” não significa zero.

| Modelo/configuração | Base e divisão | Sementes | Acurácia média ± desvio | Treino (s) | Predição (s) |
|---|---|---|---|---|---|
| RF — informar `k`, `p`, `s`, profundidade | A definir | A definir | A medir | A medir | A medir |
| oRF — informar `k`, `p`, `s`, profundidade | A definir | A definir | A medir | A medir | A medir |

Registre também CPU, memória, sistema operacional, Python, dependências, número efetivo de atributos (`int(s*d)`) e número de sorteios de linhas (`int(p*n_train)`). Documente como escolheu os hiperparâmetros.

## Investigação sugerida

Experimente variar um fator por vez: `k` para estudar custo da agregação; `p` para alterar a amostra de cada árvore; `s` para alterar a dimensão das projeções; e tamanho/dimensionalidade da base sintética para estudar escala. Mantenha os demais fatores explícitos. As primeiras três direções de PCA podem ser insuficientes quando atributos de menor variância concentram informação de classe; uma análise desse caso pode explicar falhas sem depender apenas da acurácia agregada.
