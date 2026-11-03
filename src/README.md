# Referência da implementação

[Projeto](../README.md) · [Teoria](../docs/teoria/README.md) · [Execução](../docs/execucao/README.md)

O arquivo [`oRF.py`](oRF.py) concentra as estruturas de nós, duas árvores, duas florestas e a demonstração sintética. [`GeradorCSV.py`](GeradorCSV.py) treina o modelo com um NPZ e exporta predições. Os arquivos originais foram preservados nesta etapa de documentação.

## Classes e responsabilidades

| Classe | Estado ou configuração | Responsabilidade |
|---|---|---|
| `Node` | `feature_i_star`, `th_star`, `left`, `right`, `label` | Representar um nó ortogonal ou uma folha |
| `DecisionTreeClassifier(max_depth=100)` | `root`, `max_depth` | Treinar e percorrer a árvore ortogonal |
| `RF(k=10, p=0.6, s=0.1)` | `trees`, `features_list` | Treinar árvores ortogonais com amostragem e agregar seus votos |
| `ObliqueNode` | `w`, `mean`, `std`, `th_star`, `left`, `right`, `label` | Guardar parâmetros de um nó oblíquo ou o rótulo de uma folha |
| `ObliqueDecisionTreeClassifier(max_depth=10)` | `root`, `max_depth` | Treinar e percorrer a árvore com SVD local |
| `oRF(k=250, p=0.63, s=0.27)` | `trees`, `features_list` | Treinar árvores oblíquas e combinar seus votos |

`Node` e `ObliqueNode` usam `label is not None` para identificar folhas durante a predição. Os índices de atributo de uma árvore são relativos ao subconjunto que a floresta lhe entregou.

## Métodos das árvores

| Método | Comportamento |
|---|---|
| `fit(X, Y)` | Constrói a raiz chamando `build_tree`; retorna `None` |
| `build_tree(X, Y, depth=0)` | Aplica paradas, escolhe a divisão e constrói recursivamente os filhos |
| `get_best_split(X, Y)` | Pesquisa atributo/limiar na árvore ortogonal; direção/limiar e estatísticas locais na oblíqua |
| `entropy_calc(y)` | Calcula entropia de classes em bits |
| `information_gain(parent, l_child, r_child)` | Calcula redução ponderada da entropia |
| `calculate_leaf_label(Y)` | Retorna a classe majoritária; empate favorece a menor classe ordenada |
| `predict(X)` | Percorre a árvore para cada amostra; retorna uma lista Python |
| `make_prediction(x, tree)` | Segue os testes a partir do nó recebido até uma folha |

Na árvore ortogonal, `get_best_split` retorna o índice do atributo e o limiar. Na oblíqua, retorna `w_star, mean, std, th_star`. Se não há candidato válido, o índice fica em `-1` na ortogonal ou `w_star` fica `None` na oblíqua, levando à criação de uma folha.

## Construção de uma árvore oblíqua

1. Verificar pureza, profundidade e número de amostras.
2. Calcular média e desvio padrão das colunas do nó; trocar desvios nulos por 1.
3. Padronizar os dados e calcular SVD reduzida.
4. Examinar até três linhas de `Vt` e todos os valores únicos de cada projeção como limiares.
5. Escolher o maior ganho de informação entre os cortes com dois filhos não vazios.
6. Particionar as linhas originais do nó e construir os filhos recursivamente.
7. Salvar `w`, `mean`, `std` e o limiar no nó interno.

Na predição, o nó calcula `((x - mean) / std) @ w`. Valores menores ou iguais ao limiar vão à esquerda; valores maiores, à direita. A amostra não é substituída pela projeção ao descer na árvore: o filho precisa consultar suas próprias características e estatísticas.

## Paradas, empates e detalhes relevantes

- A raiz começa na profundidade 0. Quando `depth >= max_depth`, o nó vira folha.
- Um nó com menos de dez amostras não tenta dividir. Isso não impõe dez amostras por filho.
- A pureza é verificada pela existência de uma única classe.
- A ausência de um corte válido, ou de filhos não vazios após a divisão, gera uma folha.
- Os limiares são valores observados, não médias entre valores consecutivos.
- A escolha usa comparação estrita `>`; em empate de ganho, permanece o primeiro candidato.
- Não existe limiar mínimo de ganho: um corte de ganho zero pode ser escolhido.
- O laço de SVD usa `min(3, Vt.shape[0])`; o comentário que fala em cinco componentes está desatualizado.
- O padrão de `oRF.p` é `0.63`; o comentário que menciona 60% não altera esse valor.

## Funcionamento das florestas

`fit(X_train, Y_train)` limpa as listas `trees` e `features_list`. Para cada árvore, sorteia `int(p * n)` índices de linhas com reposição e `int(s * d)` atributos sem reposição. Não há reamostragem de atributos a cada nó.

`predict(X_test)` seleciona os atributos registrados para cada árvore, reúne uma matriz de votos com forma `(k, n_test)` e aplica `mode(..., axis=0, keepdims=False).mode`. O resultado é um array NumPy com uma classe por amostra. Na versão verificada de SciPy, empates numéricos escolhem o menor rótulo.

As árvores são criadas com `DecisionTreeClassifier()` ou `ObliqueDecisionTreeClassifier()`. Portanto, as profundidades efetivas são respectivamente 100 e 10. `RF` e `oRF` não aceitam `max_depth`, `random_state` ou `n_jobs` como argumentos.

## Contrato de entrada e saída

| Entrada | Requisito para uso confiável |
|---|---|
| `X_train` | Array NumPy numérico finito, bidimensional e não vazio, forma `(n, d)` |
| `Y_train` | Array numérico finito unidimensional, forma `(n,)`, com rótulos de classes |
| `X_test` | Array numérico finito, forma `(n_test, d)`, mesmas colunas e mesma ordem do treino |
| `k` | Inteiro positivo |
| `p` | Valor positivo com `int(p * n) >= 1`; em experimentos de fração da base, usar `p <= 1` |
| `s` | Fração com `0 < s <= 1` e `int(s * d) >= 1` |

Essas condições não são verificadas de forma abrangente pelas classes. Um subconjunto de treino muito pequeno pode gerar apenas folhas majoritárias. Com os padrões, `oRF` precisa de ao menos quatro atributos para selecionar uma coluna; `RF` precisa de ao menos dez. Para bases menores, ajuste `s` conscientemente.

`fit` modifica o objeto e retorna `None`: use `model.fit(X, y)` e depois `model.predict(X_test)`, sem encadear a chamada. As classes não implementam a interface completa dos estimadores de scikit-learn, `predict_proba`, serialização própria ou controle interno de sementes.

## Pontos de entrada

### `python src/oRF.py`

O `main()` cria 100 amostras com dez características, cinco informativas e `random_state=42`. Separa 30% para teste com `random_state=42`, treina `oRF()` e imprime acurácia com quatro casas decimais. A divisão original não solicita estratificação. A floresta usa a aleatoriedade global de NumPy.

### `python src/GeradorCSV.py`

O utilitário carrega `data/data.npz`, lê as três chaves esperadas, treina `oRF()` e salva `submissions/submission.csv`. Todos os caminhos são relativos ao diretório de trabalho. É necessário executar da raiz e criar `submissions/` previamente.

**Esse arquivo não tem proteção `if __name__ == "__main__"`**: importá-lo já executa leitura, treinamento e gravação. Para reutilizar as classes, importe `src.oRF` a partir da raiz do projeto, como nos [exemplos de execução](../docs/execucao/README.md).

## Limitações existentes

Não há tratamento automático de NaN/infinito, atributos categóricos, desbalanceamento, pesos de classe, poda, ganho mínimo, paralelismo, probabilidades, cálculo out-of-bag ou persistência. Os laços explícitos e a repetição de cálculos de entropia podem tornar o treinamento demorado. Essas observações descrevem o estado atual e orientam melhorias futuras.
