# Instalação, execução e reprodução

[Projeto](../../README.md) · [Implementação](../../src/README.md) · [Kaggle](../kaggle/README.md)

## Ambiente

O código usa NumPy para arrays, amostragem e SVD; SciPy para entropia e moda; scikit-learn para dados sintéticos, divisão treino/teste e acurácia; e pandas para exportar o CSV. A floresta em si é implementada em `src/oRF.py`.

O ambiente de verificação utilizou Python **3.12.14**, NumPy **2.3.5**, SciPy **1.17.0**, scikit-learn **1.8.0** e pandas **2.2.3**. [`requirements.txt`](../../requirements.txt) registra essas versões diretas. Não é um arquivo de travamento de todas as dependências transitivas nem uma garantia de equivalência numérica entre plataformas.

## Preparar o projeto

```bash
git clone https://github.com/luigiscofano/Kaggle-oRF.git
cd Kaggle-oRF
python -m venv .venv
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

No PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências e execute o exemplo original:

```bash
python -m pip install -r requirements.txt
python src/oRF.py
```

Se o comando `python` não estiver disponível, use o executável de sua instalação, como `python3` ou `py`. Os próximos exemplos pressupõem que `python` aponta para o ambiente preparado. Não adicione `.venv/` em commits; o `.gitignore` original não a inclui.

## Exemplo com semente explícita

```bash
python docs/execucao/exemplo_minimo.py
```

O [arquivo completo](exemplo_minimo.py) usa um conjunto sintético, divisão estratificada e uma semente imediatamente antes do treino. Treina 15 árvores para permitir uma verificação rápida. Essa configuração é diferente dos padrões de `oRF()` e não representa a configuração final da competição.

O núcleo de uso da API é:

```python
import numpy as np
from src.oRF import oRF

# X_train, y_train e X_test devem respeitar o contrato de entrada.
np.random.seed(42)
model = oRF(k=15, p=0.7, s=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

O trecho acima é uma referência de uso e pressupõe arrays já definidos. Para executar um programa completo, use `exemplo_minimo.py`. `fit` retorna `None`, portanto não use `model = oRF().fit(...)`.

## Reprodutibilidade

A semente de `make_classification` controla os dados; a de `train_test_split` controla a divisão. A floresta usa `np.random.choice` sem argumento `random_state`: chame `np.random.seed(...)` antes de `fit`. Outras chamadas ao gerador global entre a semente e o treino alteram a sequência de sorteios.

Registre também versões, parâmetros, dimensão dos dados, hardware e divisão experimental. Bibliotecas numéricas, paralelismo interno de BLAS e empates em valores singulares podem afetar resultados entre ambientes. Uma única execução com semente fixa não mede a variabilidade do método.

## Dados próprios

Prepare arrays bidimensionais de características e um vetor unidimensional de classes numéricas. Mantenha as mesmas colunas, na mesma ordem, no treino e no teste. Valores ausentes e atributos textuais precisam de tratamento externo, ajustado somente no conjunto de treino de cada avaliação.

Garanta `int(p * n_train) >= 1`, `int(s * d) >= 1`, `k >= 1` e `s <= 1`. Se o nó inicial de cada árvore receber menos de dez observações, o modelo pode se limitar a folhas majoritárias. Para o formato usado no Kaggle, consulte o [validador de NPZ](../kaggle/validar_dados.py).

## Problemas comuns

| Sintoma | Causa provável e ação |
|---|---|
| `ModuleNotFoundError` para NumPy/SciPy/sklearn/pandas | Ative o ambiente e instale com `python -m pip install -r requirements.txt` |
| Não encontra `data/data.npz` | Execute da raiz e coloque o arquivo no caminho esperado |
| Não consegue salvar `submissions/submission.csv` | Crie `submissions/` antes de executar o gerador |
| `KeyError` ao carregar o NPZ | Confira as chaves `X_train`, `y_train`, `X_test` |
| Floresta produz resultados degenerados com poucos atributos | Confira `int(s * d)`; ajuste `s` para selecionar pelo menos uma coluna |
| Erro de SVD ou resultados inválidos | Verifique NaN, infinito, tipos numéricos e dimensões |
| Erro de `mode` com rótulos textuais | Converta categorias em códigos numéricos consistentes e reverta quando necessário para a submissão |
| Acurácia varia apesar de `random_state=42` no dataset | Fixe também `np.random.seed` antes do treino da floresta |
| Treinamento demorado | Comece com menos árvores e menos amostras em uma avaliação exploratória; registre qualquer mudança de configuração |

## Ferramentas opcionais da documentação

Para regenerar as três figuras, instale a dependência opcional e execute:

```bash
python -m pip install -r docs/imagens/requirements.txt
python docs/imagens/gerar_figuras.py
```

Matplotlib não é necessário para treinar a oRF ou gerar a submissão. LaTeX também é opcional e só é necessário para compilar o [modelo de relatório](../relatorio/README.md).
