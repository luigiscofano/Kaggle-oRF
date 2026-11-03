# Do conjunto de dados à submissão Kaggle

[Projeto](../../README.md) · [Execução](../execucao/README.md) · [Experimentos](../experimentos/README.md)

O objetivo desta etapa é treinar a oRF proposta e gerar predições para as observações sem rótulo da competição. O repositório contém o gerador; **os dados, a URL da competição e um arquivo oficial de exemplo de submissão não foram fornecidos**.

## 1. Obter e conferir os dados

Acesse o endereço disponibilizado pela disciplina, consulte as regras, a métrica e o formato de submissão e obtenha os arquivos oficiais. Não presuma que o esquema do gerador corresponde automaticamente ao formato atual da competição.

O [`src/GeradorCSV.py`](../../src/GeradorCSV.py) espera um arquivo NumPy `.npz` no caminho `data/data.npz`, relativo à raiz do repositório:

| Chave | Forma | Conteúdo |
|---|---|---|
| `X_train` | `(n_train, d)` | Atributos das amostras rotuladas |
| `y_train` | `(n_train,)` | Classes numéricas correspondentes |
| `X_test` | `(n_test, d)` | Atributos das amostras para submissão, na ordem oficial |

Use arrays numéricos finitos, sem valores ausentes. Os atributos de treino e teste devem ter a mesma ordem e significado. Se a fonte oficial estiver em outro formato, será necessário convertê-la respeitando seu esquema; o repositório não contém um conversor genérico de CSV para os dados da competição.

Se os arrays já estiverem preparados em memória, o padrão de gravação é:

```python
from pathlib import Path
import numpy as np

Path("data").mkdir(exist_ok=True)
np.savez("data/data.npz", X_train=X_train, y_train=y_train, X_test=X_test)
```

Esse trecho pressupõe as três variáveis previamente definidas. Não cria os dados oficiais e sobrescreve um NPZ existente no mesmo caminho.

## 2. Validar antes de um treinamento longo

```bash
python docs/kaggle/validar_dados.py
```

O [validador](validar_dados.py) confere chaves, dimensões, tipos, valores finitos, correspondência entre linhas/rótulos e a seleção de ao menos um atributo com `s=0.27`. Também informa a quantidade de sorteios de linhas e colunas por árvore. Ele não verifica semântica dos atributos, vazamento, classes aceitas pela competição ou correspondência dos IDs oficiais.

Para examinar outro arquivo:

```bash
python docs/kaggle/validar_dados.py caminho/para/data.npz
```

## 3. Escolher o modelo usando validação local

Antes da submissão, separe uma parte dos dados rotulados para validação. Ajuste hiperparâmetros usando essa etapa, sem incorporar os rótulos de validação ao treino. Eventual imputação, seleção de atributos ou codificação também deve ser ajustada apenas na partição de treino.

O `GeradorCSV.py` treina diretamente em todo `X_train` e não cria uma validação interna. O [guia experimental](../experimentos/README.md) explica como fazer a avaliação antes de gerar o arquivo final. O conjunto `X_test` de submissão não possui rótulos neste contrato; não é possível medir sua acurácia local sem informação adicional.

## 4. Gerar o arquivo

Na raiz do projeto, com `data/data.npz` preparado:

```bash
python -c "from pathlib import Path; Path('submissions').mkdir(exist_ok=True)"
python src/GeradorCSV.py
```

O utilitário instancia `oRF()` com `k=250`, `p=0.63`, `s=0.27`, treina nos dados rotulados e grava `submissions/submission.csv`. Não há argumento de linha de comando para alterar esses valores ou o caminho. Uma execução nova sobrescreve o CSV anterior.

Para fixar a semente sem editar o código original, execute da raiz:

```bash
python -c "import sys, runpy, numpy as np; sys.path.insert(0, 'src'); np.random.seed(42); runpy.run_path('src/GeradorCSV.py', run_name='__main__')"
```

Esse comando também executa o treinamento completo e grava o CSV; escolha uma das duas formas de execução.

## 5. Entender e conferir o CSV

Exemplo **fictício**, apenas para ilustrar as colunas:

```csv
ID,Prediction
1,0
2,1
3,1
```

O ID da primeira amostra é 1. A predição da linha `i` corresponde à observação de índice `i-1` em `X_test`. O código não lê IDs de um arquivo externo. Se a competição usar outra identificação, o utilitário precisará ser adaptado antes do envio.

Confira, no mínimo:

- Colunas, capitalização, separador e ausência de uma coluna extra de índice.
- Número de linhas igual a `X_test.shape[0]`.
- IDs e ordem compatíveis com o exemplo oficial de submissão.
- Rótulos no domínio aceito, sem NaN ou valores infinitos.
- Conversão reversa de rótulos, caso tenha sido usada uma codificação de classes.

## 6. Enviar e registrar

Na página da competição, envie o CSV pelo fluxo oficial e aguarde sua validação. Registre a configuração, a semente, a versão do código, a data e o resultado exibido. Uma pontuação de validação local e uma pontuação do leaderboard têm origens diferentes; registre-as separadamente. O sucesso da exportação de CSV não comprova aceitação pelo Kaggle.

`data/` e `submissions/` são ignoradas pelo Git. Essa convenção mantém dados e saídas geradas fora do código versionado. Esta documentação não executa uma submissão nem afirma uma posição no leaderboard.
