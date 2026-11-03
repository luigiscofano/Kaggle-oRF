# Figuras explicativas

[Projeto](../../README.md) · [Teoria](../teoria/README.md)

As três figuras desta pasta foram elaboradas para explicar o algoritmo. São **ilustrações conceituais**, não resultados de modelos treinados nem avaliações do Kaggle.

| Figura | O que representa |
|---|---|
| [`fronteiras.png`](fronteiras.png) | Os mesmos pontos sintéticos com uma fronteira ortogonal em degraus e uma fronteira oblíqua; regras desenhadas manualmente |
| [`no-obliquo.png`](no-obliquo.png) | Operações locais de um nó, da entrada dos dados ao armazenamento dos parâmetros e à recursão |
| [`floresta.png`](floresta.png) | Bootstrap, atributos por árvore e um exemplo fictício de votação com cinco árvores |

## Regeneração

Na raiz do repositório:

```bash
python -m pip install -r docs/imagens/requirements.txt
python docs/imagens/gerar_figuras.py
```

O [script](gerar_figuras.py) usa NumPy e Matplotlib, sem acesso à internet e sem treinar os classificadores. A semente da nuvem de pontos é fixa. Os PNGs são gravados nesta pasta, substituindo versões anteriores; os demais arquivos do projeto não são modificados.

As quatro imagens originais do enunciado estão em [`../enunciado/originais/`](../enunciado/originais/) e não são geradas por esse script.
