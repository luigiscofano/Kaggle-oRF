# Preparação do relatório

[Projeto](../../README.md) · [Enunciado](../enunciado/README.md) · [Experimentos](../experimentos/README.md)

O enunciado pede um relatório de **no máximo duas páginas**, no modelo disponibilizado pela disciplina. A documentação extensa deste repositório apoia a compreensão do projeto, mas não substitui esse relatório resumido.

## Material recebido

O arquivo `Main.zip` contém um modelo LaTeX e um PDF com título, autor, resumo, seções e figura ainda preenchidos com exemplos. Não há resultados experimentais utilizáveis nesse material. Os arquivos úteis estão preservados em [`template/`](template/README.md), incluindo o PDF original.

Há detalhes a revisar ao preparar o relatório final:

- `Title goes here` e `Name goes here` precisam ser substituídos.
- O título corrente no `.tex` menciona **PCS5022 — Neural Networks and Deep Learning**, enquanto o enunciado recebido é de **PCS 3838 — Inteligência Artificial**. Ajuste a identificação na sua versão de trabalho conforme a orientação da disciplina.
- `refs.bib` contém exemplos sobre HiRA e ReSi. Não são referências da oRF apenas por aparecerem no modelo; substitua-os pelas fontes efetivamente utilizadas.
- `Teaser.pdf` contém a figura de exemplo. Use uma figura relevante ou remova sua inclusão, conforme a organização final.
- Nomes, endereços e texto de exemplo não devem permanecer na entrega.

Os arquivos preservados em `template/` mantêm o conteúdo recebido, inclusive esses campos. O PDF fornecido e uma nova compilação do `.tex` devem ser tratados como artefatos distintos: conferir o PDF final é parte da preparação da entrega.

## Roteiro que atende aos quatro pontos exigidos

| Parte do relatório | Informação a incluir |
|---|---|
| Problema e método dos hiperplanos | Classificação; padronização local; PCA via SVD por nó; até três direções; limiar escolhido por ganho de informação |
| Construção da árvore | Entropia em base 2; limiares observados; regra esquerda/direita; profundidade; pureza; menos de dez amostras; ausência de corte válido |
| Construção da floresta | Valores usados de `k`, `p`, `s`; bootstrap de linhas; atributos fixos por árvore; votação por moda; sementes |
| Avaliação e conclusão | Base, divisão, métricas, tempos, configuração da RF, diferenças de profundidade, resultados medidos e limitações |

Uma estrutura possível é: introdução curta; metodologia com equações essenciais; experimentos com uma tabela de acurácia e tempo; conclusão apoiada pelos dados. Use o espaço conforme os resultados obtidos, evitando uma revisão bibliográfica longa que ocupe o lugar das decisões de implementação.

## Equações essenciais

Em duas páginas, normalmente basta explicitar a projeção local, a seleção do corte e a votação:

$$
z_i=((x_i-\mu_v)\oslash\sigma_v)^\top w,\qquad
IG=H(Y_v)-\frac{n_L}{n_v}H(Y_L)-\frac{n_R}{n_v}H(Y_R),
$$

$$
\widehat{y}(x)=\operatorname{mode}\{h_b(x_{F_b})\}_{b=1}^{k}.
$$

Explique como $w$ é proposto pela SVD e que a transformação é recalculada nos descendentes. O [guia teórico](../teoria/README.md) contém a derivação completa para consulta.

## Compilar uma cópia do modelo

Use uma distribuição LaTeX com os pacotes referenciados por `Main.tex` e `latexmk`. Para preservar o material original, copie `template/` para uma pasta de trabalho fora do repositório:

```bash
python -c "from pathlib import Path; import shutil; shutil.copytree('docs/relatorio/template', Path('../Kaggle-oRF-relatorio'))"
cd ../Kaggle-oRF-relatorio
latexmk -pdf -interaction=nonstopmode -halt-on-error Main.tex
```

O comando de cópia falha se o destino já existir, evitando sobrescrever uma versão de trabalho anterior. Edite `Main.tex` e `refs.bib` nessa cópia e compile novamente. Também é possível carregar os arquivos úteis em um editor LaTeX como Overleaf; os nomes dos arquivos devem ser preservados enquanto forem referenciados pelo documento.

## Conferência final

- Verifique o limite de **duas páginas**. Não presuma que referências ou apêndices estão excluídos desse limite sem orientação explícita da disciplina.
- Confirme que o método descrito é o que foi executado, incluindo o limite de três direções e a amostragem por árvore.
- Identifique cada conjunto de dados e informe como obtê-lo, conforme solicitado no enunciado.
- Informe dependências adicionais e configurações suficientes para reproduzir o experimento.
- Use apenas métricas realmente medidas, distinguindo validação local e resultado no Kaggle.
- Abra o PDF final e confira equações, figuras, legibilidade, referências e campos de exemplo remanescentes.

As [figuras didáticas](../imagens/README.md) podem apoiar a exposição, mas uma tabela ou gráfico de resultados próprios é mais adequado para sustentar a comparação experimental. Uma ilustração não substitui evidência de desempenho.
