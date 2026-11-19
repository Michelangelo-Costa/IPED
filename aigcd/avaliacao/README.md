# Avaliação experimental do AIGCD

Duas avaliações complementares, ambas reproduzíveis com os arquivos desta pasta:

| Avaliação | Onde roda | Pergunta que responde |
|---|---|---|
| [teste-larga-escala](teste-larga-escala/) | No IPED, com o plugin | O plugin funciona em escala? Quanto acerta e quanto custa, no fluxo real do perito? |
| [benchmark-colab](benchmark-colab/) | Google Colab, fora do IPED | Como o modelo do plugin se compara a outros detectores públicos? |

## Resumo dos resultados

| Conjunto | ROC-AUC | Detecção (Alto) | Falsos positivos (Alto) |
|---|---|---|---|
| GenImage, formato original | 0,990 | 92,9% | 2,6% |
| GenImage, recomprimido em JPEG | 0,871 | 43,6% | 2,9% |
| CIFAKE, 32×32 px | 0,585 | 29,0% | 18,0% |

- **Software:** 3.219 imagens processadas sem falhas, cerca de 71 ms por imagem, painel e
  filtros consistentes (Alto + Médio + Baixo = total).
- **Modelo:** excelente em imagens no formato original do gerador; perde metade da detecção
  após recompressão JPEG; próximo do acaso em imagens muito pequenas; dificuldade com
  geradores recentes.
- **Uso pericial:** um resultado Alto é um bom indicador para priorizar a análise; um
  resultado Baixo não atesta autenticidade.

Os conjuntos públicos usados são o [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage)
(subconjunto do GenImage, Zhu et al., NeurIPS 2023), o
[CIFAKE](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset) e, no
benchmark, o [AI-vs-Real](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real).
