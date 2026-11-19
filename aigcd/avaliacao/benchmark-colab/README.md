# Benchmark de detectores pré-treinados (Google Colab)

Notebook que compara o modelo do AIGCD (ConvNeXt V2 em ONNX) com cinco detectores públicos
do Hugging Face, fora do IPED, em três conjuntos de imagens de dificuldade crescente.

| Arquivo | Descrição |
|---|---|
| `AIGCD_Benchmark_Parte1_Detectores.ipynb` | Notebook pronto para o Colab |
| `build_parte1.py` | Gera o notebook (`py build_parte1.py`, requer `nbformat`) |

## O que é avaliado

**Modelos:** ConvNeXt V2 (modelo do plugin, ONNX) e os detectores do Hugging Face
SDXL-Detector, AI-image-detector, AIorNot, Deep-Fake-Detector-v2 e ai_vs_real.

**Conjuntos:**

| Conjunto | Característica | Papel |
|---|---|---|
| AI-vs-Real | IA em estilo artístico | caso fácil, limite superior |
| GenImage | IA fotorrealista de vários geradores | caso difícil, mais próximo da perícia |
| CIFAKE | imagens de 32×32 px | robustez a baixa resolução |

**Métricas:** ROC-AUC, EER, F1, calibração, tempo de inferência e distribuição nas faixas do
painel (Baixo, Médio, Alto).

## Como executar

1. Coloque o `model.onnx` (veja [../../modelo/README.md](../../modelo/README.md)) no seu
   Google Drive, em `MyDrive/AIGCD/model.onnx`. O notebook também procura o arquivo em outras
   pastas do Drive.
2. Em [colab.research.google.com](https://colab.research.google.com): **Arquivo → Fazer upload
   de notebook** e escolha o `.ipynb`.
3. Opcional, mas recomendado: **Ambiente de execução → Alterar o tipo de ambiente → GPU (T4)**.
4. **Ambiente de execução → Executar tudo** e autorize o acesso ao Google Drive quando
   solicitado.
5. Confira as verificações visuais de rótulos (seções 5.1, 11 e 12): a fileira de cima deve
   ser de imagens reais e a de baixo, de IA. Se estiver invertido, ajuste `AI_LABEL_VALUE`,
   `GENIMAGE_AI_LABEL` ou `CIFAKE_AI_LABEL` e execute de novo.
6. Ao final, os resultados (CSVs, resumo e figuras) são compactados em
   `aigcd_resultados.zip`.

Tempo estimado: 40 a 60 minutos. O parâmetro `N_PER_CLASS` (seção 2, padrão 400) controla
quantas imagens por classe são usadas em cada conjunto.

## Resultados obtidos (ROC-AUC)

Modelos com resultado registrado nos três conjuntos na execução feita para o TCC:

| Modelo | AI-vs-Real | GenImage | CIFAKE |
|---|---|---|---|
| ConvNeXt V2 (plugin AIGCD) | **0,971** | **0,998** | 0,538 |
| ai_vs_real (dima806) | 0,752 | 0,657 | **0,999** |
| SDXL-Detector | 0,792 | 0,719 | 0,664 |
| Deep-Fake-Detector-v2 | 0,735 | 0,556 | 0,510 |
| AI-image-detector | 0,621 | 0,655 | 0,632 |

Nenhum detector é universal: o modelo do plugin lidera nos conjuntos de resolução normal, mas
fica próximo do acaso em 32×32 px. O valor 0,998 no GenImage provavelmente reflete
proximidade com os dados de treino do modelo; o
[teste em larga escala](../teste-larga-escala/) mostra que a vantagem cai após recompressão
JPEG.
