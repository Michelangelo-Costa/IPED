# Teste em larga escala no IPED

Processa centenas ou milhares de imagens com rótulo conhecido **dentro do próprio IPED** e
mede o que o AIGCD acerta, erra e quanto custa em tempo.

```
montar_dataset.py  →  iped.exe processa  →  exportar CSV na interface  →  calcular_metricas.py  →  gerar_figuras.py
```

O rótulo (real ou IA) fica no **nome da pasta**. O IPED preserva o caminho de cada arquivo no
campo `path`, então nenhuma anotação manual é necessária:

```
evidencias/<variante>/<ia|real>/<grupo>/<arquivo>
```

| Script | Função |
|---|---|
| `montar_dataset.py` | Baixa imagens rotuladas (Tiny-GenImage e CIFAKE) e cria a variante recomprimida em JPEG |
| `calcular_metricas.py` | Lê o CSV exportado do IPED e calcula ROC-AUC, EER, matrizes, faixas e tempo por imagem |
| `gerar_figuras.py` | Gera histogramas, curvas ROC, detecção por gerador, imagens próprias e tempo por tarefa |

---

## 0. Dependências (uma vez)

```powershell
py -m pip install datasets pillow matplotlib
```

> Se você também usa a biblioteca `transformers`, a instalação do `datasets` pode atualizar o
> `huggingface-hub` para uma versão incompatível com ela. Nesse caso:
> `py -m pip install "huggingface-hub>=1.31,<2"`.

Os exemplos usam uma pasta de trabalho fora do repositório, `C:\aigcd-teste`, e o repositório
em `C:\IPED`.

```powershell
$SCRIPTS = "C:\IPED\aigcd\avaliacao\teste-larga-escala"
New-Item -ItemType Directory -Force C:\aigcd-teste | Out-Null
cd C:\aigcd-teste
```

## 1. Montar o conjunto de imagens

Teste rápido (35 imagens de IA e 35 reais, mais a cópia em JPEG, cerca de 1 minuto):

```powershell
py "$SCRIPTS\montar_dataset.py" --saida C:\aigcd-teste\evidencias-fumaca --por-gerador 5 --versao-jpeg 85
```

Conjunto principal usado no TCC (1.600 imagens, mais 1.600 em JPEG):

```powershell
py "$SCRIPTS\montar_dataset.py" --saida C:\aigcd-teste\evidencias --por-gerador 100 --cifake 100 --versao-jpeg 85
```

| Parâmetro | Efeito |
|---|---|
| `--por-gerador N` | N imagens de IA por gerador do GenImage (7 geradores; máximo 500) |
| `--reais N` | Imagens reais do GenImage. Padrão: o mesmo total das de IA |
| `--cifake N` | N imagens por classe do CIFAKE (32×32 px) |
| `--versao-jpeg Q` | Cria a variante com tudo recomprimido em JPEG de qualidade Q |

As imagens são gravadas com os bytes originais, e um manifesto com o SHA-256 de cada arquivo
é salvo junto.

**Por que a variante JPEG existe:** no GenImage, todas as fotos reais são JPEG e todas as
imagens geradas são PNG; um detector pode acertar só pelo formato. Recomprimir as duas classes
com a mesma qualidade remove esse atalho e simula o compartilhamento por aplicativos de
mensagem e redes sociais.

**Imagens próprias (opcional):** crie subpastas e coloque suas imagens. O nome da subpasta vira
o "grupo" no relatório:

```
evidencias\proprias\real\celular\     ← fotos originais da câmera (sem passar por WhatsApp)
evidencias\proprias\ia\chatgpt\       ← imagens baixadas pelo botão de download do gerador
evidencias\proprias\ia\gemini\
evidencias\sem_rotulo\                ← imagens comuns; o relatório mostra quantas viram "Alto"
```

## 2. Processar no IPED

```powershell
cd C:\IPED\target\release\iped-4.4.0-SNAPSHOT
.\iped.exe -d C:\aigcd-teste\evidencias -o C:\casos\caso-aigcd-macro
```

Nenhuma cópia manual é necessária. Anote o log do processamento (o mais recente):

```powershell
$LOG = (Get-ChildItem C:\IPED\target\release\iped-4.4.0-SNAPSHOT\log\IPED-*.log | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
```

## 3. Exportar os scores

1. Abra `C:\casos\caso-aigcd-macro\IPED-SearchApp.exe`, sem nenhum filtro ativo.
2. Clique em uma linha da tabela e pressione **Ctrl+A**.
3. Botão direito → **Exportar propriedades dos itens destacados** (*Export highlighted items
   properties*).
4. Em **Exibir colunas** (*Show Columns*), escolha **Todas** (*All*), marque **path** e
   **aigcd:score:image** e clique em **OK**.
5. Salve como `C:\aigcd-teste\export.csv` (sem espaço no final do nome).

O separador do CSV muda com o idioma da interface (vírgula em inglês, ponto e vírgula em
português); o script detecta sozinho.

## 4. Calcular métricas e gerar figuras

```powershell
cd C:\aigcd-teste
py "$SCRIPTS\calcular_metricas.py" export.csv --log $LOG
py "$SCRIPTS\gerar_figuras.py" --csv resultados_export\resultados_por_imagem.csv --saida figuras --log $LOG
```

Saídas em `C:\aigcd-teste\resultados_export\`:

| Arquivo | Conteúdo |
|---|---|
| `relatorio.txt` | Todas as métricas, por variante e por grupo |
| `resultados_por_imagem.csv` | Score, faixa e acerto de cada imagem |
| `piores_erros_<variante>.csv` | Reais com score mais alto e imagens de IA com score mais baixo |
| `nao_processadas.csv` | Imagens sem score (formato não lido) |
| `grafico_<variante>.png` | Histograma e curva ROC por variante |

As faixas usam a mesma regra do painel e dos filtros: Alto ≥ 0,70; Médio de 0,40 a 0,70;
Baixo < 0,40.

---

## Resultados obtidos no TCC

Caso com 3.219 imagens, Intel Core i5-12400F, 12 threads de processamento:

| Conjunto | ROC-AUC [IC 95%] | Detecção (Alto) | Falsos positivos (Alto) | Precisão |
|---|---|---|---|---|
| GenImage original | 0,990 [0,985; 0,994] | 92,9% | 2,6% | 97,3% |
| GenImage em JPEG q85 | 0,871 [0,850; 0,892] | 43,6% | 2,9% | 93,8% |
| CIFAKE (32×32) | 0,585 [0,504; 0,654] | 29,0% | 18,0% | 61,7% |

Detecção por gerador (porcentagem de imagens de IA em Alto), antes e depois da recompressão:

| ADM | BigGAN | VQDM | SD 1.5 | GLIDE | Wukong | Midjourney |
|---|---|---|---|---|---|---|
| 93 → 13 | 99 → 24 | 99 → 29 | 94 → 49 | 98 → 68 | 90 → 61 | 77 → 61 |

Desempenho: 71 ms por imagem em média (tempo acumulado do `AIGCDTask` dividido pelo número de
imagens); o caso inteiro foi processado em 7 min 43 s, com reconhecimento facial e estimativa
de idade também habilitados.

Principais conclusões:

- Em imagens no formato original do gerador, o detector separa muito bem real e IA.
- A recompressão JPEG reduz a detecção pela metade, sem alterar os falsos positivos: parte do
  sinal usado pelo modelo está ligada ao formato e à compressão do arquivo.
- Em imagens de 32×32 px o desempenho fica próximo do acaso.
- Em uma pequena amostra de imagens próprias, 2 de 8 fotos de geradores recentes (ChatGPT e
  Gemini) foram detectadas, e 2 de 4 fotos reais de smartphone receberam Alto.
- O modelo concentra os scores nos extremos e erra com confiança alta; **um resultado Baixo
  não prova que a imagem é real**.
