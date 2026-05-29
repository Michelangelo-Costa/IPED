# Documentacao do Plugin AIGCD para IPED

## Visao geral

O AIGCD e uma extensao em desenvolvimento para o IPED com foco em deteccao de conteudo gerado por inteligencia artificial. A proposta e calcular scores de suspeita por modalidade, armazenar esses scores no indice do caso e exibi-los na interface grafica do IPED para apoiar a analise pericial.

O nome AIGCD pode ser lido como "AI-Generated Content Detection".

Estado atual do desenvolvimento:

- A deteccao implementada cobre imagens.
- A Task `AIGCDTask` fica no modulo separado `iped-aigcd-plugin` e usa o modelo ConvNeXt V2 Base exportado para ONNX para produzir um score de 0.0 a 1.0.
- A aba `AIGCD` da interface exibe o painel de score e a arvore de filtros em um unico lugar.
- Os filtros AIGCD usam faixas High (>=0.70), Medium (>=0.40) e Low (<0.40) para os campos de score.
- Video, audio e texto aparecem como arquitetura planejada, mas ainda nao possuem tasks reais implementadas.

## Modelo usado

**Modelo**: ConvNeXt V2 Base (`convnextv2_base`) treinado pelo projeto xRayon/Toufik.
**Repositorio de referencia**: `https://huggingface.co/xRayon/convnext-ai-images-detector`
**Checkpoint local**: `C:\IPED\xrayon-convnext\AI Images Detector\checkpoints\checkpoint_phase2.pth`
**Formato de deploy**: ONNX (exportado com `torch.onnx.export`, dynamo=False), tamanho ~334 MB.
**Caminho esperado pelo plugin**: `<iped-root>/models/convnext-ai-detector/model.onnx`

O modelo foi treinado para classificar imagens em duas classes:
- Classe 0: real (humana/fotografada)
- Classe 1: fake (gerada por IA)

O plugin aplica softmax nas logits brutas e retorna a probabilidade da classe 1 como score.

## Onde o codigo foi alterado

Arquivos principais:

- `iped-aigcd-plugin/src/main/java/iped/aigcd/AIGCDTask.java`
  - Task de processamento do IPED.
  - Carrega o modelo ONNX via ONNX Runtime.
  - Processa itens cujo MIME type comeca com `image/`.
  - Preprocessamento: Resize(288) -> CenterCrop(256) -> Normalize ImageNet.
  - Aplica softmax nas logits e retorna probabilidade da classe 1 (fake) como score.
  - Grava o score numerico no atributo `aigcd:score:image`.

- `iped-aigcd-plugin/pom.xml`
  - Compila o JAR do plugin.
  - Copia `iped-aigcd-plugin-4.4.0-SNAPSHOT.jar` para `target/release/iped-4.4.0-SNAPSHOT/lib`.
  - Copia a dependencia `onnxruntime-1.17.0.jar` para `target/release/plugins`.

- `iped-app/src/main/java/iped/app/ui/AIGCDPanel.java`
  - Painel visual exibido na interface do IPED.
  - Mostra score geral e scores por modalidade.
  - Classifica o maior score como `HIGH`, `MEDIUM`, `LOW` ou `N/A`.

- `iped-app/src/main/java/iped/app/ui/App.java`
  - Registra a aba `AIGCD` no layout da interface.
  - Agrupa o painel de score e a arvore de filtros AIGCD em uma unica aba.

- `iped-app/src/main/java/iped/app/ui/FileProcessor.java`
  - Linha 186: `App.get().aigcdPanel.loadDoc(doc);`
  - Atualiza o painel AIGCD quando o usuario seleciona um item.

- `iped-app/resources/config/conf/TaskInstaller.xml`
  - Registra `iped.aigcd.AIGCDTask` na pipeline de processamento.

- `iped-app/resources/config/conf/AIFiltersConfig.json`
  - Adiciona filtros da aba AI para:
    - `aigcd:score:image`
    - `aigcd:score:video`
    - `aigcd:score:audio`
    - `aigcd:score:text`

- `pom.xml` (raiz)
  - Inclui o modulo `iped-aigcd-plugin` no build Maven.

## Como rodar o IPED com o AIGCD

As alteracoes estao no codigo-fonte em:

```powershell
C:\IPED\IPED
```

Depois do build, a versao gerada fica em:

```powershell
C:\IPED\IPED\target\release\iped-4.4.0-SNAPSHOT
```

### Requisitos

- Java 11 Liberica Full JDK (com JavaFX)
- Maven 3.9+
- Windows 11

### Build completo

```powershell
cd C:\IPED\IPED
mvn install -DskipTests
```

Para recompilar somente os modulos relevantes durante o desenvolvimento:

```powershell
cd C:\IPED\IPED
mvn install -pl iped-app,iped-aigcd-plugin --am -DskipTests
```

Evite usar `mvn clean` sem necessidade. Se o modelo `model.onnx` estiver somente dentro de `target\release\iped-4.4.0-SNAPSHOT\models\convnext-ai-detector`, voce precisara copia-lo novamente depois do clean.

### Modelo ONNX

A Task procura o modelo em:

```powershell
<iped-root>\models\convnext-ai-detector\model.onnx
```

Exemplo com a versao de build:

```powershell
C:\IPED\IPED\target\release\iped-4.4.0-SNAPSHOT\models\convnext-ai-detector\model.onnx
```

Se o arquivo nao existir, a Task registra um aviso no log e continua o processamento sem calcular o score AIGCD.

Para exportar o modelo do checkpoint PyTorch para ONNX, use o script `export_model.py` disponivel em `iped-aigcd-plugin/` (requer Python com torch e timm instalados). O modelo exportado tem aproximadamente 334 MB.

### Processar uma pasta de teste

```powershell
cd C:\IPED\IPED\target\release\iped-4.4.0-SNAPSHOT
.\iped.exe -d C:\IPED\amostras-aigcd -o C:\IPED\casos\caso-aigcd-XX
```

Onde:
- `-d` indica a evidencia ou pasta de entrada.
- `-o` indica a pasta de saida do caso processado (use um nome novo a cada teste).

### Onde o JAR do plugin precisa estar

O JAR do plugin **deve estar na pasta `lib/`** da instalacao do IPED, nao em `plugins/`. Isso e necessario porque o `TaskInstallerConfig` usa `Class.forName()` (classloader do sistema), que so enxerga JARs do `lib/`.

O `pom.xml` do plugin ja cuida disso automaticamente no build:

```
target/release/iped-4.4.0-SNAPSHOT/lib/iped-aigcd-plugin-4.4.0-SNAPSHOT.jar
```

O `onnxruntime-1.17.0.jar` vai para `plugins/`:

```
target/release/plugins/onnxruntime-1.17.0.jar
```

### Abrir o caso na interface grafica

```powershell
cd C:\IPED\casos\caso-aigcd-XX
.\IPED-SearchApp.exe
```

O `IPED-SearchApp.exe` dentro da pasta do caso ja aponta para o `iped-search-app.jar` correto. Certifique-se de que esse JAR vem da versao compilada com `AIGCDPanel`, nao de uma versao anterior.

## Fluxo tecnico do AIGCD

### Fase 1: Processamento (indexacao)

```
TaskInstaller.xml
  <task class="iped.aigcd.AIGCDTask"/>
        |
        v
TaskInstallerConfig.java
  Class.forName("iped.aigcd.AIGCDTask")   <- requer JAR em lib/
        |
        v
AIGCDTask.init()
  carrega models/convnext-ai-detector/model.onnx via ONNX Runtime
        |
        v  (para cada arquivo do caso)
AIGCDTask.process(IItem item)
  verifica MIME type comeca com "image/"
  le o stream da imagem
  Resize(288) -> CenterCrop(256x256) -> Normalize ImageNet
  roda inferencia ONNX -> logits [real, fake]
  softmax -> score = exp(fake) / (exp(real) + exp(fake))
  item.setExtraAttribute("aigcd:score:image", score)
        |
        v
Lucene Index
  StoredField("aigcd:score:image", 0.97f)
```

### Fase 2: Visualizacao (SearchApp)

```
Usuario clica em um arquivo
        |
        v
FileProcessor.java:186
  App.get().aigcdPanel.loadDoc(doc)
        |
        v
AIGCDPanel.loadDoc(Document doc)
  doc.getField("aigcd:score:image")   <- le do Lucene
  calcula HIGH/MEDIUM/LOW
  atualiza badge + barra de progresso na tela
```

## Campos de metadados

| Campo | Modalidade | Status |
| --- | --- | --- |
| `aigcd:score:image` | Imagem | Implementado e validado |
| `aigcd:score:video` | Video | Planejado |
| `aigcd:score:audio` | Audio | Planejado |
| `aigcd:score:text` | Texto | Planejado |

Convencao de score:

- `0.00` a `0.39`: baixa suspeita (LOW).
- `0.40` a `0.69`: suspeita media (MEDIUM).
- `0.70` a `1.00`: alta suspeita (HIGH).

Importante: esses scores sao indicadores probabilisticos. Eles nao devem ser tratados como conclusao automatica. A decisao final precisa ser feita por uma pessoa avaliadora, com contexto e validacao.

## Pontos de atencao tecnica

### 1. A Task esta isolada em JAR, mas a UI ainda e integrada

A parte de processamento do AIGCD fica em um modulo proprio (`iped-aigcd-plugin`), com JAR copiado para `lib/` e classe registrada em `TaskInstaller.xml`.

A parte da interface grafica ainda altera o `iped-app`, porque o `AIGCDPanel` e instanciado diretamente em `App.java` e atualizado por `FileProcessor.java`. O IPED nao tem um contrato publico para plugins criarem novas abas/paineis da UI sem alterar essas classes. Isso e uma limitacao arquitetural documentada.

### 2. Como saber se existe ponto oficial de extensao para abas/paineis

O IPED tem mecanismo claro para:
- carregar JARs da pasta configurada em `LocalConfig.txt` por `pluginFolder`
- adicionar esses JARs ao classpath no bootstrap
- ler recursos de configuracao dentro de JARs de plugin
- instanciar tasks registradas em `TaskInstaller.xml` com `Class.forName(...)`

Para a UI, a criacao dos paineis e feita diretamente em `App.java` com chamadas como `createDockable(...)`. Nao existe uma interface do tipo `UIPanelPlugin` ou `ServiceLoader` para paineis. A conclusao e: o IPED tem extensao oficial para tasks; para novas abas da interface, pelo codigo atual, nao ha ponto de extensao formal.

### 3. Falta configuracao para habilitar/desabilitar

`AIGCDTask.isEnabled()` retorna sempre `true`. Para uso real, o ideal e criar uma configuracao propria:

```text
enableAIGCD=true
aigcdModelPath=models/convnext-ai-detector/model.onnx
aigcdMinScoreHigh=0.70
aigcdMinScoreMedium=0.40
```

### 4. Filtros de video, audio e texto ainda sao futuros

O arquivo `AIFiltersConfig.json` ja adiciona filtros para video, audio e texto, mas a Task atual so calcula imagem. Esses filtros ficam vazios ate existirem tasks especificas.

### 5. Validacao experimental

Testes iniciais com quatro imagens:

| Imagem | Score | Classificacao |
|---|---|---|
| Pizza.png (IA) | 97.43% | HIGH |
| Gatos.png (IA) | 97.17% | HIGH |
| Gato.jpeg (real) | 15.47% | LOW |
| Pizzas.jpeg (real) | 2.87% | LOW |

O modelo mostrou boa separacao para essas amostras. Deve ser avaliado com um conjunto maior e com imagens de diferentes geradores e estilos.

### 6. Falta teste automatizado

Sugestoes:
- Teste unitario para classificacao High/Medium/Low.
- Teste da Task com uma imagem pequena e um modelo mockado.
- Teste para confirmar que `aigcd:score:image` aparece no documento Lucene.
- Teste para confirmar que o filtro AI retorna itens nas faixas esperadas.

## Organizacao recomendada para o TCC

1. **Problema**: Crescimento de conteudos gerados por IA e dificuldade de triagem em pericia digital.
2. **Objetivo geral**: Integrar ao IPED um modulo de apoio a deteccao de conteudo gerado por IA.
3. **Objetivos especificos**:
   - Criar uma Task de processamento para imagens.
   - Armazenar scores no indice do caso.
   - Criar filtros pesquisaveis na aba AI.
   - Criar painel de visualizacao dos scores.
   - Avaliar desempenho e confiabilidade em uma base de teste.
4. **Artefato de software**: Customizacao/plugin AIGCD para o IPED.
5. **Avaliacao**: Testar com imagens reais e geradas por IA, medir metricas quantitativas, discutir limitacoes.

## Git, fork e organizacao do repositorio

Para registrar autoria no TCC, crie um fork no GitHub e aponte seu clone para ele:

```powershell
cd C:\IPED\IPED
git remote rename origin upstream
git remote add origin https://github.com/SEU_USUARIO/IPED.git
git switch -c tcc-aigcd
git push -u origin tcc-aigcd
```

Para atualizar com mudancas do IPED oficial:

```powershell
git fetch upstream
git merge upstream/master
```

## Registro de software

Para registro de software, separe claramente:
- O que e codigo original do IPED (GPL).
- O que foi criado/modificado: `AIGCDTask.java`, `AIGCDPanel.java`, modificacoes em `App.java`, `FileProcessor.java`, `TaskInstaller.xml`, `AIFiltersConfig.json`, `pom.xml`.
- Modelo ConvNeXt V2 Base: treinado por xRayon/Toufik, verificar licenca antes de distribuir.
- Dependencia ONNX Runtime: licenca MIT (Microsoft).

O IPED usa GPL, entao uma distribuicao derivada precisa respeitar essa licenca.

## Roadmap sugerido

**Prioridade alta**:
- Criar configuracao para habilitar/desabilitar a Task e trocar o caminho do modelo.
- Documentar origem e licenca do modelo ONNX.
- Avaliar com um conjunto maior de imagens (reais e geradas por IA).
- Medir tempo medio de processamento por imagem.

**Prioridade media**:
- Adicionar testes automatizados.
- Criar capturas de tela da aba AIGCD para o TCC.
- Calibrar thresholds com uma base rotulada.
- Melhorar logs e mensagens de erro.

**Prioridade futura**:
- Implementar deteccao de video, audio e texto.
- Criar relatorio exportavel com scores AIGCD.
- Propor ao IPED um mecanismo de extensao de UI via plugins (trabalho futuro).

## Resumo honesto do estado atual

O prototipo esta funcional para imagens: o plugin carrega o modelo ConvNeXt V2 Base via ONNX Runtime, processa cada imagem no pipeline do IPED, grava o score no indice Lucene e exibe o resultado na aba AIGCD do SearchApp. Os testes iniciais com quatro amostras mostraram boa separacao entre imagens reais e geradas por IA.

A Task de processamento esta separada em JAR de plugin. A interface grafica ainda requer modificacao do codigo-fonte do `iped-app`, pois o IPED nao tem ponto de extensao de UI formal. Isso e documentado como limitacao arquitetural e motivacao para trabalho futuro.
