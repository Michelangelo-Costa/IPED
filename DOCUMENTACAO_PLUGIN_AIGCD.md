# Documentacao do Plugin AIGCD para IPED

## Visao geral

O AIGCD e uma extensao para o IPED com foco em deteccao de conteudo gerado por
inteligencia artificial. Calcula um score de suspeita para imagens, armazena no
indice do caso e exibe na interface grafica para apoiar a analise pericial.

AIGCD = "AI-Generated Content Detection".

**Estado atual:**
- Deteccao implementada para imagens (score 0.0 a 1.0).
- `AIGCDTask` fica no modulo separado `iped-aigcd-plugin` usando ConvNeXt V2 Base via ONNX Runtime.
- Painel visual (`AIGCDPanel`) integrado na aba **AI** do SearchApp, junto com as demais features de IA do IPED (faces, idade estimada).
- Filtros High/Medium/Low na arvore AI permitem triagem rapida.
- Deteccao de video, audio e texto e trabalho futuro.

---

## Modelo usado

| Item | Valor |
|---|---|
| Arquitetura | ConvNeXt V2 Base (`convnextv2_base`) |
| Referencia | https://huggingface.co/xRayon/convnext-ai-images-detector |
| Checkpoint local | `C:\IPED\xrayon-convnext\AI Images Detector\checkpoints\checkpoint_phase2.pth` |
| Formato de deploy | ONNX (~334 MB), exportado com `torch.onnx.export(dynamo=False)` |
| Caminho esperado | `<iped-root>/models/convnext-ai-detector/model.onnx` |
| Classes | 0 = real, 1 = gerada por IA |

O plugin aplica softmax nas logits brutas e retorna a probabilidade da classe 1 como score.

---

## Arquivos criados e modificados

### Novos arquivos

| Arquivo | Descricao |
|---|---|
| `iped-aigcd-plugin/pom.xml` | Modulo Maven do plugin |
| `iped-aigcd-plugin/src/main/java/iped/aigcd/AIGCDTask.java` | Task de inferencia ONNX |
| `iped-app/src/main/java/iped/app/ui/AIGCDPanel.java` | Painel visual Swing |
| `iped-app/src/main/resources/iped/app/ui/filter/AIGCDImage.*.png` | Icones dos filtros na aba AI |

### Arquivos modificados

| Arquivo | O que mudou |
|---|---|
| `pom.xml` (raiz) | Incluiu `iped-aigcd-plugin` no build Maven |
| `iped-app/src/main/java/iped/app/ui/App.java` | Instancia e registra `AIGCDPanel` na aba AI |
| `iped-app/src/main/java/iped/app/ui/FileProcessor.java` | Chama `aigcdPanel.loadDoc(doc)` ao selecionar item |
| `iped-app/resources/config/IPEDConfig.txt` | Adicionado `enableAIGCDDetector = true` |
| `iped-app/resources/config/conf/TaskInstaller.xml` | Registra `iped.aigcd.AIGCDTask` na pipeline |
| `iped-app/resources/config/conf/AIFiltersConfig.json` | Adiciona filtro `aigcd:score:image` na aba AI |
| `iped-app/resources/localization/iped-ai-filters*.properties` | Labels do filtro AIGCD (6 idiomas) |
| `iped-app/resources/localization/iped-desktop-messages*.properties` | Strings do painel AIGCD (6 idiomas) |

---

## Qualidade do codigo

O codigo passou por code review seguindo os padroes do projeto IPED:

### Thread safety em `AIGCDTask`
Uso de `AtomicBoolean` + `synchronized` no `init()` e `finish()`, igual ao
`ImageSimilarityTask`. Evita dupla inicializacao do modelo quando multiplas
threads Worker do IPED chamam `init()` simultaneamente.

```java
private static final AtomicBoolean init     = new AtomicBoolean(false);
private static final AtomicBoolean finished = new AtomicBoolean(false);
private static volatile OrtEnvironment env;
private static volatile OrtSession session;

public void init(ConfigurationManager configurationManager) throws Exception {
    synchronized (init) {
        if (init.get()) return;
        // ... carrega modelo ...
        init.set(true);
    }
}
```

### Habilitacao via IPEDConfig.txt
Usa `EnableTaskProperty` (padrao do IPED) para ler `enableAIGCDDetector` do
`IPEDConfig.txt`. A task so roda se explicitamente habilitada.

```java
public static final String ENABLE_PARAM = "enableAIGCDDetector";

@Override
public List<Configurable<?>> getConfigurables() {
    return Arrays.asList(new EnableTaskProperty(ENABLE_PARAM));
}
```

### Verificacao `isToAddToCase()`
`process()` verifica se o item deve ser indexado antes de processar, seguindo
o padrao de todos os tasks do IPED.

### Internacionalizacao
`AIGCDPanel` usa `Messages.getString()` para todos os textos visiveis.
Strings adicionadas nos 6 arquivos de localizacao do IPED (en, pt-BR, de, es, fr, it).

### Licenca GPL
`AIGCDPanel.java` e `AIGCDTask.java` incluem o cabecalho GPL padrao do projeto.

---

## Fluxo tecnico

### Fase 1: Processamento (indexacao)

```
IPEDConfig.txt
  enableAIGCDDetector = true
        |
        v
TaskInstaller.xml
  <task class="iped.aigcd.AIGCDTask"/>
        |
        v
TaskInstallerConfig.java
  Class.forName("iped.aigcd.AIGCDTask")  <- JAR deve estar em lib/
        |
        v
AIGCDTask.init()  [synchronized, AtomicBoolean]
  le enableAIGCDDetector do IPEDConfig.txt
  carrega models/convnext-ai-detector/model.onnx
        |
        v  (para cada arquivo)
AIGCDTask.process(IItem item)
  isToAddToCase() && isImage()
  Resize(288) -> CenterCrop(256) -> Normalize ImageNet
  ONNX Runtime -> logits [real, fake]
  score = softmax(fake) = exp(fake) / (exp(real) + exp(fake))
  item.setExtraAttribute("aigcd:score:image", score)
        |
        v
Lucene Index
  FloatPoint + StoredField "aigcd:score:image"
```

### Fase 2: Visualizacao (SearchApp)

```
Usuario clica em um arquivo
        |
        v
FileProcessor.java
  App.get().aigcdPanel.loadDoc(doc)
        |
        v
AIGCDPanel.loadDoc(Document doc)
  parseScore(doc, "aigcd:score:image")  <- le do Lucene
  applyScore(int img)
  badge HIGH/MEDIUM/LOW + barra de progresso
```

---

## Como compilar e testar

### Requisitos

- Java 11 Liberica Full JDK (com JavaFX)
- Maven 3.9+
- Python 3.9.x (para Face Recognition e Age Estimation)

### Build

```powershell
# Build completo
cd C:\IPED\IPED
mvn install -DskipTests

# Build rapido (so os modulos alterados)
mvn install -pl iped-app,iped-aigcd-plugin --am -DskipTests
```

### Processar um caso de teste

```powershell
cd C:\IPED\IPED\target\release\iped-4.4.0-SNAPSHOT
.\iped.exe -d C:\IPED\amostras-aigcd -o C:\IPED\casos\caso-aigcd-XX
```

### Copiar plugins para o caso (obrigatorio apos processar)

```powershell
mkdir C:\IPED\casos\caso-aigcd-XX\plugins

copy "C:\IPED\IPED\target\release\iped-4.4.0-SNAPSHOT\lib\iped-aigcd-plugin-4.4.0-SNAPSHOT.jar" `
     C:\IPED\casos\caso-aigcd-XX\plugins\

copy "C:\IPED\IPED\target\release\plugins\onnxruntime-1.17.0.jar" `
     C:\IPED\casos\caso-aigcd-XX\plugins\

copy "C:\IPED\IPED\target\release\iped-4.4.0-SNAPSHOT\lib\iped-search-app.jar" `
     "C:\IPED\casos\caso-aigcd-XX\iped\lib\" -Force
```

> **Por que copiar o iped-search-app.jar?**
> O JAR dentro da pasta do caso pode ser de uma versao anterior sem `AIGCDPanel`.
> A copia garante que o painel visual vai aparecer corretamente.

> **Por que a pasta plugins/?**
> O Bootstrap do SearchApp monta o classpath como
> `lib/iped-search-app.jar` + `plugins/*`. Sem os JARs em `plugins/`, o
> SearchApp nao encontra as classes do plugin ao abrir o caso.

### Abrir o caso

```powershell
cd C:\IPED\casos\caso-aigcd-XX
.\IPED-SearchApp.exe
```

### Modelo ONNX

Se o modelo nao existir, a task registra aviso e continua sem calcular score.

Caminho esperado:
```
C:\IPED\IPED\target\release\iped-4.4.0-SNAPSHOT\models\convnext-ai-detector\model.onnx
```

Apos `mvn clean`, o modelo e apagado junto com `target/`. Copiar novamente.

---

## Campos de metadados

| Campo | Modalidade | Status |
|---|---|---|
| `aigcd:score:image` | Imagem | Implementado e validado |
| `aigcd:score:video` | Video | Trabalho futuro |
| `aigcd:score:audio` | Audio | Trabalho futuro |
| `aigcd:score:text` | Texto | Trabalho futuro |

Convencao de score:
- `0.00` a `0.39` = LOW (baixa suspeita)
- `0.40` a `0.69` = MEDIUM (suspeita media)
- `0.70` a `1.00` = HIGH (alta suspeita)

Os scores sao indicadores probabilisticos. A decisao final deve ser feita por
um perito humano com contexto e validacao adequados.

---

## Pontos de atencao tecnica

### 1. Task isolada em JAR, UI integrada no fonte

O `AIGCDTask` fica em modulo proprio (`iped-aigcd-plugin`), com JAR copiado
para `lib/` e registrado em `TaskInstaller.xml`. Isolamento completo.

O `AIGCDPanel` (UI) ainda modifica `iped-app`, porque o IPED nao tem contrato
publico para plugins criarem abas/paineis da UI sem alterar `App.java` e
`FileProcessor.java`. Isso e uma limitacao arquitetural documentada do IPED.

### 2. Por que o JAR precisa estar em lib/ E plugins/

- `lib/iped-aigcd-plugin.jar`: necessario para o **engine** carregar a task via
  `Class.forName()` durante o processamento.
- `plugins/iped-aigcd-plugin.jar`: necessario para o **SearchApp** encontrar a
  classe via `plugins/*` no classpath ao abrir o caso.

O `pom.xml` do plugin cuida do `lib/` automaticamente. A pasta `plugins/` do
caso precisa ser preenchida manualmente apos o processamento (ver secao acima).

### 3. Localizacao e icones

Strings visiveis no painel estao em `iped-desktop-messages*.properties`.
Labels dos filtros estao em `iped-ai-filters*.properties`.
Icones dos filtros estao em `iped-app/src/main/resources/iped/app/ui/filter/AIGCDImage.*.png`.

---

## Validacao experimental

Testes com seis imagens apos implementacao completa:

| Imagem | Tipo | Score | Classificacao |
|---|---|---|---|
| Pizza.png | IA (gerada) | 97.43% | HIGH |
| Gatos.png | IA (gerada) | 97.17% | HIGH |
| Idosa.jpg | Real (foto) | ~2% | LOW |
| Mulher nova.jpg | Real (foto) | ~2% | LOW |
| Gato.jpeg | Real (foto) | 15.47% | LOW |
| Pizzas.jpeg | Real (foto) | 2.87% | LOW |

O modelo mostrou boa separacao nas amostras iniciais. Deve ser avaliado com
conjunto maior e mais diverso para o TCC.

---

## Git e repositorio

Fork do projeto: https://github.com/Michelangelo-Costa/IPED/tree/tcc-aigcd

```powershell
# Clonar a branch do TCC
git clone -b tcc-aigcd https://github.com/Michelangelo-Costa/IPED.git

# Atualizar com mudancas do IPED oficial
git fetch upstream
git merge upstream/master

# Publicar mudancas no fork
git add .
git commit -m "mensagem"
git push origin tcc-aigcd
```

---

## Registro de software

Separacao entre codigo original e contribuicoes:

| Componente | Tipo | Autoria |
|---|---|---|
| `AIGCDTask.java` | Novo arquivo | Criado para o TCC |
| `AIGCDPanel.java` | Novo arquivo | Criado para o TCC |
| `App.java` | Modificado | 3 linhas adicionadas |
| `FileProcessor.java` | Modificado | 1 linha adicionada |
| `TaskInstaller.xml` | Modificado | 1 linha adicionada |
| `AIFiltersConfig.json` | Modificado | Filtro AIGCD adicionado |
| `IPEDConfig.txt` | Modificado | Flag `enableAIGCDDetector` adicionada |
| Arquivos de localizacao | Modificado | 4 chaves por idioma |
| Icones PNG | Novo | 4 arquivos criados |
| Demais arquivos | Original IPED | Licenca GPL |

**Modelo ConvNeXt V2 Base**: treinado por xRayon — verificar licenca antes de distribuir.
**ONNX Runtime**: licenca MIT (Microsoft).
**IPED**: licenca GPL — distribuicoes derivadas devem respeitar a licenca.

---

## Organizacao do TCC

1. **Problema**: Crescimento de conteudos gerados por IA e dificuldade de triagem em pericia digital.
2. **Objetivo geral**: Integrar ao IPED um modulo de apoio a deteccao de conteudo gerado por IA.
3. **Objetivos especificos**:
   - Task de processamento para imagens com inferencia ONNX.
   - Armazenamento de scores no indice Lucene do caso.
   - Filtros pesquisaveis na aba AI (High/Medium/Low).
   - Painel visual com badge e barra de progresso.
   - Avaliacao de desempenho com dataset rotulado.
4. **Artefato**: Plugin AIGCD + integracao no IPED.
5. **Avaliacao**: Metricas de acuracia, precisao, revocacao, F1-score, tempo de processamento.

---

## Roadmap

**Prioridade alta (antes da defesa):**
- Criar dataset rotulado (imagens reais + IA) e medir metricas para o TCC
- Medir tempo medio de processamento por imagem
- Capturas de tela do painel para o texto do TCC

**Prioridade media:**
- Criar `AIGCDConfig.txt` para configurar caminho do modelo sem recompilar
- Adicionar testes unitarios minimos
- Calibrar thresholds com base rotulada

**Prioridade futura (pos-TCC):**
- Implementar deteccao de video, audio e texto
- Preparar PR para o repositorio oficial `sepinf-inc/IPED`
- Propor mecanismo de extensao de UI via plugins ao IPED

---

## Resumo do estado atual

Plugin funcional para imagens. Codigo passou por code review completo seguindo
os padroes do IPED: thread safety com `AtomicBoolean`, habilitacao via
`EnableTaskProperty`, verificacao `isToAddToCase()`, internacionalizacao via
`Messages.getString()`, cabecalhos GPL e icones padronizados.

A Task esta isolada em JAR de plugin. A UI ainda requer modificacao do
codigo-fonte do `iped-app` por limitacao arquitetural do IPED (sem ponto de
extensao formal para paineis). Isso e documentado como motivacao para trabalho futuro.
