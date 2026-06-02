# Guia de Teste — Plugin AIGCD para IPED

Este guia explica como instalar, configurar e testar o plugin de detecção de
conteúdo gerado por IA (AIGCD) integrado ao IPED.

---

## Pré-requisitos

| Requisito | Versão | Observação |
|---|---|---|
| Java JDK | 11 (com JavaFX) | Recomendado: Liberica Full JDK 11 |
| Maven | 3.9+ | |
| Python | 3.9.x | Para Face Recognition e Age Estimation |
| Git | qualquer | |

### Instalar Java 11 Liberica (com JavaFX)

Baixar em: https://bell-sw.com/pages/downloads/#jdk-11-lts

Escolher **Full JDK** (inclui JavaFX).

### Verificar instalação

```powershell
java -version
mvn -version
python --version
```

---

## 1. Clonar o repositório

```powershell
git clone -b tcc-aigcd https://github.com/Michelangelo-Costa/IPED.git
cd IPED
```

---

## 2. Baixar o modelo ONNX

O modelo de detecção de IA não está no repositório (334 MB). Baixe separadamente:

**Opção A — Baixar diretamente no drive** (https://drive.google.com/file/d/1Kfit6v2LSwYkfjol7m9RC7PSESPVbpx_/view?usp=sharing)

**Opção B — Exportar do checkpoint** (requer Python + torch + timm):
```powershell
# Instalar dependências
py -m pip install torch timm onnx onnxscript

# Exportar (executar da pasta raiz do projeto)
# O script export_model.py está disponível separadamente
```

Após obter o `model.onnx`, copiar para:

```
IPED\target\release\iped-4.4.0-SNAPSHOT\models\convnext-ai-detector\model.onnx
```

> A pasta `models\convnext-ai-detector\` precisa ser criada manualmente se não existir.

---

## 3. Compilar o projeto

```powershell
cd IPED
mvn install -pl iped-app,iped-aigcd-plugin --am -DskipTests
```

O build leva alguns minutos. Ao terminar, a distribuição estará em:

```
IPED\target\release\iped-4.4.0-SNAPSHOT\
```

---

## 4. Instalar dependências Python (para Face Recognition e Age Estimation)

O IPED usa um Python embutido em `target\release\iped-4.4.0-SNAPSHOT\python\`.
As dependências já foram instaladas na máquina do desenvolvedor.

Se precisar reinstalar:

```powershell
# Usar pip apontando para o Python embutido do IPED
$SITE = "IPED\target\release\iped-4.4.0-SNAPSHOT\python\lib\site-packages"
py -m pip install face_recognition opencv-python "transformers==4.40.0" "huggingface-hub==0.23.0" "numpy==1.26.4" --target $SITE --python-version 3.9 --only-binary=:all:
```

---

## 5. Preparar imagens de teste

Crie uma pasta com imagens para processar. Misture imagens reais e geradas por IA:

```
C:\imagens-teste\
    foto_real_1.jpg
    foto_real_2.jpg
    imagem_ia_1.png
    imagem_ia_2.jpg
    ...
```

> **Dica:** Para imagens com rostos, o IPED também vai mostrar detecção de faces
> e estimativa de idade. Use fotos com pessoas para ver todas as funcionalidades.

---

## 6. Processar um caso

```powershell
cd IPED\target\release\iped-4.4.0-SNAPSHOT
.\iped.exe -d C:\imagens-teste -o C:\casos\meu-caso-01
```

O processamento pode levar alguns minutos dependendo da quantidade de imagens.
Acompanhe pelo log no terminal.

---

## 7. Copiar o plugin para o caso

Após o processamento terminar, copie os JARs necessários:

```powershell
mkdir C:\casos\meu-caso-01\plugins

copy "IPED\target\release\iped-4.4.0-SNAPSHOT\lib\iped-aigcd-plugin-4.4.0-SNAPSHOT.jar" C:\casos\meu-caso-01\plugins\

copy "IPED\target\release\plugins\onnxruntime-1.17.0.jar" C:\casos\meu-caso-01\plugins\

copy "IPED\target\release\iped-4.4.0-SNAPSHOT\lib\iped-search-app.jar" "C:\casos\meu-caso-01\iped\lib\" -Force
```

---

## 8. Abrir o caso na interface

```powershell
cd C:\casos\meu-caso-01
.\IPED-SearchApp.exe
```

---

## 9. Usando as funcionalidades de IA

### Aba AI — Filtros

No painel inferior esquerdo, clique na aba **AI**. Você verá:

```
▼ AI
  ▼ Detected Faces (N)       ← rostos detectados
  ▼ Estimated Age (N)         ← faixa etária estimada
  ▼ AIGCD - Image Score (N)   ← score de detecção de IA
      High (>=70%)             ← provável gerado por IA
      Medium (40-69%)          ← inconclusivo
      Low (<40%)               ← provável imagem real
```

- Clique em **High** para filtrar apenas imagens com alta suspeita de serem geradas por IA
- Clique em **Detected Faces > 1** para ver imagens com rostos

### Painel AIGCD

Ao clicar em uma imagem na tabela de resultados, o painel no canto superior
esquerdo da aba AI mostra:

- **Badge HIGH / MEDIUM / LOW** — classificação geral
- **Barra de progresso** — score de 0% a 100%

> Score próximo de 100% = alta probabilidade de ser gerado por IA  
> Score próximo de 0% = alta probabilidade de ser imagem real

### Aba Gallery

Clique em **Gallery** para ver as imagens em grade e identificar visualmente
padrões entre as classificadas como HIGH.

---

## 10. Entendendo os resultados

| Score | Classificação | Interpretação |
|---|---|---|
| >= 70% | HIGH | Provável conteúdo gerado por IA |
| 40–69% | MEDIUM | Inconclusivo — requer análise manual |
| < 40% | LOW | Provável imagem real |

> **Importante:** Os scores são indicadores probabilísticos. A decisão final
> deve ser feita por um perito humano com contexto e validação adequados.
> O modelo pode apresentar falsos positivos e falsos negativos.

---

## Problemas comuns

| Erro | Causa | Solução |
|---|---|---|
| `ClassNotFoundException: iped.aigcd.AIGCDTask` | JARs não foram copiados para `plugins/` | Refaça o passo 7 |
| Painel mostra N/A ao clicar na imagem | `iped-search-app.jar` desatualizado | Refaça a cópia do JAR no passo 7 |
| Face recognition desabilitado no log | Dependências Python ausentes | Refaça o passo 4 |
| Modelo não encontrado | `model.onnx` não está no caminho correto | Refaça o passo 2 |

---

## Contato

Dúvidas ou problemas: **Michelangelo Costa** — [GitHub](https://github.com/Michelangelo-Costa/IPED/tree/tcc-aigcd)
