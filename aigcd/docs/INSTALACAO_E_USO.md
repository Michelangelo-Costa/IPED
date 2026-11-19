# Instalação e uso do AIGCD

Guia para reproduzir o ambiente do zero em uma máquina Windows 10/11 (64 bits): compilar o
IPED com o AIGCD, obter o modelo, processar um caso e usar o resultado na interface.

Tempo estimado na primeira vez: 30 a 60 minutos, a maior parte em downloads.

---

## 1. Pré-requisitos

| Ferramenta | Versão | Observação |
|---|---|---|
| Git | qualquer recente | [git-scm.com](https://git-scm.com/) |
| JDK | **11**, Liberica **Full** JDK (inclui JavaFX) | [bell-sw.com/pages/downloads](https://bell-sw.com/pages/downloads/#jdk-11-lts) |
| Maven | 3.9 ou superior | [maven.apache.org](https://maven.apache.org/download.cgi) |
| Espaço em disco | cerca de 15 GB | código, dependências do Maven e versão compilada |
| Internet | apenas na compilação e no download do modelo | o uso do AIGCD é totalmente offline |

O AIGCD **não precisa de Python**. Python só é necessário para as funções opcionais de
reconhecimento facial e estimativa de idade do IPED (seção 8).

Confira as instalações em um PowerShell novo:

```powershell
git --version
java -version     # deve mostrar versão 11
mvn -version      # deve mostrar Maven 3.9+ usando o Java 11
```

Se `mvn -version` mostrar outro Java, ajuste a variável de ambiente `JAVA_HOME` para a pasta
do JDK 11 (por exemplo `C:\Program Files\BellSoft\LibericaJDK-11-Full`) e abra um novo
PowerShell.

---

## 2. Obter o código

```powershell
cd C:\
git clone -b tcc-aigcd https://github.com/Michelangelo-Costa/IPED.git
cd C:\IPED
```

> Para um clone mais rápido, sem o histórico completo do IPED:
> `git clone --depth 1 -b tcc-aigcd https://github.com/Michelangelo-Costa/IPED.git`

Os comandos a seguir assumem que o repositório está em `C:\IPED`. Ajuste os caminhos se
clonar em outro lugar.

---

## 3. Compilar

```powershell
cd C:\IPED
mvn clean install -DskipTests
```

A primeira compilação baixa as dependências do IPED (JRE embutida, Python embutido,
bibliotecas e modelos do próprio IPED) e pode levar de 15 a 40 minutos. As seguintes são bem
mais rápidas.

Ao final deve aparecer `BUILD SUCCESS`, e a distribuição pronta fica em:

```
C:\IPED\target\release\iped-4.4.0-SNAPSHOT\
```

Para recompilar só o que o AIGCD altera (mais rápido, depois da primeira vez):

```powershell
mvn install -pl iped-app,iped-aigcd-plugin -am -DskipTests
```

---

## 4. Obter o modelo

O modelo (ConvNeXt V2 Base exportado para ONNX, 335 MB, licença MIT) não fica no Git por causa
do tamanho. Ele está publicado nos Releases do fork:

```powershell
$REL = "C:\IPED\target\release\iped-4.4.0-SNAPSHOT"
New-Item -ItemType Directory -Force "$REL\models\convnext-ai-detector" | Out-Null
curl.exe -L -o "$REL\models\convnext-ai-detector\model.onnx" https://github.com/Michelangelo-Costa/IPED/releases/download/aigcd-modelo-v1.0/model.onnx
```

Confira a integridade do arquivo:

```powershell
(Get-FileHash "$REL\models\convnext-ai-detector\model.onnx" -Algorithm SHA256).Hash
```

O resultado deve ser:

```
D9B32082803010AB2EB701F193FD853373A3D49F8429BFF7BB63EC8664F907FB
```

Se preferir gerar o modelo a partir do checkpoint original do autor, siga
[../modelo/README.md](../modelo/README.md). O resultado é idêntico (mesmo SHA-256).

> **Atenção:** `mvn clean` apaga a pasta `target`, inclusive o modelo. Guarde uma cópia do
> `model.onnx` fora do repositório e copie de novo após um `mvn clean`.

---

## 5. Processar um caso

O AIGCD vem habilitado (`enableAIGCDDetector = true` em `IPEDConfig.txt`). Processe qualquer
pasta, imagem de disco ou evidência suportada pelo IPED:

```powershell
cd C:\IPED\target\release\iped-4.4.0-SNAPSHOT
.\iped.exe -d C:\minhas-imagens -o C:\casos\caso-teste
```

Para confirmar que o AIGCD rodou, procure no log do processamento
(`target\release\iped-4.4.0-SNAPSHOT\log\IPED-<data>.log`):

```
[iped.aigcd.AIGCDTask]  Model loaded from ...\models\convnext-ai-detector\model.onnx
...
AIGCDTask      <segundos>   <percentual>
```

Mensagens `Delete failed on '...onnxruntime.dll'` no final do log são inofensivas: o Windows
apenas impede a remoção da DLL temporária enquanto ela ainda está carregada.

---

## 6. Usar na interface

```powershell
C:\casos\caso-teste\IPED-SearchApp.exe
```

1. Abra a aba **IA** (*AI* com a interface em inglês), no canto inferior esquerdo.
2. **Painel AIGCD** (topo da aba): ao clicar em uma imagem da tabela, mostra o selo
   **ALTO / MÉDIO / BAIXO**, a barra com o score em porcentagem e uma nota lembrando que a
   decisão final é do perito.
3. **Filtros** (árvore abaixo do painel): em **AIGCD - Score de Imagem**, clique em **Alto**,
   **Médio** ou **Baixo** para listar só as imagens daquela faixa. A soma das três faixas é
   igual ao total de imagens com score.
4. Use a aba **Galeria** para ver em miniaturas as imagens filtradas.

### Idioma da interface

O idioma é definido por `locale` no `LocalConfig.txt`:

- para casos novos: `C:\IPED\target\release\iped-4.4.0-SNAPSHOT\LocalConfig.txt`;
- para um caso já processado: `<caso>\iped\LocalConfig.txt`.

Exemplos: `locale = pt-BR`, `locale = en`. Os filtros do AIGCD funcionam em todos os idiomas
do IPED.

### Exportar os scores para planilha

1. Na tabela de resultados, clique em uma linha e pressione **Ctrl+A**.
2. Botão direito → **Exportar propriedades dos itens destacados** (*Export highlighted items
   properties*).
3. Em **Exibir colunas** (*Show Columns*), escolha **Todas** (*All*), digite `aigcd` no
   campo de filtro e marque **aigcd:score:image**; marque também **path**.
4. Salve o arquivo `.csv`.

Esse CSV é a entrada das ferramentas de avaliação em [../avaliacao/](../avaliacao/).

---

## 7. Desabilitar o AIGCD

Em `C:\IPED\target\release\iped-4.4.0-SNAPSHOT\IPEDConfig.txt`:

```
enableAIGCDDetector = false
```

Casos processados assim não têm score: o painel mostra **N/D** e os filtros ficam vazios. O
score é calculado durante o processamento; para obtê-lo, o caso precisa ser processado
novamente com o AIGCD habilitado.

---

## 8. Opcional: reconhecimento facial e estimativa de idade

Essas funções são do IPED, não do AIGCD, e vêm desabilitadas. Elas aparecem na mesma aba
**IA** e foram usadas junto com o AIGCD nos ensaios do TCC. Para habilitá-las:

1. Em `IPEDConfig.txt` da versão compilada:
   ```
   enableFaceRecognition = true
   enableAgeEstimation = true
   ```
2. Instale as dependências Python seguindo o manual oficial do IPED:
   [reconhecimento facial](https://github.com/sepinf-inc/IPED/wiki/User-Manual#facerecognition)
   e [estimativa de idade](https://github.com/sepinf-inc/IPED/wiki/User-Manual#AgeEstimation).
3. Na primeira execução, a estimativa de idade baixa o modelo dela do Hugging Face para
   `models\age_estimation`. Em uma máquina sem internet, copie essa pasta de outra instalação.

---

## 9. Solução de problemas

| Sintoma | Causa provável | Solução |
|---|---|---|
| Log mostra `convnext-ai-detector model not found` | Modelo ausente ou no caminho errado | Refaça a seção 4 e confira o SHA-256 |
| Painel mostra **N/D** em todas as imagens | Caso processado sem o AIGCD (flag desligada ou modelo ausente) | Corrija e processe o caso novamente |
| Filtro **Alto** vazio, mas o painel mostra scores altos | Caso aberto com uma versão anterior a esta correção | Processe o caso com a versão atual |
| `mvn` usa Java diferente de 11 | `JAVA_HOME` apontando para outro JDK | Ajuste `JAVA_HOME` (seção 1) |
| Modelo sumiu depois de compilar | `mvn clean` apagou `target` | Copie o `model.onnx` de novo (seção 4) |
| Imagens WEBP ou HEIC sem score | O leitor de imagens do Java não suporta esses formatos | Limitação conhecida; converta para JPEG/PNG se necessário |
