# Modelo de detecção (ConvNeXt V2 Base, ONNX)

O AIGCD usa o modelo **AI Images Detector**, de Toufik Abbadi (xRayon): uma rede ConvNeXt V2
Base (`convnextv2_base` do timm, pré-treinada no ImageNet e ajustada para 2 classes, real e
gerada por IA).

| Item | Valor |
|---|---|
| Repositório original | [huggingface.co/xRayon/convnext-ai-images-detector](https://huggingface.co/xRayon/convnext-ai-images-detector) |
| Licença | MIT, Copyright (c) 2026 Toufik Abbadi ([LICENSE](LICENSE)) |
| Checkpoint | `AI Images Detector/checkpoints/checkpoint_phase2.pth` (cerca de 1 GB) |
| Arquivo usado pelo plugin | `model.onnx`, cerca de 335 MB |
| SHA-256 do `model.onnx` | `d9b32082803010ab2eb701f193fd853373a3d49f8429bff7bb63ec8664f907fb` |
| Onde colocar | `<raiz do IPED>/models/convnext-ai-detector/model.onnx` |

A licença MIT permite redistribuir o modelo convertido, desde que o aviso de copyright e a
licença acompanhem o arquivo. Por isso este diretório inclui o [LICENSE](LICENSE) original.

## Opção 1: baixar o arquivo pronto

O `model.onnx` está publicado nos Releases deste fork:

```powershell
curl.exe -L -o model.onnx https://github.com/Michelangelo-Costa/IPED/releases/download/aigcd-modelo-v1.0/model.onnx
(Get-FileHash model.onnx -Algorithm SHA256).Hash
# D9B32082803010AB2EB701F193FD853373A3D49F8429BFF7BB63EC8664F907FB
```

## Opção 2: gerar a partir do checkpoint original

O script [export_onnx.py](export_onnx.py) recria a rede com o timm, carrega os pesos do
checkpoint e exporta para ONNX no formato que o `AIGCDTask` consome.

```powershell
# 1. Dependências
py -m pip install torch==2.12.0 timm==1.0.27 onnx onnxruntime

# 2. Checkpoint do autor (cerca de 1 GB)
curl.exe -L -o checkpoint_phase2.pth "https://huggingface.co/xRayon/convnext-ai-images-detector/resolve/main/AI%20Images%20Detector/checkpoints/checkpoint_phase2.pth"

# 3. Exportação
py export_onnx.py --checkpoint checkpoint_phase2.pth --saida model.onnx
```

Com PyTorch 2.12.0 e timm 1.0.27, o arquivo gerado é **idêntico byte a byte** ao publicado
(mesmo SHA-256). Com outras versões, os bytes podem mudar, mas as saídas do modelo
continuam as mesmas; para conferir, compare com um ONNX de referência:

```powershell
py export_onnx.py --checkpoint checkpoint_phase2.pth --saida model.onnx --referencia <outro model.onnx>
```

O script imprime a diferença máxima entre os logits do ONNX gerado, do modelo em PyTorch e
do ONNX de referência.

## Formato do ONNX

| Item | Valor |
|---|---|
| Opset | 17 (exportador clássico do PyTorch, `dynamo=False`) |
| Entrada | `pixel_values`, float32 [lote, 3, 256, 256], já pré-processada |
| Saída | `logits`, float32 [lote, 2]; índice 1 = gerada por IA |

O pré-processamento (lado menor para 288 px bicúbico, recorte central 256 px, normalização
ImageNet) é feito em Java pelo `AIGCDTask` e reproduz o `test_transforms` usado no treinamento
do modelo.
