"""
Gera o model.onnx usado pelo plugin AIGCD a partir do checkpoint original do
modelo xRayon/convnext-ai-images-detector (ConvNeXt V2 Base treinado com timm,
licença MIT, autor Toufik Abbadi).

O ONNX gerado tem entrada "pixel_values" [batch, 3, 256, 256] (imagem já
pré-processada: lado menor 288 px bicúbico, recorte central 256 px e
normalização ImageNet) e saída "logits" [batch, 2], com o índice 1 = "gerada
por IA". É exatamente o formato consumido pelo AIGCDTask.

Requisitos:
    py -m pip install torch timm onnx onnxruntime

Uso:
    py export_onnx.py --checkpoint checkpoint_phase2.pth --saida model.onnx
    py export_onnx.py --checkpoint checkpoint_phase2.pth --saida model.onnx --referencia <outro model.onnx>

O checkpoint fica no repositório do autor no Hugging Face:
    https://huggingface.co/xRayon/convnext-ai-images-detector
    arquivo: "AI Images Detector/checkpoints/checkpoint_phase2.pth"
"""

import argparse
import hashlib

import numpy as np
import timm
import torch

ARQUITETURA = "convnextv2_base"
TAMANHO_ENTRADA = 256
OPSET = 17


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar_modelo(checkpoint):
    """Recria a rede do treino (timm) e carrega os pesos do checkpoint."""
    modelo = timm.create_model(ARQUITETURA, pretrained=False, num_classes=2)
    estado = torch.load(checkpoint, map_location="cpu", weights_only=True)
    modelo.load_state_dict(estado["model"])
    modelo.eval()
    return modelo


def exportar(modelo, saida):
    exemplo = torch.randn(1, 3, TAMANHO_ENTRADA, TAMANHO_ENTRADA)
    torch.onnx.export(
        modelo,
        exemplo,
        saida,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )


def comparar(modelo_torch, onnx_gerado, onnx_referencia=None, amostras=8, semente=0):
    """Compara os logits do ONNX gerado com o PyTorch e, opcionalmente, com outro ONNX."""
    import onnxruntime as ort

    gerador = np.random.default_rng(semente)
    entrada = gerador.standard_normal((amostras, 3, TAMANHO_ENTRADA, TAMANHO_ENTRADA)).astype(np.float32)

    with torch.no_grad():
        esperado = modelo_torch(torch.from_numpy(entrada)).numpy()

    def rodar(caminho):
        sessao = ort.InferenceSession(caminho, providers=["CPUExecutionProvider"])
        return sessao.run(None, {sessao.get_inputs()[0].name: entrada})[0]

    gerado = rodar(onnx_gerado)
    print("Diferença máxima de logits, ONNX gerado x PyTorch: %.2e" % np.abs(gerado - esperado).max())
    if onnx_referencia:
        referencia = rodar(onnx_referencia)
        print("Diferença máxima de logits, ONNX gerado x referência: %.2e" % np.abs(gerado - referencia).max())


def main():
    ap = argparse.ArgumentParser(description="Exporta o checkpoint do xRayon para o model.onnx do plugin AIGCD.")
    ap.add_argument("--checkpoint", required=True, help="Caminho do checkpoint_phase2.pth")
    ap.add_argument("--saida", default="model.onnx", help="Arquivo ONNX de saída. Padrão: model.onnx")
    ap.add_argument("--referencia", help="Outro model.onnx para comparar as saídas (opcional)")
    args = ap.parse_args()

    print("PyTorch %s | timm %s" % (torch.__version__, timm.__version__))
    modelo = carregar_modelo(args.checkpoint)
    exportar(modelo, args.saida)
    print("Gerado: %s" % args.saida)
    print("SHA-256: %s" % sha256(args.saida))
    comparar(modelo, args.saida, args.referencia)


if __name__ == "__main__":
    main()
