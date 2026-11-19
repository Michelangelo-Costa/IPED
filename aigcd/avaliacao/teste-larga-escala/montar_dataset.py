"""
Monta um conjunto de imagens ROTULADO (real x IA) para o teste em larga escala
do plugin AIGCD no IPED.

O rotulo fica no CAMINHO do arquivo, que o IPED preserva dentro do caso. Assim,
depois de processar e exportar o CSV, o script calcular_metricas.py sabe qual
imagem e real e qual e IA sem nenhuma anotacao manual.

Estrutura gerada:

    <saida>/
        original/
            ia/GenImage_ADM/ ... ia/GenImage_Wukong/   (7 geradores)
            ia/CIFAKE/                                 (opcional, --cifake)
            real/GenImage/
            real/CIFAKE/                               (opcional, --cifake)
        jpeg85/                                        (opcional, --versao-jpeg 85)
            ... mesma estrutura, tudo recomprimido em JPEG ...
        manifesto_original.csv                         (arquivo, rotulo, grupo, sha256)

As imagens de "original" sao gravadas com os BYTES ORIGINAIS do dataset (sem
recompressao), para nao alterar os artefatos que o detector analisa.

A variante JPEG recomprime TODAS as imagens (reais e IA) com a mesma qualidade.
Ela serve para dois fins: (1) eliminar o vies de formato do GenImage, em que as
reais sao JPEG e as geradas costumam ser PNG, e (2) simular o compartilhamento
por redes sociais e aplicativos de mensagem.

Requisitos:  py -m pip install datasets pillow
Exemplo:     py montar_dataset.py --saida C:\\IPED\\teste-macro\\evidencias --por-gerador 60
"""

import argparse
import csv
import hashlib
import io
import os
import sys

GENIMAGE = "TheKernel01/Tiny-GenImage"
CIFAKE = "dragonintelligence/CIFAKE-image-dataset"

# Usados apenas se o streaming nao expuser os metadados do dataset.
GENIMAGE_LABELS = ["real", "fake"]
GENIMAGE_GENERATORS = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]
CIFAKE_LABELS = ["FAKE", "REAL"]

# O Tiny-GenImage declara a classe SD14, mas nao possui nenhuma imagem dela
# (0 exemplos em train e validation). Sem esta exclusao a cota nunca seria
# atingida e o script percorreria o split inteiro.
GENIMAGE_GERADORES_VAZIOS = {"SD14"}
N_GERADORES_GENIMAGE = len(GENIMAGE_GENERATORS) - 1 - len(GENIMAGE_GERADORES_VAZIOS)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def detectar_extensao(dados, caminho_original):
    """Identifica o formato pelos bytes iniciais (assinatura), com fallback no nome."""
    if dados[:3] == b"\xff\xd8\xff":
        return "jpg"
    if dados[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if dados[:4] == b"RIFF" and dados[8:12] == b"WEBP":
        return "webp"
    if dados[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if dados[:2] == b"BM":
        return "bmp"
    if dados[:4] in (b"II*\x00", b"MM\x00*"):
        return "tif"
    ext = os.path.splitext(caminho_original or "")[1].lstrip(".").lower()
    return ext or "bin"


def abrir_stream(nome, split):
    """Abre o dataset em streaming, sem decodificar as imagens (bytes originais)."""
    from datasets import Image, load_dataset

    ds = load_dataset(nome, split=split, streaming=True)
    return ds.cast_column("image", Image(decode=False))


def nomes_da_coluna(ds, coluna, padrao):
    try:
        return ds.features[coluna].names
    except Exception:
        return padrao


class Coletor:
    """Grava as imagens em disco, evita duplicatas e registra o manifesto."""

    def __init__(self, raiz):
        self.raiz = raiz
        self.vistos = set()
        self.contagem = {}
        self.formatos = {}
        self.linhas = []

    def salvar(self, dados, caminho_original, rotulo, grupo, fonte):
        sha = hashlib.sha256(dados).hexdigest()
        if sha in self.vistos:
            return False
        self.vistos.add(sha)

        ext = detectar_extensao(dados, caminho_original)
        n = self.contagem.get(grupo, 0) + 1
        self.contagem[grupo] = n
        self.formatos[ext] = self.formatos.get(ext, 0) + 1

        pasta = os.path.join(self.raiz, rotulo, grupo)
        os.makedirs(pasta, exist_ok=True)
        nome = "%s_%05d.%s" % (grupo, n, ext)
        with open(os.path.join(pasta, nome), "wb") as f:
            f.write(dados)

        relativo = os.path.join(rotulo, grupo, nome)
        self.linhas.append([relativo, rotulo, grupo, fonte, ext, sha])
        return True


def coletar_genimage(coletor, por_gerador, n_reais):
    ds = abrir_stream(GENIMAGE, "validation")
    labels = nomes_da_coluna(ds, "label", GENIMAGE_LABELS)
    geradores = nomes_da_coluna(ds, "generator", GENIMAGE_GENERATORS)

    cotas = {g: por_gerador for g in geradores if g != "Real" and g not in GENIMAGE_GERADORES_VAZIOS}
    cotas["Real"] = n_reais
    obtidos = {g: 0 for g in cotas}
    inconsistentes = 0

    print("[GenImage] cotas: %d por gerador (%d geradores) + %d reais"
          % (por_gerador, len(cotas) - 1, n_reais))

    for i, ex in enumerate(ds):
        if all(obtidos[g] >= cotas[g] for g in cotas):
            break
        if i and i % 500 == 0:
            print("  ... %d linhas lidas, %d imagens salvas" % (i, sum(obtidos.values())))

        rotulo_ds = labels[ex["label"]]
        gerador = geradores[ex["generator"]]

        # Sanidade: imagem "real" precisa vir do gerador "Real" e vice-versa.
        if (rotulo_ds == "real") != (gerador == "Real"):
            inconsistentes += 1
            continue
        if gerador not in cotas or obtidos[gerador] >= cotas[gerador]:
            continue

        img = ex["image"]
        if gerador == "Real":
            ok = coletor.salvar(img["bytes"], img.get("path"), "real", "GenImage", GENIMAGE)
        else:
            ok = coletor.salvar(img["bytes"], img.get("path"), "ia", "GenImage_" + gerador, GENIMAGE)
        if ok:
            obtidos[gerador] += 1

    if inconsistentes:
        print("  [aviso] %d linhas com rotulo/gerador inconsistentes foram ignoradas" % inconsistentes)
    faltando = {g: cotas[g] - obtidos[g] for g in cotas if obtidos[g] < cotas[g]}
    if faltando:
        print("  [aviso] cotas nao atingidas (o split acabou): %s" % faltando)


def coletar_cifake(coletor, por_classe):
    ds = abrir_stream(CIFAKE, "test")
    labels = nomes_da_coluna(ds, "label", CIFAKE_LABELS)
    obtidos = {"FAKE": 0, "REAL": 0}
    print("[CIFAKE] cota: %d por classe (imagens 32x32)" % por_classe)

    for ex in ds:
        if all(v >= por_classe for v in obtidos.values()):
            break
        classe = labels[ex["label"]]
        if obtidos[classe] >= por_classe:
            continue
        img = ex["image"]
        rotulo = "ia" if classe == "FAKE" else "real"
        if coletor.salvar(img["bytes"], img.get("path"), rotulo, "CIFAKE", CIFAKE):
            obtidos[classe] += 1


def gravar_manifesto(caminho, linhas):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "rotulo", "grupo", "fonte", "formato", "sha256"])
        w.writerows(linhas)


def criar_versao_jpeg(raiz_original, raiz_jpeg, linhas, qualidade):
    """Recomprime todas as imagens em JPEG com a mesma qualidade (mesma estrutura)."""
    from PIL import Image

    novas = []
    for relativo, rotulo, grupo, fonte, _, _ in linhas:
        origem = os.path.join(raiz_original, relativo)
        destino_rel = os.path.splitext(relativo)[0] + ".jpg"
        destino = os.path.join(raiz_jpeg, destino_rel)
        os.makedirs(os.path.dirname(destino), exist_ok=True)

        with Image.open(origem) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=qualidade)
        dados = buf.getvalue()
        with open(destino, "wb") as f:
            f.write(dados)
        novas.append([destino_rel, rotulo, grupo, fonte + " (JPEG q=%d)" % qualidade, "jpg",
                      hashlib.sha256(dados).hexdigest()])
    return novas


def main():
    ap = argparse.ArgumentParser(description="Monta dataset rotulado real x IA para testar o AIGCD no IPED.")
    ap.add_argument("--saida", required=True, help="Pasta raiz das evidencias (sera processada pelo IPED)")
    ap.add_argument("--por-gerador", type=int, default=60,
                    help="Imagens de IA por gerador do GenImage (7 geradores, max. 500). Padrao: 60")
    ap.add_argument("--reais", type=int, default=None,
                    help="Imagens reais do GenImage. Padrao: igual ao total de IA (dataset balanceado)")
    ap.add_argument("--cifake", type=int, default=0,
                    help="Imagens por classe do CIFAKE (32x32, teste de baixa resolucao). Padrao: 0")
    ap.add_argument("--versao-jpeg", type=int, default=0, metavar="QUALIDADE",
                    help="Cria tambem a variante com tudo recomprimido em JPEG (ex.: 85). Padrao: desligado")
    args = ap.parse_args()

    raiz_original = os.path.join(args.saida, "original")
    if os.path.isdir(raiz_original) and os.listdir(raiz_original):
        sys.exit("A pasta %s ja existe e nao esta vazia. Use outra --saida ou apague-a." % raiz_original)

    n_reais = args.reais if args.reais is not None else args.por_gerador * N_GERADORES_GENIMAGE

    coletor = Coletor(raiz_original)
    coletar_genimage(coletor, args.por_gerador, n_reais)
    if args.cifake > 0:
        coletar_cifake(coletor, args.cifake)

    gravar_manifesto(os.path.join(args.saida, "manifesto_original.csv"), coletor.linhas)

    n_ia = sum(1 for l in coletor.linhas if l[1] == "ia")
    n_real = len(coletor.linhas) - n_ia
    print("\nOK: %d imagens em %s (IA=%d, Real=%d)" % (len(coletor.linhas), raiz_original, n_ia, n_real))
    print("Formatos: %s" % coletor.formatos)
    if coletor.formatos.get("webp") or coletor.formatos.get("tif"):
        print("[aviso] Ha WEBP/TIFF: confira no relatorio se essas imagens receberam score no IPED.")

    if args.versao_jpeg:
        raiz_jpeg = os.path.join(args.saida, "jpeg%d" % args.versao_jpeg)
        print("\nCriando variante JPEG q=%d em %s ..." % (args.versao_jpeg, raiz_jpeg))
        novas = criar_versao_jpeg(raiz_original, raiz_jpeg, coletor.linhas, args.versao_jpeg)
        gravar_manifesto(os.path.join(args.saida, "manifesto_jpeg%d.csv" % args.versao_jpeg), novas)
        print("OK: %d imagens recomprimidas" % len(novas))


if __name__ == "__main__":
    main()
