# -*- coding: utf-8 -*-
"""
Builder do Notebook — Parte 1 (Avaliacao comparativa de detectores pre-treinados,
em tres conjuntos com dificuldade crescente). Gera o .ipynb via nbformat.
Cada celula de codigo e armazenada como raw-string r'''...'''.
REGRA: nunca usar tres aspas simples dentro das celulas (apenas aspas duplas).
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))

def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ======================================================================
md(r'''
# Avaliacao Comparativa de Detectores de Imagens Geradas por IA — Parte 1

**Estudo experimental de apoio ao Plugin AIGCD (IPED)**

Programa de Pos-Graduacao em Ciencias Forenses (PPGCF) — UNIFESSPA.
Trabalho de Conclusao de Curso (Michelangelo de Carvalho) e Projeto de Dissertacao (Wemelly Santos Nunes).

Este notebook conduz a avaliacao experimental comparativa de **detectores pre-treinados** de imagens
geradas por Inteligencia Artificial. O modelo utilizado pela versao 1.0 do plugin AIGCD (ConvNeXt V2 Base,
em formato ONNX) e incluido como **referencia (baseline)**.

A avaliacao e conduzida sobre **tres conjuntos de dados com dificuldade crescente**, o que permite observar
como o desempenho dos detectores varia a medida que as imagens sinteticas se tornam mais realistas —
questao central para a aplicacao forense:

1. **AI-vs-Real (estilizado)** — imagens de IA predominantemente em estilo artistico/ilustracao. Constitui
   o caso mais **facil** e serve como limite superior de desempenho.
2. **GenImage (fotorrealista)** — imagens **fotorrealistas** de multiplos geradores modernos (Stable
   Diffusion, Midjourney, ADM, GLIDE, VQDM, Wukong, BigGAN) contra fotografias reais do ImageNet. Representa
   o caso de **maior relevancia pericial**: a imagem sintetica e dificil de distinguir de uma foto autentica.
3. **CIFAKE (32x32)** — CIFAR-10 (real) e Stable Diffusion 1.4 (sintetico) em baixa resolucao. Teste de
   **robustez** sob mudanca de resolucao e de dominio.

O experimento produz as metricas previstas na documentacao do plugin: acuracia, precisao, revocacao, F1,
ROC-AUC, EER, matriz de confusao, curvas ROC e Precisao-Revocacao, calibracao, tempo medio de inferencia,
e a distribuicao das evidencias pelas faixas do plugin (LOW / MEDIUM / HIGH).

Esta e a **Parte 1** (modelos ja treinados; nao ha treinamento). A **Parte 2**, documentada separadamente,
aborda o treinamento de modelos (MesoNet, EfficientNetV2 e uma CNN de base).

## Modelos avaliados

| Modelo | Arquitetura | Funcao no experimento |
|---|---|---|
| ConvNeXt V2 (ONNX) | ConvNeXtV2-Base | Referencia (baseline): modelo empregado pela versao 1.0 do plugin |
| SDXL-Detector | Swin | Detector representativo da familia Transformer |
| AI-image-detector | Swin | Detector generalista de referencia |
| AIorNot | Swin-Tiny | Detector alternativo |
| DeepFake-Detector-v2 | ViT | Detector orientado a deepfake |
| AIvsReal (dima806) | ViT | Detector generalista adicional |

## Modelos considerados e nao incluidos nesta etapa

- **DIRE** — requer reconstrucao das imagens por um modelo de difusao (custo elevado, baixa reprodutibilidade).
- **NoisePrint** — assinatura de camera e localizacao de manipulacoes, nao sendo classificador binario real/IA.
- **CNNSpot e UnivFD** — pesos em repositorios externos de disponibilidade intermitente.
''')

# ======================================================================
md(r'''
## Instrucoes de execucao

1. Ambiente com GPU (opcional): *Runtime -> Change runtime type -> GPU*. A inferencia do modelo ONNX e
   executada em CPU (modo de operacao do plugin); a GPU acelera os modelos Transformer.
2. Executar as celulas em ordem (*Runtime -> Run all*). Quando a janela de autorizacao do Google Drive
   aparecer, **concluir a autorizacao** (escolher a conta e clicar em Permitir).
3. O experimento avalia todos os modelos em tres conjuntos de dados; o tempo total e da ordem de dezenas de
   minutos (a inferencia ONNX em CPU e o componente mais lento). Para reduzir, diminuir `N_PER_CLASS`.
4. Ao final, as metricas (CSV) e as figuras sao salvas em `aigcd_resultados/` e compactadas em `.zip`.
''')

# ======================================================================
md(r'''---
## 1. Instalacao das dependencias''')

code(r'''
import sys, subprocess

IN_COLAB = "google.colab" in sys.modules

def pip_install(pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs], check=False)

pip_install(["transformers", "timm", "onnxruntime", "datasets", "accelerate"])
print("Ambiente:", "Google Colab" if IN_COLAB else "Local/Jupyter")
print("Instalacao concluida.")
''')

# ======================================================================
md(r'''## 2. Importacoes e configuracao''')

code(r'''
import os, time, hashlib, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import onnxruntime as ort
from datasets import load_dataset

from sklearn.metrics import (roc_curve, roc_auc_score, average_precision_score,
                             precision_recall_curve, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# Configuracao do experimento
# =====================================================================
N_PER_CLASS = 400            # imagens por classe, em cada conjunto de dados
BATCH_SIZE  = 16             # tamanho do lote para os modelos Transformer
MAX_SCAN    = 60000          # limite de varredura ao coletar em streaming

# Conjunto 1 — estilizado (caso facil / limite superior)
DATASET_ID     = "Parveshiiii/AI-vs-Real"
DATASET_SPLIT  = "train"
AI_LABEL_VALUE = 0           # 0 = IA, 1 = Real (verificar na secao 5.1)
DATASET_TITLE  = "AI-vs-Real (estilizado)"

# Conjunto 2 — fotorrealista (caso dificil / maior relevancia pericial)
GENIMAGE_ID       = "TheKernel01/Tiny-GenImage"
GENIMAGE_SPLIT    = "validation"
GENIMAGE_AI_LABEL = 1         # neste conjunto: 0 = real, 1 = IA
GENIMAGE_TITLE    = "GenImage (fotorrealista)"

# Conjunto 3 — baixa resolucao (robustez)
CIFAKE_ID       = "dragonintelligence/CIFAKE-image-dataset"
CIFAKE_SPLIT    = "test"
CIFAKE_AI_LABEL = 0           # CIFAKE: 0 = FAKE (IA), 1 = REAL
CIFAKE_TITLE    = "CIFAKE (32x32)"

# Modelos do repositorio Hugging Face. Falha no carregamento de um modelo nao interrompe o experimento.
HF_MODELS = [
    "Organika/sdxl-detector",
    "umm-maybe/AI-image-detector",
    "Nahrawy/AIorNot",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "dima806/ai_vs_real_image_detection",
]
DISPLAY = {
    "Organika/sdxl-detector":                    "SDXL-Detector (Swin)",
    "umm-maybe/AI-image-detector":               "AI-image-detector (Swin)",
    "Nahrawy/AIorNot":                           "AIorNot (Swin-T)",
    "prithivMLmods/Deep-Fake-Detector-v2-Model": "DeepFake-Detector-v2 (ViT)",
    "dima806/ai_vs_real_image_detection":        "AIvsReal-ViT (dima806)",
}

ONNX_CANDIDATE_PATHS = [
    "/content/drive/MyDrive/model.onnx/model.onnx",
    "/content/drive/MyDrive/AIGCD/model.onnx",
    "/content/drive/MyDrive/model.onnx",
    "/content/model.onnx",
    "C:/IPED/IPED/target/release/iped-4.4.0-SNAPSHOT/models/convnext-ai-detector/model.onnx",
    "model.onnx",
]

# Faixas de score do plugin (LOW 0.00-0.39, MEDIUM 0.40-0.69, HIGH 0.70-1.00).
BANDS = [("LOW", 0.0, 0.40), ("MEDIUM", 0.40, 0.70), ("HIGH", 0.70, 1.01)]

# Convencao unificada: y = 1 -> IA (fake); y = 0 -> Real. Score = probabilidade de "gerada por IA".

RESULTS_DIR = "/content/aigcd_resultados" if IN_COLAB else "./aigcd_resultados"
FIG_DIR = os.path.join(RESULTS_DIR, "figuras")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(fn):
    plt.savefig(os.path.join(FIG_DIR, fn), dpi=150, bbox_inches="tight")

print("Dispositivo para modelos Transformer:", DEVICE.upper())
print("Inferencia ONNX executada em CPU (modo de operacao do plugin).")
print("Versoes: torch", torch.__version__, "| onnxruntime", ort.__version__)
print("Resultados serao salvos em:", RESULTS_DIR)
''')

# ======================================================================
md(r'''---
## 3. Modelo de referencia do plugin (ConvNeXt V2 em ONNX)

Registra-se o hash SHA-256 do arquivo e a versao do ONNX Runtime (reprodutibilidade pericial — RNF12).''')

code(r'''
if IN_COLAB:
    if os.path.isdir("/content/drive/MyDrive"):
        print("Google Drive ja montado.")
    else:
        from google.colab import drive
        print("Autorize o acesso ao Google Drive na janela que aparecer "
              "(escolha a conta e clique em Permitir).")
        montado = False
        for tentativa in range(1, 3):
            try:
                drive.mount("/content/drive", force_remount=True)
                montado = os.path.isdir("/content/drive/MyDrive")
                if montado:
                    break
            except Exception as e:
                print(f"Tentativa {tentativa} de montar o Drive falhou: {e}")
        if not montado:
            print("Nao foi possivel montar o Drive automaticamente.")
            print("Alternativa: no painel esquerdo do Colab, clique no icone de pasta e no botao")
            print("'Montar Drive'; em seguida, reexecute apenas esta celula.")

import glob
ONNX_PATH = None
for p in ONNX_CANDIDATE_PATHS:
    if os.path.exists(p):
        ONNX_PATH = p
        break

if ONNX_PATH is None and IN_COLAB and os.path.isdir("/content/drive/MyDrive"):
    print("Modelo nao encontrado nos caminhos padrao; buscando no Drive...")
    matches = glob.glob("/content/drive/MyDrive/**/model.onnx", recursive=True)
    if matches:
        matches.sort(key=lambda x: os.path.getsize(x), reverse=True)
        ONNX_PATH = matches[0]
        print("Localizado por busca:", ONNX_PATH)

if ONNX_PATH:
    h = hashlib.sha256()
    with open(ONNX_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print("Modelo ONNX localizado:", ONNX_PATH)
    print(f"  Tamanho     : {size_mb:.1f} MB")
    print("  SHA-256     :", h.hexdigest())
    print("  ONNX Runtime:", ort.__version__)
else:
    print("AVISO: modelo ONNX nao localizado. O experimento prossegue sem o modelo de referencia.")
''')

# ======================================================================
md(r'''---
## 4. Funcoes auxiliares

Define-se, de forma reutilizavel, o conjunto de funcoes aplicadas a todos os conjuntos de dados: coleta
balanceada, inferencia (ONNX e Transformer), resolucao de rotulos, orientacao de scores, metricas e graficos.''')

code(r'''
# ---------- Coleta de dados ----------
def detect_columns(example):
    img_col = None
    for k, v in example.items():
        if hasattr(v, "convert") or "Image" in type(v).__name__:
            img_col = k
            break
    if "binary_label" in example:
        label_col = "binary_label"
    elif "label" in example:
        label_col = "label"
    else:
        label_col = [k for k, v in example.items()
                     if isinstance(v, int) or (isinstance(v, str) and str(v).isdigit())][0]
    return img_col, label_col

def collect_balanced(dataset_id, split, n_per_class, ai_label_value):
    """Coleta, em streaming, um subconjunto balanceado. Retorna (images, y, info)."""
    peek = next(iter(load_dataset(dataset_id, split=split, streaming=True)))
    img_col, label_col = detect_columns(peek)
    ds = load_dataset(dataset_id, split=split, streaming=True).shuffle(seed=SEED, buffer_size=5000)
    buckets = {}
    scanned = 0
    for ex in ds:
        scanned += 1
        try:
            lbl = int(ex[label_col])
        except Exception:
            continue
        buckets.setdefault(lbl, [])
        if len(buckets[lbl]) < n_per_class:
            try:
                buckets[lbl].append(ex[img_col].convert("RGB"))
            except Exception:
                pass
        if len(buckets) >= 2 and all(len(v) >= n_per_class for v in buckets.values()):
            break
        if scanned >= MAX_SCAN:
            break
    images, y = [], []
    for lbl, imgs in buckets.items():
        for im in imgs:
            images.append(im)
            y.append(1 if lbl == ai_label_value else 0)
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(images)); rng.shuffle(idx)
    images = [images[i] for i in idx]
    y = np.array([y[i] for i in idx])
    info = {"img_col": img_col, "label_col": label_col,
            "counts": {k: len(v) for k, v in buckets.items()}}
    return images, y, info

# ---------- Pre-processamento e inferencia ONNX ----------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_onnx(img):
    # Resize (menor lado = 288, bicubico) -> CenterCrop (256) -> Normalize ImageNet -> [1,3,256,256].
    img = img.convert("RGB")
    w, h = img.size
    short = 288
    if w <= h:
        nw, nh = short, int(round(short * h / w))
    else:
        nh, nw = short, int(round(short * w / h))
    img = img.resize((nw, nh), Image.BICUBIC)
    left = (nw - 256) // 2; top = (nh - 256) // 2
    img = img.crop((left, top, left + 256, top + 256))
    a = np.asarray(img).astype(np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    a = np.transpose(a, (2, 0, 1))[None, ...]
    return a.astype(np.float32)

def softmax_np(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

def run_onnx(path, imgs):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    scores = []
    t0 = time.time()
    for im in imgs:
        x = preprocess_onnx(im)
        out = np.asarray(sess.run(None, {iname: x})[0])  # logits [1,2] = [real, fake]
        scores.append(float(softmax_np(out)[0, 1]))      # indice 1 = classe gerada por IA
    ms = (time.time() - t0) / len(imgs) * 1000
    return np.array(scores), ms

# ---------- Inferencia Transformer ----------
FAKE_KW = ["ai", "artificial", "fake", "deepfake", "synthetic", "generated",
           "gan", "diffusion", "sdxl", "midjourney", "dalle"]
REAL_KW = ["real", "human", "authentic", "genuine", "natural", "photo", "nature", "pristine"]

def resolve_fake_index(id2label):
    labels = {int(k): str(v).lower() for k, v in id2label.items()}
    for idx, name in labels.items():
        if any(k in name for k in FAKE_KW):
            return idx
    real_idx = [idx for idx, name in labels.items() if any(k in name for k in REAL_KW)]
    if real_idx and len(labels) == 2:
        return [i for i in labels if i not in real_idx][0]
    return 1 if len(labels) > 1 else 0

def load_processor(mid):
    # Alguns modelos publicam apenas AutoFeatureExtractor (nao AutoImageProcessor); trata-se ambos.
    try:
        return AutoImageProcessor.from_pretrained(mid)
    except Exception:
        from transformers import AutoFeatureExtractor
        return AutoFeatureExtractor.from_pretrained(mid)

def run_hf(mid, imgs):
    proc = load_processor(mid)
    model = AutoModelForImageClassification.from_pretrained(mid).to(DEVICE).eval()
    id2label = dict(model.config.id2label)
    fake_idx = resolve_fake_index(id2label)
    with torch.no_grad():  # aquecimento (exclui inicializacao da GPU da medicao)
        _ = model(**proc(images=imgs[:min(4, len(imgs))], return_tensors="pt").to(DEVICE))
    if DEVICE == "cuda": torch.cuda.synchronize()
    scores = []
    t0 = time.time()
    for i in range(0, len(imgs), BATCH_SIZE):
        batch = imgs[i:i + BATCH_SIZE]
        inp = proc(images=batch, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inp).logits
        p = torch.softmax(logits, dim=-1)[:, fake_idx].detach().float().cpu().numpy()
        scores.extend(p.tolist())
    if DEVICE == "cuda": torch.cuda.synchronize()
    ms = (time.time() - t0) / len(imgs) * 1000
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return np.array(scores), ms, id2label, fake_idx

def orient(y, s):
    # Assegura score alto = maior evidencia de IA. Reorienta se AUC < 0.5.
    a = roc_auc_score(y, s)
    if a < 0.5:
        return 1.0 - s, True
    return s, False

def evaluate_all(imgs, y):
    """Executa o modelo ONNX e todos os modelos Transformer sobre (imgs, y)."""
    scores, times, devs, flips = {}, {}, {}, {}
    if ONNX_PATH:
        try:
            s, ms = run_onnx(ONNX_PATH, imgs)
            s, fl = orient(y, s)
            nm = "ConvNeXt V2 (ONNX/plugin)"
            scores[nm] = s; times[nm] = ms; devs[nm] = "CPU"; flips[nm] = fl
            print(f"  {nm}: AUC={roc_auc_score(y, s):.3f} | {ms:.1f} ms/img")
        except Exception as e:
            print("  falha na execucao do modelo ONNX:", repr(e))
    for mid in HF_MODELS:
        nm = DISPLAY.get(mid, mid)
        try:
            s, ms, id2label, fake_idx = run_hf(mid, imgs)
            s, fl = orient(y, s)
            scores[nm] = s; times[nm] = ms; devs[nm] = DEVICE.upper(); flips[nm] = fl
            print(f"  {nm}: AUC={roc_auc_score(y, s):.3f} | {ms:.1f} ms/img | fake_idx={fake_idx}")
        except Exception as e:
            print(f"  ignorado ({nm}): {repr(e)}")
    return scores, times, devs, flips

print("Funcoes de coleta e inferencia definidas.")
''')

code(r'''
# ---------- Metricas ----------
def eer_from_scores(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return (fpr[i] + fnr[i]) / 2.0, thr[i]

def metrics_at(y, s, thr):
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(Acc=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred, zero_division=0), F1=f1_score(y, pred, zero_division=0),
                Especificidade=spec)

def build_results(scores, times, devs, y):
    rows = []
    for name, s in scores.items():
        eer, _ = eer_from_scores(y, s)
        m05 = metrics_at(y, s, 0.5); m70 = metrics_at(y, s, 0.70)
        rows.append({
            "Modelo": name, "ROC-AUC": roc_auc_score(y, s),
            "PR-AUC": average_precision_score(y, s), "EER": eer,
            "Acc@0.5": m05["Acc"], "Precision@0.5": m05["Precision"],
            "Recall@0.5": m05["Recall"], "F1@0.5": m05["F1"], "Espec@0.5": m05["Especificidade"],
            "F1@0.70": m70["F1"], "Recall@0.70": m70["Recall"],
            "Brier": brier_score_loss(y, s), "ms/img": times[name], "Device": devs[name],
        })
    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    num = [c for c in df.columns if c not in ("Modelo", "Device")]
    df[num] = df[num].astype(float).round(4)
    return df

# ---------- Faixas do plugin ----------
def band_of(score):
    for nm, lo, hi in BANDS:
        if lo <= score < hi:
            return nm
    return "HIGH"

def band_table(s, y):
    df = pd.DataFrame({"classe": ["IA" if t == 1 else "Real" for t in y],
                       "faixa": [band_of(x) for x in s]})
    return pd.crosstab(df["classe"], df["faixa"]).reindex(columns=["LOW", "MEDIUM", "HIGH"], fill_value=0)

def mostra(titulo, imgs, n=5):
    n = min(n, len(imgs))
    if n == 0:
        print("(sem imagens desta classe)"); return
    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1: axs = [axs]
    for ax, im in zip(axs, imgs[:n]):
        ax.imshow(im); ax.axis("off")
    fig.suptitle(titulo, fontsize=13)
    plt.tight_layout(); plt.show()

def show_result(df):
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string(index=False))

# ---------- Graficos ----------
def plot_roc(scores, y, suffix, title_suffix):
    plt.figure(figsize=(7.5, 6.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, lw=2, label=f"{name}  (AUC={roc_auc_score(y, s):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="aleatorio")
    plt.xlabel("Taxa de Falso Positivo (FPR)"); plt.ylabel("Taxa de Verdadeiro Positivo (TPR)")
    plt.title(f"Curvas ROC — {title_suffix}")
    plt.legend(loc="lower right", fontsize=9); plt.tight_layout()
    savefig(f"roc_{suffix}.png"); plt.show()

def plot_pr(scores, y, suffix, title_suffix):
    plt.figure(figsize=(7.5, 6.5))
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y, s)
        plt.plot(rec, prec, lw=2, label=f"{name}  (AP={average_precision_score(y, s):.3f})")
    plt.xlabel("Revocacao"); plt.ylabel("Precisao")
    plt.title(f"Curvas Precisao-Revocacao — {title_suffix}")
    plt.legend(loc="lower left", fontsize=9); plt.tight_layout()
    savefig(f"pr_{suffix}.png"); plt.show()

def plot_bars(df, suffix, title_suffix):
    p = df.set_index("Modelo")[["ROC-AUC", "F1@0.5", "Acc@0.5", "PR-AUC"]]
    ax = p.plot(kind="bar", figsize=(10, 5.5), rot=20, width=0.8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("valor"); ax.set_title(f"Metricas por modelo — {title_suffix}")
    ax.legend(loc="lower right", fontsize=9)
    for c in ax.containers: ax.bar_label(c, fmt="%.2f", fontsize=7, padding=1)
    plt.tight_layout(); savefig(f"barras_{suffix}.png"); plt.show()

def plot_confusion(scores, y, suffix, title_suffix, thr=0.5):
    names = list(scores.keys()); n = len(names)
    cols = min(3, n); rows_ = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows_, cols, figsize=(4.2 * cols, 3.8 * rows_))
    axs = np.array(axs).reshape(-1)
    for ax, name in zip(axs, names):
        cm = confusion_matrix(y, (scores[name] >= thr).astype(int), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Real", "IA"], yticklabels=["Real", "IA"])
        ax.set_title(name, fontsize=9); ax.set_xlabel("previsto"); ax.set_ylabel("real")
    for ax in axs[n:]: ax.axis("off")
    plt.suptitle(f"Matrizes de confusao (limiar {thr}) — {title_suffix}", y=1.02, fontsize=13)
    plt.tight_layout(); savefig(f"confusao_{suffix}.png"); plt.show()

def plot_scoredist(scores, y, suffix, title_suffix):
    names = list(scores.keys()); n = len(names)
    cols = min(3, n); rows_ = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows_, cols, figsize=(4.6 * cols, 3.6 * rows_))
    axs = np.array(axs).reshape(-1)
    for ax, name in zip(axs, names):
        s = scores[name]
        ax.hist(s[y == 0], bins=25, range=(0, 1), alpha=0.6, label="Real", color="#2a9d8f")
        ax.hist(s[y == 1], bins=25, range=(0, 1), alpha=0.6, label="IA", color="#e76f51")
        for x, c in [(0.40, "orange"), (0.70, "red")]:
            ax.axvline(x, ls="--", lw=1, color=c, alpha=0.7)
        ax.set_title(name, fontsize=9); ax.set_xlabel("score (P de ser IA)"); ax.legend(fontsize=8)
    for ax in axs[n:]: ax.axis("off")
    plt.suptitle(f"Distribuicao dos scores — {title_suffix}", y=1.02, fontsize=13)
    plt.tight_layout(); savefig(f"distribuicao_{suffix}.png"); plt.show()

def plot_calibration(scores, y, suffix, title_suffix):
    plt.figure(figsize=(7, 6.5))
    for name, s in scores.items():
        try:
            fp, mp = calibration_curve(y, s, n_bins=10, strategy="uniform")
            plt.plot(mp, fp, "o-", lw=1.8, label=f"{name} (Brier={brier_score_loss(y, s):.3f})")
        except Exception:
            pass
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="calibracao ideal")
    plt.xlabel("Score medio previsto"); plt.ylabel("Fracao efetiva de IA")
    plt.title(f"Curvas de calibracao — {title_suffix}"); plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout(); savefig(f"calibracao_{suffix}.png"); plt.show()

def plot_timing(df, suffix, title_suffix):
    t = df[["Modelo", "ms/img", "Device"]].sort_values("ms/img")
    plt.figure(figsize=(9, 4.8))
    colors = ["#264653" if d == "CPU" else "#e9c46a" for d in t["Device"]]
    plt.barh(t["Modelo"], t["ms/img"], color=colors)
    for i, (v, d) in enumerate(zip(t["ms/img"], t["Device"])):
        plt.text(v, i, f" {v:.1f} ms ({d})", va="center", fontsize=9)
    plt.xlabel("tempo medio por imagem (ms)"); plt.title(f"Custo de inferencia — {title_suffix}")
    plt.tight_layout(); savefig(f"tempo_{suffix}.png"); plt.show()

def plot_bands(scores, y, target, suffix, title_suffix):
    ct = band_table(scores[target], y)
    ax = ct.plot(kind="bar", stacked=True, figsize=(6.5, 4.5),
                 color=["#2a9d8f", "#e9c46a", "#e76f51"])  # ordem: LOW, MEDIUM, HIGH
    ax.set_title(f"Faixas LOW / MEDIUM / HIGH — {target} ({title_suffix})")
    ax.set_ylabel("numero de imagens"); ax.set_xlabel("")
    plt.xticks(rotation=0)
    for c in ax.containers: ax.bar_label(c, fmt="%d", label_type="center", fontsize=9)
    plt.tight_layout(); savefig(f"faixas_{suffix}.png"); plt.show()
    return ct

def target_model(scores, df):
    return "ConvNeXt V2 (ONNX/plugin)" if "ConvNeXt V2 (ONNX/plugin)" in scores else df.iloc[0]["Modelo"]

print("Funcoes de metricas e graficos definidas.")
''')

# ======================================================================
md(r'''---
## 5. Conjunto 1 — `AI-vs-Real` (estilizado, caso facil)

Este conjunto contem imagens de IA predominantemente em estilo artistico/ilustracao. Serve como limite
superior de desempenho: a distincao entre arte digital evidente e fotografia real e comparativamente facil.''')

code(r'''
print(f"Coletando {N_PER_CLASS} imagens por classe de '{DATASET_ID}'...")
images, y_true, info1 = collect_balanced(DATASET_ID, DATASET_SPLIT, N_PER_CLASS, AI_LABEL_VALUE)
print("Colunas:", info1["img_col"], "/", info1["label_col"], "| coletadas:", info1["counts"])
print(f"Total: {len(images)} imagens (IA={int(y_true.sum())}, Real={int((1-y_true).sum())})")
''')

md(r'''### 5.1 Verificacao dos rotulos

A primeira fileira deve conter fotografias reais e a segunda, imagens geradas por IA. Em caso de inversao,
ajustar `AI_LABEL_VALUE` e reexecutar.''')

code(r'''
mostra("AI-vs-Real — Classe Real (y = 0)", [images[i] for i in range(len(images)) if y_true[i] == 0])
mostra("AI-vs-Real — Classe IA (y = 1)", [images[i] for i in range(len(images)) if y_true[i] == 1])

plt.figure(figsize=(4, 3.2))
vals = [int((y_true == 0).sum()), int((y_true == 1).sum())]
plt.bar(["Real", "IA"], vals, color=["#2a9d8f", "#e76f51"])
for i, v in enumerate(vals): plt.text(i, v + 1, str(v), ha="center")
plt.title("Balanceamento — AI-vs-Real"); plt.ylabel("numero de imagens")
plt.tight_layout(); savefig("balanceamento_aivsreal.png"); plt.show()
''')

# ======================================================================
md(r'''---
## 6. Avaliacao no Conjunto 1''')

code(r'''
print("Avaliacao em AI-vs-Real:")
scores_by_model, times_by_model, device_by_model, flip_log = evaluate_all(images, y_true)
assert scores_by_model, "Nenhum modelo foi executado."
if all(flip_log.values()):
    print("\n[ATENCAO] Todos os modelos foram reorientados; revisar AI_LABEL_VALUE (secao 5.1).")
''')

# ======================================================================
md(r'''---
## 7. Metricas (Conjunto 1)''')

code(r'''
results_df = build_results(scores_by_model, times_by_model, device_by_model, y_true)
results_df.to_csv(os.path.join(RESULTS_DIR, "metricas_aivsreal.csv"), index=False)
show_result(results_df)
''')

# ======================================================================
md(r'''---
## 8. Graficos (Conjunto 1)''')

code(r'''
plot_roc(scores_by_model, y_true, "aivsreal", DATASET_TITLE)
plot_pr(scores_by_model, y_true, "aivsreal", DATASET_TITLE)
''')

code(r'''
plot_bars(results_df, "aivsreal", DATASET_TITLE)
plot_confusion(scores_by_model, y_true, "aivsreal", DATASET_TITLE)
''')

code(r'''
plot_scoredist(scores_by_model, y_true, "aivsreal", DATASET_TITLE)
plot_calibration(scores_by_model, y_true, "aivsreal", DATASET_TITLE)
''')

code(r'''
plot_timing(results_df, "aivsreal", DATASET_TITLE)
''')

# ======================================================================
md(r'''---
## 9. Analise de limiar (Conjunto 1)

Variacao de precisao, revocacao e F1 em funcao do limiar, para o modelo de maior ROC-AUC.''')

code(r'''
best_name = results_df.iloc[0]["Modelo"]
s_best = scores_by_model[best_name]
ths = np.linspace(0.01, 0.99, 99)
P = [metrics_at(y_true, s_best, t)["Precision"] for t in ths]
R = [metrics_at(y_true, s_best, t)["Recall"] for t in ths]
F = [metrics_at(y_true, s_best, t)["F1"] for t in ths]
best_t = float(ths[int(np.argmax(F))])

plt.figure(figsize=(8.5, 5))
plt.plot(ths, P, label="Precisao"); plt.plot(ths, R, label="Revocacao"); plt.plot(ths, F, label="F1", lw=2.5)
plt.axvline(best_t, ls="--", color="gray", label=f"F1 maximo em thr={best_t:.2f}")
plt.axvline(0.70, ls=":", color="red", alpha=0.7, label="limiar HIGH do plugin (0,70)")
plt.xlabel("limiar"); plt.ylabel("valor")
plt.title(f"Precisao / Revocacao / F1 em funcao do limiar — {best_name}")
plt.legend(fontsize=9); plt.tight_layout(); savefig("analise_limiar_aivsreal.png"); plt.show()
print(f"F1 maximo do modelo '{best_name}' no limiar {best_t:.2f}; o plugin adota 0,70 para HIGH.")
''')

# ======================================================================
md(r'''---
## 10. Faixas de classificacao do plugin (Conjunto 1)

A frequencia de imagens reais classificadas em HIGH corresponde a taxa de falsos positivos no ponto de
operacao do plugin.''')

code(r'''
for name, s in scores_by_model.items():
    print("-" * 60); print(name); show_result(band_table(s, y_true))
tgt1 = target_model(scores_by_model, results_df)
ct1 = plot_bands(scores_by_model, y_true, tgt1, "aivsreal", DATASET_TITLE)
''')

# ======================================================================
md(r'''---
## 11. Conjunto 2 — `GenImage` (fotorrealista, caso dificil)

Este e o conjunto de **maior relevancia pericial**. As imagens sinteticas sao **fotorrealistas**, geradas
por multiplos modelos modernos (Stable Diffusion, Midjourney, ADM, GLIDE, VQDM, Wukong, BigGAN), e as reais
sao fotografias do ImageNet. Diferentemente do Conjunto 1, aqui a imagem gerada por IA e dificil de
distinguir de uma foto autentica — situacao que caracteriza o desafio real enfrentado pelo perito. Espera-se
desempenho inferior ao do Conjunto 1; a magnitude dessa queda e um resultado central do estudo.''')

code(r'''
print(f"Coletando {N_PER_CLASS} imagens por classe de '{GENIMAGE_ID}' ({GENIMAGE_SPLIT})...")
img_gi, y_gi, info_gi = collect_balanced(GENIMAGE_ID, GENIMAGE_SPLIT, N_PER_CLASS, GENIMAGE_AI_LABEL)
print("Colunas:", info_gi["img_col"], "/", info_gi["label_col"], "| coletadas:", info_gi["counts"])
print(f"Total: {len(img_gi)} imagens (IA={int(y_gi.sum())}, Real={int((1-y_gi).sum())})")

mostra("GenImage — Classe Real (y = 0)", [img_gi[i] for i in range(len(img_gi)) if y_gi[i] == 0])
mostra("GenImage — Classe IA / fotorrealista (y = 1)", [img_gi[i] for i in range(len(img_gi)) if y_gi[i] == 1])
''')

code(r'''
print("Avaliacao em GenImage:")
scores_gi, times_gi, dev_gi, flip_gi = evaluate_all(img_gi, y_gi)
assert scores_gi, "Nenhum modelo foi executado no GenImage."
results_gi = build_results(scores_gi, times_gi, dev_gi, y_gi)
results_gi.to_csv(os.path.join(RESULTS_DIR, "metricas_genimage.csv"), index=False)
show_result(results_gi)
''')

code(r'''
plot_roc(scores_gi, y_gi, "genimage", GENIMAGE_TITLE)
plot_scoredist(scores_gi, y_gi, "genimage", GENIMAGE_TITLE)
tgt_gi = target_model(scores_gi, results_gi)
ct_gi = plot_bands(scores_gi, y_gi, tgt_gi, "genimage", GENIMAGE_TITLE)
''')

# ======================================================================
md(r'''---
## 12. Conjunto 3 — `CIFAKE` (baixa resolucao, robustez)

CIFAKE combina fotografias do CIFAR-10 e imagens do Stable Diffusion 1.4, em resolucao 32x32. Constitui um
teste de robustez sob mudanca de resolucao e de dominio (as imagens sao ampliadas para a entrada dos modelos).''')

code(r'''
print(f"Coletando {N_PER_CLASS} imagens por classe de '{CIFAKE_ID}' ({CIFAKE_SPLIT})...")
img_cf, y_cf, info_cf = collect_balanced(CIFAKE_ID, CIFAKE_SPLIT, N_PER_CLASS, CIFAKE_AI_LABEL)
print("Colunas:", info_cf["img_col"], "/", info_cf["label_col"], "| coletadas:", info_cf["counts"])
print(f"Total: {len(img_cf)} imagens (IA={int(y_cf.sum())}, Real={int((1-y_cf).sum())})")

mostra("CIFAKE — Classe Real (y = 0)", [img_cf[i] for i in range(len(img_cf)) if y_cf[i] == 0])
mostra("CIFAKE — Classe IA (y = 1)", [img_cf[i] for i in range(len(img_cf)) if y_cf[i] == 1])
''')

code(r'''
print("Avaliacao em CIFAKE:")
scores_cf, times_cf, dev_cf, flip_cf = evaluate_all(img_cf, y_cf)
assert scores_cf, "Nenhum modelo foi executado no CIFAKE."
results_cf = build_results(scores_cf, times_cf, dev_cf, y_cf)
results_cf.to_csv(os.path.join(RESULTS_DIR, "metricas_cifake.csv"), index=False)
show_result(results_cf)
''')

code(r'''
plot_roc(scores_cf, y_cf, "cifake", CIFAKE_TITLE)
plot_scoredist(scores_cf, y_cf, "cifake", CIFAKE_TITLE)
tgt_cf = target_model(scores_cf, results_cf)
ct_cf = plot_bands(scores_cf, y_cf, tgt_cf, "cifake", CIFAKE_TITLE)
''')

# ======================================================================
md(r'''---
## 13. Comparacao entre conjuntos (gradacao de dificuldade)

Compara-se o ROC-AUC de cada modelo nos tres conjuntos. A leitura horizontal evidencia a **queda de
desempenho a medida que as imagens sinteticas se tornam mais realistas** — do conjunto estilizado (facil)
ao fotorrealista (dificil). Esse comportamento e o resultado central para o uso forense.''')

code(r'''
datasets_eval = [
    (DATASET_TITLE,  y_true, scores_by_model),
    (GENIMAGE_TITLE, y_gi,   scores_gi),
    (CIFAKE_TITLE,   y_cf,   scores_cf),
]

rows = []
for nm in scores_by_model:
    row = {"Modelo": nm}
    for title, yy, sc in datasets_eval:
        row[title] = round(roc_auc_score(yy, sc[nm]), 4) if nm in sc else np.nan
    rows.append(row)
comp_df = pd.DataFrame(rows)
comp_df.to_csv(os.path.join(RESULTS_DIR, "comparacao_datasets.csv"), index=False)
show_result(comp_df)

d = comp_df.set_index("Modelo")[[t for t, _, _ in datasets_eval]]
ax = d.plot(kind="bar", figsize=(11, 5.8), rot=15, width=0.8,
            color=["#2a9d8f", "#e76f51", "#e9c46a"])
ax.set_ylim(0, 1.02); ax.set_ylabel("ROC-AUC")
ax.set_title("ROC-AUC por modelo nos tres conjuntos (facil -> fotorrealista -> baixa resolucao)")
ax.axhline(0.5, ls="--", color="gray", alpha=0.6)
ax.legend(loc="lower left", fontsize=9)
for c in ax.containers: ax.bar_label(c, fmt="%.2f", fontsize=7, padding=1)
plt.tight_layout(); savefig("comparacao_datasets.png"); plt.show()
''')

# ======================================================================
md(r'''---
## 14. Exportacao dos resultados''')

code(r'''
import shutil

resumo = []
resumo.append("RESUMO DO EXPERIMENTO — AIGCD (Parte 1)")
resumo.append("=" * 64)
for title, yy, sc in datasets_eval:
    resumo.append(f"{title}: {len(yy)} imagens (IA={int(yy.sum())}, Real={int((1-yy).sum())})")
resumo.append(f"Dispositivo Transformer: {DEVICE.upper()} | ONNX: CPU")
if ONNX_PATH:
    resumo.append(f"Modelo de referencia (ONNX): {ONNX_PATH}")
resumo.append("")
resumo.append("ROC-AUC por modelo nos tres conjuntos:")
for _, r in comp_df.iterrows():
    vals = " | ".join(f"{r[t]:.3f}" if pd.notna(r[t]) else "  -  " for t, _, _ in datasets_eval)
    resumo.append(f"  - {r['Modelo']:<32} {vals}")
resumo_txt = "\n".join(resumo)
print(resumo_txt)

with open(os.path.join(RESULTS_DIR, "resumo.txt"), "w", encoding="utf-8") as f:
    f.write(resumo_txt)

zip_base = os.path.join(RESULTS_DIR, "aigcd_resultados")
shutil.make_archive(zip_base, "zip", RESULTS_DIR)
print("\nResultados salvos. Arquivo compactado:", zip_base + ".zip")

if IN_COLAB:
    try:
        from google.colab import files
        files.download(zip_base + ".zip")
    except Exception as e:
        print("(Efetuar o download manualmente pela aba de arquivos.)", e)
''')

md(r'''
---
### Encerramento da Parte 1

Foram gerados: as tabelas de metricas dos tres conjuntos, a comparacao entre eles
(`comparacao_datasets.csv`), as figuras e um resumo, destinados a composicao do texto do TCC e da dissertacao.

**Consideracoes metodologicas.** A comparacao apoia-se preferencialmente em metricas independentes de
limiar (ROC-AUC, EER). A avaliacao sobre tres conjuntos com dificuldade crescente — do estilizado ao
fotorrealista — mitiga o risco de conclusoes decorrentes de um cenario artificialmente facil e evidencia
o desempenho no caso de maior relevancia pericial (imagens fotorrealistas). Para afirmacoes de
generalizacao mais amplas recomenda-se a inclusao de conjuntos adicionais (por exemplo, FaceForensics++),
conforme previsto no plano da dissertacao.

**Parte 2 (documentada separadamente).** Treinamento de modelos sobre os mesmos conjuntos (MesoNet,
EfficientNetV2 e uma CNN de base), com analise de arquiteturas, otimizadores e tecnicas de regularizacao.
''')

# ======================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "toc_visible": True},
    "accelerator": "GPU",
}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "AIGCD_Benchmark_Parte1_Detectores.ipynb")
nbf.write(nb, OUT)
print("Notebook gerado:", OUT)
print("Total de celulas:", len(cells))
