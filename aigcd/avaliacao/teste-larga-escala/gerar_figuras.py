"""
Gera as figuras do documento de resultados do teste em larga escala do AIGCD.

Entrada: resultados_por_imagem.csv produzido por calcular_metricas.py e,
         opcionalmente, o log do processamento do IPED (tempo por tarefa).
Saída:   figuras *.png (300 dpi), reaproveitáveis em documentos e slides.

Paleta validada para daltonismo:
  classe   -> Real #2a78d6 (azul) | IA #eb6834 (laranja)
  variante -> Original #1baf7a (verde-água) | JPEG q=85 #6250d6 (violeta)
O verde-água tem contraste < 3:1, por isso as séries levam rótulo direto.

Uso:
    py gerar_figuras.py --csv resultados\\resultados_export\\resultados_por_imagem.csv --saida figuras --log <IPED-....log>
"""

import argparse
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calcular_metricas import curva_roc, faixa, roc_auc  # noqa: E402

SAIDA = "figuras"

COR_REAL, COR_IA = "#2a78d6", "#eb6834"
COR_ORIG, COR_JPEG = "#1baf7a", "#6250d6"
COR_CONTEXTO = "#8a8984"
TINTA, TINTA_2, GRADE = "#0b0b0b", "#52514e", "#e6e4df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def carregar(caminho_csv):
    with open(caminho_csv, encoding="utf-8-sig") as f:
        linhas = list(csv.reader(f, delimiter=";"))[1:]
    return [dict(var=l[0], rot=l[1], grp=l[2], s=float(l[3])) for l in linhas]


def ler_tempos(caminho_log, maximo=3):
    """Lê a tabela "Processing Times per Task" do log do IPED (tarefas com tempo > 0)."""
    tempos, na_tabela = [], False
    with open(caminho_log, encoding="utf-8", errors="replace") as f:
        for linha in f:
            if "Processing Times per Task" in linha:
                na_tabela = True
            elif "Processing Times per Parser" in linha:
                break
            elif na_tabela:
                m = re.search(r"(\S+)\s+(\d+)\s+\d+\s*$", linha)
                if m and int(m.group(2)) > 0:
                    tempos.append((m.group(1).replace(".py", ""), int(m.group(2))))
    tempos.sort(key=lambda t: -t[1])
    if len(tempos) > maximo:
        demais = tempos[maximo:]
        tempos = tempos[:maximo] + [("Demais tarefas (%d)" % len(demais), sum(t for _, t in demais))]
    return tempos


def pct(x):
    return ("%.1f%%" % (100 * x)).replace(".", ",")


def salvar(fig, nome):
    fig.savefig(os.path.join(SAIDA, nome), facecolor="white")
    plt.close(fig)
    print("ok", nome)


def zonas(ax, rotular):
    ax.axvspan(0.40, 0.70, color="#f2f1ee", zorder=0)
    for x in (0.40, 0.70):
        ax.axvline(x, color=TINTA_2, lw=1, ls="--", zorder=1)
    if rotular:
        for x, txt in ((0.20, "LOW"), (0.55, "MEDIUM"), (0.85, "HIGH")):
            ax.text(x, 1.02, txt, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9, color=TINTA_2, fontweight="bold")


def fig_histogramas(D):
    gen = [d for d in D if d["grp"].startswith("GenImage")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True, sharey=True)
    colunas = (("original", "Original (como saiu do gerador)"),
               ("jpeg85", "Recomprimido em JPEG (qualidade 85)"))
    linhas = (("real", "Fotos reais", COR_REAL), ("ia", "Imagens de IA", COR_IA))
    bins = [i / 20 for i in range(21)]
    for c, (var, titulo) in enumerate(colunas):
        for r, (rot, nome, cor) in enumerate(linhas):
            ax = axes[r][c]
            sc = [d["s"] for d in gen if d["var"] == var and d["rot"] == rot]
            zonas(ax, rotular=(r == 0))
            ax.hist(sc, bins=bins, color=cor, rwidth=0.86, zorder=2)
            alto = sum(1 for s in sc if faixa(s) == "HIGH") / len(sc)
            rotulo = ("HIGH: %s" if rot == "ia" else "HIGH (falso positivo): %s") % pct(alto)
            ax.text(0.97, 0.90, rotulo, transform=ax.transAxes, ha="right", fontsize=10,
                    color=TINTA, bbox=dict(fc="white", ec=GRADE, boxstyle="round,pad=0.3"))
            if r == 0:
                ax.set_title(titulo, pad=18)
            if c == 0:
                ax.set_ylabel("%s (n=%d)\nquantidade" % (nome, len(sc)))
            if r == 1:
                ax.set_xlabel("score aigcd:score:image")
    fig.suptitle("GenImage: distribuição dos scores por classe", y=1.02, fontsize=13, fontweight="bold")
    salvar(fig, "fig1_histogramas_genimage.png")


def fig_roc(D):
    series = (
        ("GenImage original", [d for d in D if d["var"] == "original" and d["grp"].startswith("GenImage")], COR_ORIG, "-"),
        ("GenImage em JPEG q=85", [d for d in D if d["var"] == "jpeg85" and d["grp"].startswith("GenImage")], COR_JPEG, "-"),
        ("CIFAKE 32x32", [d for d in D if d["var"] == "original" and d["grp"] == "CIFAKE"], COR_CONTEXTO, "--"),
    )
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot([0, 1], [0, 1], ls=":", color=COR_CONTEXTO, lw=1.2)
    ax.text(0.62, 0.55, "acaso (AUC 0,5)", rotation=43, color=TINTA_2, fontsize=9)
    for nome, dados, cor, estilo in series:
        pos = [d["s"] for d in dados if d["rot"] == "ia"]
        neg = [d["s"] for d in dados if d["rot"] == "real"]
        pts = curva_roc(pos, neg)
        auc = roc_auc(pos, neg)
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color=cor, lw=2, ls=estilo,
                label="%s (AUC %s)" % (nome, ("%.3f" % auc).replace(".", ",")))
        tpr = sum(1 for s in pos if s >= 0.70) / len(pos)
        fpr = sum(1 for s in neg if s >= 0.70) / len(neg)
        ax.plot([fpr], [tpr], "o", ms=8, color=cor, mec="white", mew=2, zorder=5)
    ax.plot([], [], "o", ms=8, color=TINTA_2, mec="white", mew=2, ls="none",
            label="ponto de operação do painel (limiar 0,70)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("Fotos reais acusadas (taxa de falsos positivos)")
    ax.set_ylabel("Imagens de IA detectadas (taxa de verdadeiros positivos)")
    ax.set_title("Curva ROC por conjunto")
    ax.legend(loc="lower right", fontsize=9.5)
    salvar(fig, "fig2_roc_comparativa.png")


def fig_geradores(D):
    nomes = ["ADM", "BigGAN", "VQDM", "SD15", "GLIDE", "Wukong", "Midjourney"]
    taxa = {}
    for var in ("original", "jpeg85"):
        for g in nomes:
            sc = [d["s"] for d in D if d["var"] == var and d["grp"] == "GenImage_" + g]
            taxa[(var, g)] = sum(1 for s in sc if faixa(s) == "HIGH") / len(sc)
    nomes.sort(key=lambda g: taxa[("jpeg85", g)])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    h = 0.38
    for i, g in enumerate(nomes):
        for desl, var, cor in ((-h / 2, "original", COR_ORIG), (h / 2, "jpeg85", COR_JPEG)):
            v = taxa[(var, g)]
            ax.barh(i + desl, 100 * v, height=h - 0.04, color=cor, zorder=2,
                    label=None if i else ("Original" if var == "original" else "Recomprimido em JPEG q=85"))
            ax.text(100 * v + 1, i + desl, "%d%%" % round(100 * v), va="center", fontsize=9, color=TINTA_2)
    ax.set_yticks(range(len(nomes)), nomes)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Imagens de IA classificadas como HIGH (%)")
    ax.set_title("Detecção por gerador: efeito da recompressão JPEG")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9.5)
    salvar(fig, "fig3_deteccao_por_gerador.png")


def fig_proprias(D):
    # Grupos = subpastas de proprias/real e proprias/ia; reais primeiro
    pares = sorted({(d["rot"], d["grp"]) for d in D if d["var"] == "proprias"}, key=lambda p: (p[0] != "real", p[1]))
    grupos = [(g, rot, "%s\n(%s)" % (g, "real" if rot == "real" else "IA")) for rot, g in pares]
    fig, ax = plt.subplots(figsize=(9.5, 1.2 + 0.85 * len(grupos)))
    zonas(ax, rotular=True)
    for i, (g, rot, rotulo) in enumerate(grupos):
        sc = sorted(d["s"] for d in D if d["var"] == "proprias" and d["grp"] == g)
        offs = [((k % 5) - 2) * 0.07 for k in range(len(sc))]  # espalha pontos sobrepostos
        ax.scatter(sc, [i + o for o in offs], s=70, color=COR_REAL if rot == "real" else COR_IA,
                   edgecolor="white", linewidth=1.5, zorder=3)
        altos = sum(1 for s in sc if faixa(s) == "HIGH")
        ax.text(1.02, i, "%d de %d em HIGH" % (altos, len(sc)), va="center", fontsize=9.5,
                color=TINTA, transform=ax.get_yaxis_transform())
    ax.scatter([], [], s=70, color=COR_REAL, label="Real")
    ax.scatter([], [], s=70, color=COR_IA, label="IA")
    ax.set_yticks(range(len(grupos)), [g[2] for g in grupos])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("score aigcd:score:image")
    ax.set_title("Imagens próprias: geradores atuais e fotos de smartphone", pad=18)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9.5)
    salvar(fig, "fig4_imagens_proprias.png")


def fig_tempo(tempos, n_imagens):
    total = sum(t for _, t in tempos)
    maior = max(t for _, t in tempos)
    fig, ax = plt.subplots(figsize=(8.5, 0.6 + 0.65 * len(tempos)))
    for i, (nome, t) in enumerate(tempos):
        ax.barh(i, t, height=0.6, color="#3d3c39" if nome == "AIGCDTask" else "#b5b3ad", zorder=2)
        ax.text(t + maior * 0.01, i, "%d s (%d%%)" % (t, round(100 * t / total)), va="center",
                fontsize=9.5, color=TINTA)
    ax.set_yticks(range(len(tempos)), [n for n, _ in tempos])
    ax.invert_yaxis()
    ax.set_xlim(0, maior * 1.25)
    ax.set_xlabel("Tempo acumulado nas threads de processamento (s)")
    ax.set_title("Custo por tarefa no caso de %s imagens" % format(n_imagens, ",").replace(",", "."))
    ax.grid(axis="y", visible=False)
    salvar(fig, "fig5_tempo_por_tarefa.png")


def main():
    global SAIDA
    ap = argparse.ArgumentParser(description="Gera as figuras do teste em larga escala do AIGCD.")
    ap.add_argument("--csv", required=True, help="resultados_por_imagem.csv gerado por calcular_metricas.py")
    ap.add_argument("--saida", default="figuras", help="Pasta de saída. Padrão: figuras")
    ap.add_argument("--log", help="Log do processamento do IPED, para a figura de tempo por tarefa")
    args = ap.parse_args()

    SAIDA = args.saida
    os.makedirs(SAIDA, exist_ok=True)
    D = carregar(args.csv)
    variantes = {d["var"] for d in D}
    if {"original", "jpeg85"} <= variantes:
        fig_histogramas(D)
        fig_roc(D)
        fig_geradores(D)
    if "proprias" in variantes:
        fig_proprias(D)
    if args.log:
        fig_tempo(ler_tempos(args.log), len(D))


if __name__ == "__main__":
    main()
