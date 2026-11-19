"""
Calcula as metricas do plugin AIGCD a partir do CSV exportado pelo IPED.

Entrada: CSV gerado em "Exportar propriedades dos itens destacados" na interface
do IPED, contendo ao menos as colunas "path" e "aigcd:score:image".

O rotulo verdadeiro vem do caminho: um diretorio chamado "ia" ou "real" (sem
diferenciar maiusculas). O diretorio imediatamente ANTES dele e a "variante"
(ex.: original, jpeg85, proprias) e o diretorio imediatamente DEPOIS e o "grupo"
(ex.: GenImage_Midjourney, celular). Imagens fora de "ia"/"real" entram como
"sem rotulo" e servem para estimar quantos alertas o plugin gera em dados comuns.

As faixas reproduzem exatamente o painel e os filtros do plugin:
    HIGH   -> score >= 0.70
    MEDIUM -> 0.40 <= score < 0.70
    LOW    -> score < 0.40

Usa apenas a biblioteca padrao. Se o matplotlib estiver instalado, gera tambem
histogramas e curvas ROC.

Exemplo:
    py calcular_metricas.py export.csv --log "C:\\...\\log\\IPED-2026-11-06-10-00-00.log"
"""

import argparse
import csv
import io
import math
import os
import random
import re
import sys

LIMIARES = (0.50, 0.70)
LIMIAR_HIGH, LIMIAR_MEDIUM = 0.70, 0.40  # mesmos limites do AIGCDPanel e do AIFiltersConfig.json
EXT_IMAGEM = {"jpg", "jpeg", "png", "gif", "bmp", "webp", "tif", "tiff", "heic", "heif", "jfif"}
COL_SCORE = "AIGCD:SCORE:IMAGE"
COL_PATH = "PATH"


# ----------------------------------------------------------------------------
# Leitura do CSV do IPED
# ----------------------------------------------------------------------------

def ler_export(caminho):
    """Le o CSV do IPED (UTF-8 com BOM, cabecalhos em maiusculas).

    O separador depende do idioma da interface do IPED (';' em portugues,
    ',' em ingles), entao e detectado pelo cabecalho: "CAMPO";"CAMPO" ou "CAMPO","CAMPO".
    """
    with open(caminho, encoding="utf-8-sig", newline="") as f:
        texto = f.read()
    separador = ";" if '";"' in texto.split("\n", 1)[0] else ","
    linhas = list(csv.reader(io.StringIO(texto), delimiter=separador))
    if not linhas:
        sys.exit("CSV vazio: %s" % caminho)

    cab = [c.strip().upper() for c in linhas[0]]
    if COL_SCORE not in cab or COL_PATH not in cab:
        sys.exit("O CSV precisa das colunas 'path' e 'aigcd:score:image'. Encontradas: %s"
                 % [c for c in cab if c])
    i_score, i_path = cab.index(COL_SCORE), cab.index(COL_PATH)

    itens = []
    for lin in linhas[1:]:
        if len(lin) <= max(i_score, i_path):
            continue
        itens.append((lin[i_path].strip(), converter_score(lin[i_score])))
    return itens


def converter_score(texto):
    texto = texto.split("|")[0].strip().replace(",", ".")
    if not texto:
        return None
    try:
        return float(texto)
    except ValueError:
        return None


def classificar_caminho(caminho):
    """Retorna (variante, rotulo, grupo) ou (None, None, None) se nao rotulado."""
    partes = [p for p in re.split(r"[\\/]+", caminho) if p]
    for i, p in enumerate(partes[:-1]):
        if p.lower() in ("ia", "real"):
            variante = partes[i - 1] if i > 0 else "(raiz)"
            grupo = partes[i + 1] if i + 1 < len(partes) - 1 else "(raiz)"
            return variante, p.lower(), grupo
    return None, None, None


def extensao(caminho):
    return os.path.splitext(caminho)[1].lstrip(".").lower()


def faixa(score):
    # Mesma regra do AIGCDPanel e dos filtros do AIFiltersConfig.json (score bruto)
    if score >= LIMIAR_HIGH:
        return "HIGH"
    if score >= LIMIAR_MEDIUM:
        return "MEDIUM"
    return "LOW"


# ----------------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------------

def roc_auc(pos, neg):
    """AUC pela estatistica de Mann-Whitney (com tratamento de empates)."""
    if not pos or not neg:
        return float("nan")
    todos = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    soma_pos, i = 0.0, 0
    while i < len(todos):
        j = i
        while j + 1 < len(todos) and todos[j + 1][0] == todos[i][0]:
            j += 1
        rank_medio = (i + j) / 2.0 + 1
        soma_pos += rank_medio * sum(1 for k in range(i, j + 1) if todos[k][1] == 1)
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return (soma_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_ic95(pos, neg, reamostras=500, semente=42):
    rnd = random.Random(semente)
    valores = []
    for _ in range(reamostras):
        p = [rnd.choice(pos) for _ in pos]
        n = [rnd.choice(neg) for _ in neg]
        valores.append(roc_auc(p, n))
    valores.sort()
    return valores[int(0.025 * reamostras)], valores[int(0.975 * reamostras) - 1]


def curva_roc(pos, neg):
    """Pontos (limiar, FPR, TPR) varrendo todos os scores observados."""
    limiares = sorted(set(pos) | set(neg), reverse=True)
    pontos = [(1.01, 0.0, 0.0)]
    for t in limiares:
        tpr = sum(1 for s in pos if s >= t) / len(pos)
        fpr = sum(1 for s in neg if s >= t) / len(neg)
        pontos.append((t, fpr, tpr))
    return pontos


def eer_e_youden(pos, neg):
    pontos = curva_roc(pos, neg)
    eer_pt = min(pontos, key=lambda p: abs(p[1] - (1 - p[2])))
    youden_pt = max(pontos, key=lambda p: p[2] - p[1])
    eer = (eer_pt[1] + (1 - eer_pt[2])) / 2
    return eer, eer_pt[0], youden_pt


def wilson(acertos, total, z=1.96):
    """Intervalo de confianca de Wilson (95%) para uma proporcao."""
    if total == 0:
        return float("nan"), float("nan")
    p = acertos / total
    den = 1 + z * z / total
    centro = (p + z * z / (2 * total)) / den
    margem = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0.0, centro - margem), min(1.0, centro + margem)


def matriz(pos, neg, t):
    tp = sum(1 for s in pos if s >= t)
    fn = len(pos) - tp
    fp = sum(1 for s in neg if s >= t)
    tn = len(neg) - fp
    return tp, fp, tn, fn


def pct(x):
    return "  n/a " if x != x else "%5.1f%%" % (100 * x)


def div(a, b):
    return a / b if b else float("nan")


# ----------------------------------------------------------------------------
# Log de processamento do IPED
# ----------------------------------------------------------------------------

def ler_log(caminho):
    """Extrai o tempo do AIGCDTask e o tempo total da tabela final do log do IPED."""
    info = {}
    with open(caminho, encoding="utf-8", errors="replace") as f:
        for linha in f:
            m = re.search(r"AIGCDTask\s+(\d+)\s+(\d+)\s*$", linha)
            if m:
                info["aigcd_s"], info["aigcd_pct"] = int(m.group(1)), int(m.group(2))
            m = re.search(r"Total processed: (\d+) items in (\d+) seconds", linha)
            if m:
                info["itens"], info["total_s"] = int(m.group(1)), int(m.group(2))
    return info


# ----------------------------------------------------------------------------
# Relatorio
# ----------------------------------------------------------------------------

def relatorio_variante(nome, dados, out):
    pos = [d["score"] for d in dados if d["rotulo"] == "ia"]
    neg = [d["score"] for d in dados if d["rotulo"] == "real"]

    out.append("")
    out.append("=" * 78)
    out.append("VARIANTE: %s   (IA = %d, Real = %d)" % (nome, len(pos), len(neg)))
    out.append("=" * 78)

    resumo = {"variante": nome, "n_ia": len(pos), "n_real": len(neg)}
    if pos and neg:
        auc = roc_auc(pos, neg)
        lo, hi = auc_ic95(pos, neg)
        eer, eer_t, (y_t, y_fpr, y_tpr) = eer_e_youden(pos, neg)
        out.append("ROC-AUC: %.4f  [IC95%% bootstrap: %.4f - %.4f]" % (auc, lo, hi))
        out.append("EER:     %s (no limiar ~%.2f)" % (pct(eer), eer_t))
        out.append("Limiar otimo (Youden): %.2f  -> TPR %s | FPR %s" % (y_t, pct(y_tpr), pct(y_fpr)))
        resumo.update(auc=auc, eer=eer)

        for t in LIMIARES:
            tp, fp, tn, fn = matriz(pos, neg, t)
            rec, fpr = div(tp, tp + fn), div(fp, fp + tn)
            prec, acc = div(tp, tp + fp), div(tp + tn, tp + tn + fp + fn)
            f1 = div(2 * prec * rec, prec + rec)
            r_lo, r_hi = wilson(tp, tp + fn)
            f_lo, f_hi = wilson(fp, fp + tn)
            out.append("")
            out.append("Limiar %.2f%s" % (t, "  (= faixa HIGH do painel)" if t == 0.70 else ""))
            out.append("  Matriz:            pred IA   pred Real")
            out.append("    IA  (verdade)    %7d   %9d" % (tp, fn))
            out.append("    Real(verdade)    %7d   %9d" % (fp, tn))
            out.append("  Acuracia %s | Precisao %s | F1 %s" % (pct(acc), pct(prec), pct(f1)))
            out.append("  Recall/TPR %s [IC95%% %s-%s] | FPR %s [IC95%% %s-%s]"
                       % (pct(rec), pct(r_lo).strip(), pct(r_hi).strip(),
                          pct(fpr), pct(f_lo).strip(), pct(f_hi).strip()))
            resumo["recall_%.2f" % t], resumo["fpr_%.2f" % t] = rec, fpr

    # Faixas do painel (LOW/MEDIUM/HIGH)
    out.append("")
    out.append("Faixas do painel        LOW     MEDIUM      HIGH")
    for rot, nome_rot in (("ia", "IA"), ("real", "Real")):
        fx = [faixa(d["score"]) for d in dados if d["rotulo"] == rot]
        if fx:
            out.append("  %-6s (n=%5d)   %s    %s    %s" % (
                nome_rot, len(fx), pct(fx.count("LOW") / len(fx)),
                pct(fx.count("MEDIUM") / len(fx)), pct(fx.count("HIGH") / len(fx))))
    decididas = [d for d in dados if faixa(d["score"]) != "MEDIUM"]
    if decididas:
        certas = sum(1 for d in decididas
                     if (faixa(d["score"]) == "HIGH") == (d["rotulo"] == "ia"))
        out.append("  Cobertura (fora de MEDIUM): %s | Acuracia nas decididas: %s"
                   % (pct(len(decididas) / len(dados)), pct(certas / len(decididas))))

    # Por grupo (gerador / origem)
    out.append("")
    out.append("Por grupo                         rotulo     n   media    LOW    MEDIUM   HIGH")
    grupos = sorted({(d["rotulo"], d["grupo"]) for d in dados})
    for rot, grp in grupos:
        sc = [d["score"] for d in dados if d["rotulo"] == rot and d["grupo"] == grp]
        fx = [faixa(s) for s in sc]
        out.append("  %-32s %-5s %5d   %.3f  %s  %s  %s" % (
            grp[:32], rot, len(sc), sum(sc) / len(sc), pct(fx.count("LOW") / len(fx)),
            pct(fx.count("MEDIUM") / len(fx)), pct(fx.count("HIGH") / len(fx))))
    out.append("  (para grupos 'ia', HIGH = acerto; para grupos 'real', HIGH = falso positivo)")
    return resumo, pos, neg


def gerar_graficos(series, pasta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    arquivos = []
    for nome, pos, neg in series:
        if not pos or not neg:
            continue
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        bins = [i / 20 for i in range(21)]
        a1.hist(neg, bins=bins, alpha=0.6, label="Real (n=%d)" % len(neg), color="#3b7dd8")
        a1.hist(pos, bins=bins, alpha=0.6, label="IA (n=%d)" % len(pos), color="#d84b3b")
        for t, cor in ((0.40, "#e0a000"), (0.70, "#c83232")):
            a1.axvline(t, ls="--", lw=1, color=cor)
        a1.set_xlabel("score aigcd:score:image")
        a1.set_ylabel("quantidade de imagens")
        a1.set_title("Distribuicao dos scores - %s" % nome)
        a1.legend()

        pts = curva_roc(pos, neg)
        a2.plot([p[1] for p in pts], [p[2] for p in pts], color="#d84b3b",
                label="AUC = %.3f" % roc_auc(pos, neg))
        a2.plot([0, 1], [0, 1], ls=":", color="gray")
        a2.set_xlabel("Taxa de falsos positivos (FPR)")
        a2.set_ylabel("Taxa de verdadeiros positivos (TPR)")
        a2.set_title("Curva ROC - %s" % nome)
        a2.legend(loc="lower right")
        fig.tight_layout()
        arq = os.path.join(pasta, "grafico_%s.png" % re.sub(r"[^\w-]", "_", nome))
        fig.savefig(arq, dpi=150)
        plt.close(fig)
        arquivos.append(arq)
    return arquivos


def gravar_csv(caminho, cabecalho, linhas):
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(cabecalho)
        w.writerows(linhas)


def main():
    ap = argparse.ArgumentParser(description="Metricas do plugin AIGCD a partir do CSV exportado pelo IPED.")
    ap.add_argument("csv", help="CSV exportado pelo IPED (colunas path e aigcd:score:image)")
    ap.add_argument("--log", help="Log de processamento do IPED (IPED-AAAA-MM-DD-...log) para medir tempo")
    ap.add_argument("--saida", help="Pasta dos resultados. Padrao: resultados_<nome do csv>")
    ap.add_argument("--top-erros", type=int, default=20, help="Quantos piores erros listar. Padrao: 20")
    args = ap.parse_args()

    itens = ler_export(args.csv)
    pasta = args.saida or os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                       "resultados_" + os.path.splitext(os.path.basename(args.csv))[0])
    os.makedirs(pasta, exist_ok=True)

    rotulados, sem_rotulo, nao_processadas = [], [], []
    for caminho, score in itens:
        variante, rotulo, grupo = classificar_caminho(caminho)
        eh_imagem = score is not None or extensao(caminho) in EXT_IMAGEM
        if not eh_imagem:
            continue  # pastas, documentos etc.
        if score is None:
            nao_processadas.append(caminho)
        elif rotulo:
            rotulados.append({"caminho": caminho, "score": score, "variante": variante,
                              "rotulo": rotulo, "grupo": grupo})
        else:
            sem_rotulo.append({"caminho": caminho, "score": score})

    out = ["RELATORIO - TESTE EM LARGA ESCALA DO PLUGIN AIGCD (IPED)",
           "CSV analisado: %s" % os.path.abspath(args.csv),
           "Linhas no CSV: %d | imagens com score: %d (rotuladas %d, sem rotulo %d)"
           % (len(itens), len(rotulados) + len(sem_rotulo), len(rotulados), len(sem_rotulo)),
           "Imagens SEM score (nao processadas pelo plugin): %d" % len(nao_processadas)]
    if nao_processadas:
        exts = {}
        for c in nao_processadas:
            exts[extensao(c) or "(sem ext)"] = exts.get(extensao(c) or "(sem ext)", 0) + 1
        out.append("  por extensao: %s  -> lista em nao_processadas.csv" % exts)

    if not rotulados:
        out.append("\nNenhuma imagem rotulada encontrada. Os caminhos precisam conter uma pasta 'ia' ou 'real'.")

    series, resumos = [], []
    for variante in sorted({d["variante"] for d in rotulados}):
        dados = [d for d in rotulados if d["variante"] == variante]
        resumo, pos, neg = relatorio_variante(variante, dados, out)
        resumos.append(resumo)
        series.append((variante, pos, neg))

        fps = sorted((d for d in dados if d["rotulo"] == "real"), key=lambda d: -d["score"])[:args.top_erros]
        fns = sorted((d for d in dados if d["rotulo"] == "ia"), key=lambda d: d["score"])[:args.top_erros]
        gravar_csv(os.path.join(pasta, "piores_erros_%s.csv" % variante),
                   ["tipo", "score", "faixa", "grupo", "caminho"],
                   [["falso positivo (real com score alto)", "%.4f" % d["score"], faixa(d["score"]),
                     d["grupo"], d["caminho"]] for d in fps] +
                   [["falso negativo (IA com score baixo)", "%.4f" % d["score"], faixa(d["score"]),
                     d["grupo"], d["caminho"]] for d in fns])

    if len(resumos) > 1:
        out.append("")
        out.append("=" * 78)
        out.append("COMPARACAO ENTRE VARIANTES")
        out.append("=" * 78)
        out.append("  variante              n_IA  n_Real   AUC     EER    Recall@0.70  FPR@0.70")
        for r in resumos:
            out.append("  %-20s %5d  %6d   %s  %s    %s     %s" % (
                r["variante"][:20], r["n_ia"], r["n_real"],
                "%.3f" % r["auc"] if "auc" in r else "  n/a", pct(r.get("eer", float("nan"))),
                pct(r.get("recall_0.70", float("nan"))), pct(r.get("fpr_0.70", float("nan")))))

    if sem_rotulo:
        fx = [faixa(d["score"]) for d in sem_rotulo]
        out.append("")
        out.append("IMAGENS SEM ROTULO (n=%d): LOW %s | MEDIUM %s | HIGH %s"
                   % (len(fx), pct(fx.count("LOW") / len(fx)), pct(fx.count("MEDIUM") / len(fx)),
                      pct(fx.count("HIGH") / len(fx))))
        out.append("  (indica quantos alertas HIGH o examinador receberia em dados comuns)")

    if args.log:
        info = ler_log(args.log)
        n_score = len(rotulados) + len(sem_rotulo)
        out.append("")
        out.append("DESEMPENHO (log: %s)" % os.path.basename(args.log))
        if "aigcd_s" in info:
            out.append("  AIGCDTask: %d s acumulados nas threads (%d%% do tempo das tarefas)"
                       % (info["aigcd_s"], info["aigcd_pct"]))
            if n_score:
                out.append("  Tempo medio por imagem: ~%.0f ms (tempo acumulado / %d imagens com score)"
                           % (1000.0 * info["aigcd_s"] / n_score, n_score))
        else:
            out.append("  Linha do AIGCDTask nao encontrada na tabela 'Processing Times per Task'.")
        if "total_s" in info:
            out.append("  Processamento total do caso: %d itens em %d s (tempo de parede)"
                       % (info["itens"], info["total_s"]))

    graficos = gerar_graficos(series, pasta)

    gravar_csv(os.path.join(pasta, "resultados_por_imagem.csv"),
               ["variante", "rotulo", "grupo", "score", "faixa", "acerto_limiar_0.70", "caminho"],
               [[d["variante"], d["rotulo"], d["grupo"], "%.4f" % d["score"], faixa(d["score"]),
                 "sim" if (d["score"] >= 0.70) == (d["rotulo"] == "ia") else "nao", d["caminho"]]
                for d in rotulados])
    if nao_processadas:
        gravar_csv(os.path.join(pasta, "nao_processadas.csv"), ["caminho"], [[c] for c in nao_processadas])

    out.append("")
    out.append("Arquivos gerados em: %s" % pasta)
    for g in graficos:
        out.append("  %s" % os.path.basename(g))

    texto = "\n".join(out)
    with open(os.path.join(pasta, "relatorio.txt"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(texto)


if __name__ == "__main__":
    main()
