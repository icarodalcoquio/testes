#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import difflib
import math
import argparse
import pandas as pd
import numpy as np


# =========================
# Utilitários
# =========================
def pad8(x):
    """Normaliza códigos para 8 dígitos como string."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    digits = re.sub(r"\D", "", s)
    if digits == "":
        return None
    return digits.zfill(8)


def find_col(df, patterns):
    """Acha a primeira coluna cujo nome casa com algum regex (case-insensitive)."""
    for c in df.columns:
        name = str(c)
        for p in patterns:
            if re.search(p, name, flags=re.I):
                return c
    return None


def similaridade(a, b):
    """Similaridade textual (0..1) entre duas descrições."""
    if a is None or b is None:
        return np.nan
    a = str(a).strip()
    b = str(b).strip()
    if not a or not b:
        return np.nan
    return round(difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio(), 4)


def alerta(score):
    if pd.isna(score):
        return ""
    if score >= 0.85:
        return "OK"
    if score >= 0.60:
        return "VERIFICAR"
    return "ALTA_MUDANCA"


def classify_transposition(parents, outdeg_by_old):
    """
    1-1: 1 pai e esse pai gera 1 filho
    split: 1 pai e esse pai gera >1 filho
    merge: >1 pais e todos esses pais geram 1 filho
    cluster: N↔N (bolsa) — >1 pais e algum pai gera >1 filho
    sem-correlacao: nenhum pai encontrado
    """
    n = len(parents)
    if n == 0:
        return "sem-correlacao"
    if n == 1:
        od = outdeg_by_old.get(parents[0], 0)
        return "1-1" if od == 1 else "split"
    ods = [outdeg_by_old.get(p, 0) for p in parents]
    return "merge" if all(o == 1 for o in ods) else "cluster"


def concat_desc_pais(parents, desc_by_old, max_items=8):
    descs = []
    for p in parents:
        d = str(desc_by_old.get(p, "")).strip()
        if d:
            descs.append(d)
    uniq = list(dict.fromkeys(descs))
    if not uniq:
        return ""
    if len(uniq) <= max_items:
        return " | ".join(uniq)
    return " | ".join(uniq[:max_items]) + " | ..."


# =========================
# Stage (opcional) – mantenho, mas você pode remover se quiser
# =========================
def stage_to_cronograma(stage):
    if stage is None or str(stage).strip() == "":
        return ""
    s = str(stage).strip().upper()
    if s == "FREE":
        return "Duty free"
    if s == "E":
        return "Excluded"
    if s == "TRQ":
        return "TQR"
    if re.fullmatch(r"\d+", s):
        return s
    m = re.fullmatch(r"FP(\d+)%Y(\d+)", s)
    if m:
        return f"Tariff preference {m.group(1)}% Year {m.group(2)}"
    return str(stage).strip()


def parse_fp(stage):
    if stage is None:
        return None
    s = str(stage).strip().upper()
    m = re.fullmatch(r"FP(\d+)%Y(\d+)", s)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def choose_stage_conservative(stages):
    clean = [str(s).strip().upper() for s in stages if s is not None and str(s).strip() != ""]
    if not clean:
        return ""
    if len(set(clean)) == 1:
        return clean[0]
    if "E" in clean:
        return "E"
    if "TRQ" in clean:
        return "TRQ"
    fp_pairs = [parse_fp(s) for s in clean]
    fp_pairs = [t for t in fp_pairs if t is not None]  # FIX: filtra None antes de desempacotar
    if fp_pairs:
        pct_min = min(p for p, _ in fp_pairs)
        yr_max = max(y for p, y in fp_pairs if p == pct_min)
        return f"FP{pct_min}%Y{yr_max}"
    nums = [int(s) for s in clean if re.fullmatch(r"\d+", s)]
    if nums:
        return str(max(nums))
    if "FREE" in clean:
        return "FREE"
    return clean[0]


# =========================
# Main
# =========================
def main(path_orig, path_trans, path_corr, out_path):
    df_orig = pd.read_excel(path_orig)
    df_trans = pd.read_excel(path_trans)
    df_corr = pd.read_excel(path_corr)

    # Remove "Unnamed"
    df_orig = df_orig.loc[:, ~df_orig.columns.astype(str).str.match(r"^Unnamed")]
    df_trans = df_trans.loc[:, ~df_trans.columns.astype(str).str.match(r"^Unnamed")]
    df_corr = df_corr.loc[:, ~df_corr.columns.astype(str).str.match(r"^Unnamed")]

    # Identifica colunas (heurística)
    col_old_code = find_col(df_orig, [r"^Tariff\s*Line$", r"tariff", r"\bncm\b"])
    col_old_stage = find_col(df_orig, [r"staging", r"categoria", r"cronograma"])
    col_old_desc = find_col(df_orig, [r"^Description$", r"descri", r"description", r"descripcion", r"descrição"])

    col_new_code = find_col(df_trans, [r"\bncm\b", r"tariff"])
    col_new_desc = find_col(df_trans, [r"descri", r"description", r"descripcion", r"descrição"])

    col_c_old = find_col(df_corr, [r"^NCM_2012$", r"2012"])
    col_c_new = find_col(df_corr, [r"^NCM_2017$", r"2017", r"2021"])

    if col_old_code is None or col_old_stage is None:
        raise ValueError(f"Oferta original: não achei colunas chave. code={col_old_code}, stage={col_old_stage}")
    if col_new_code is None:
        raise ValueError("Oferta transposta: não achei coluna de NCM/código.")
    if col_c_old is None or col_c_new is None:
        raise ValueError(f"Correlação: não achei colunas NCM_2012/NCM_2017. old={col_c_old}, new={col_c_new}")

    if col_old_desc is None:
        df_orig["__desc_old__"] = ""
        col_old_desc = "__desc_old__"
    if col_new_desc is None:
        df_trans["__desc_new__"] = ""
        col_new_desc = "__desc_new__"

    # ---- Oferta original (pais)
    orig = df_orig[[col_old_code, col_old_stage, col_old_desc]].copy()
    orig.columns = ["code_old", "stage_old", "desc_old"]
    orig["code_old"] = orig["code_old"].apply(pad8)
    orig["stage_old"] = orig["stage_old"].astype(str).str.strip()
    orig["desc_old"] = orig["desc_old"].astype(str).str.strip()
    orig = orig.dropna(subset=["code_old"]).drop_duplicates(subset=["code_old"])

    stage_by_old = orig.set_index("code_old")["stage_old"].to_dict()
    desc_by_old = orig.set_index("code_old")["desc_old"].to_dict()

    # ---- Correlação (pais -> filhos)
    corr = df_corr[[col_c_old, col_c_new]].copy()
    corr.columns = ["NCM_2012", "NCM_2017"]
    corr["NCM_2012"] = corr["NCM_2012"].apply(pad8)
    corr["NCM_2017"] = corr["NCM_2017"].apply(pad8)
    corr = corr.dropna().drop_duplicates()

    # Usamos toda a correlação cujos pais existem na oferta original
    corr_use = corr[corr["NCM_2012"].isin(stage_by_old.keys())].copy()

    parents_by_new = corr_use.groupby("NCM_2017")["NCM_2012"].apply(lambda s: sorted(set(s.dropna()))).to_dict()
    outdeg_by_old = corr_use.groupby("NCM_2012").size().to_dict()
    repet_by_new = corr_use.groupby("NCM_2017").size().to_dict()

    # ---- Oferta transposta (mantém TODAS as linhas)
    trans_all = df_trans.copy()
    trans_all["__code_new__"] = trans_all[col_new_code].apply(pad8)
    trans_all["__desc_new__"] = trans_all[col_new_desc].astype(str).str.strip()

    def compute_row(code_new, desc_new):
        # garante preenchimento mesmo sem correlação
        if not code_new:
            return ("", "", 0, 0, "", "", "", np.nan, "")
        parents = parents_by_new.get(code_new, [])
        tipo = classify_transposition(parents, outdeg_by_old)
        referencia = ",".join(parents) if parents else ""
        padres = len(parents)
        rep = int(repet_by_new.get(code_new, 0)) if code_new else 0

        # stage/cronograma só se houver pais
        stage = choose_stage_conservative([stage_by_old.get(p, "") for p in parents]) if parents else ""
        cron = stage_to_cronograma(stage) if stage else ""

        desc_origem = concat_desc_pais(parents, desc_by_old) if parents else ""
        sim = similaridade(desc_origem, desc_new)
        return (tipo, referencia, padres, rep, stage, cron, desc_origem, sim, alerta(sim))

    rows = trans_all.apply(
        lambda r: compute_row(r["__code_new__"], r["__desc_new__"]),
        axis=1,
        result_type="expand",
    )
    rows.columns = [
        "Tipo transposição",
        "Referencia NCM (antigas)",
        "Padres",
        "Repeticiones",
        "Stage escolhido",
        "Cronograma escolhido",
        "Descrição origem (pais)",
        "Similaridade descr.",
        "Alerta descrição",
    ]

    result = pd.concat([trans_all.drop(columns=["__desc_new__"]), rows], axis=1)
    result.insert(0, "NCM destino (normalizado)", trans_all["__code_new__"])

    # salva
    result.to_excel(out_path, index=False)
    print("OK:", out_path)
    print("Linhas (oferta transposta):", len(df_trans))
    print("Linhas com NCM válido:", int(trans_all["__code_new__"].notna().sum()))
    print("Sem correlação:", int((rows["Tipo transposição"] == "sem-correlacao").sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transposição NCM (mantém todas as linhas; sem comparação externa).")
    parser.add_argument("--orig", default="oferta original.xlsx")
    parser.add_argument("--trans", default="oferta transposta.xlsx")
    parser.add_argument("--corr", default="tabela de correlação.xlsx")
    parser.add_argument("--out", default="oferta_transposta_enriquecida.xlsx")
    args = parser.parse_args()
    main(args.orig, args.trans, args.corr, args.out)
