# clinical_baseline_pipeline.py
# Extract one row per PTID using baseline visit priority + light EDA + optional imputation
# Produces:
#   - clinical_cognitive_demographic_baseline.xlsx
#   - (optional) clinical_imputed.xlsx  (if --impute is passed)
#   - plots/*.png
#
# Usage:
#   python clinical_baseline_pipeline.py --input "/path/dem_cli_cog ADNI.xlsx" --outdir "./out" --impute
#
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional import for imputation (only used if --impute flag is provided)
try:
    from DataImputation import ClassAwareKNNImputer
except Exception:
    ClassAwareKNNImputer = None


# ---------- Config ----------

VISIT_PRIORITY = {
    "bl": 1, "init": 1,
    "sc": 2, "screening": 2,
    "m03": 3, "month3": 3, "3m": 3,
    "m06": 4, "month6": 4, "6m": 4,
    "m12": 5, "month12": 5, "12m": 5,
    "m24": 6, "month24": 6, "24m": 6,
}

GENDER_MAP = {1: "male", 2: "female", "1": "male", "2": "female"}
DIAG_MAP   = {1: "CN",   2: "MCI",    3: "DEMENTIA", "1": "CN", "2": "MCI", "3": "DEMENTIA"}

PTID_TOKENS   = ["ptid", "subjectid", "subject_id", "participantid", "participant_id"]
VISIT_TOKENS  = ["visit", "visist", "viscode", "viscode2"]
GENDER_TOKENS = ["gender"]
DIAG_TOKENS   = ["diagnosis", "diagnoses", "diag"]
AGE_TOKENS    = ["entry_age", "age", "ptage", "baselineage"]

MMSE_TOKENS   = ["mmscore", "mmse"]
CDRSB_TOKENS  = ["cdr sum of boxes", "cdrsb"]
FAQ_TOKENS    = ["faq total", "faq total score", "faq"]
ADAS_TOKENS   = ["adas13", "adas 13"]

COMORB_TOKENS = ["hypertension", "stroke", "smok", "diabet", "cardio", "t2dm"]


# ---------- Helpers ----------

def normalize_colnames(cols):
    def norm(c):
        c2 = str(c).strip()
        c2 = re.sub(r"\s+", " ", c2)
        return c2
    return [norm(c) for c in cols]

def find_exact_col(df, candidate_keys):
    for c in df.columns:
        lc = c.lower().replace(" ", "")
        for cand in candidate_keys:
            if lc == cand:
                return c
    return None

def find_contains_col(df, token_list):
    for col in df.columns:
        lc = col.lower()
        for tok in token_list:
            if tok in lc:
                return col
    return None

def parse_visit_priority(raw):
    if pd.isna(raw):
        return 10000
    s = str(raw).strip().lower().replace(" ", "")
    if s in VISIT_PRIORITY:
        return VISIT_PRIORITY[s]
    m = re.match(r"m(\d+)", s)
    if m:
        try:
            months = int(m.group(1))
            base = 7
            return base + months
        except Exception:
            return 10000
    v = re.match(r"v(\d+)", s)
    if v:
        return 9000 + int(v.group(1))
    return 10000

def drop_empty_columns(df: pd.DataFrame, keep_cols=None, min_non_null=1):
    keep_cols = keep_cols or []
    drop = [c for c in df.columns if c not in keep_cols and df[c].notna().sum() < min_non_null]
    return df.drop(columns=drop), drop

def standardize_yes_no(df: pd.DataFrame, yes_tokens=("y","yes","1"), no_tokens=("n","no","0")):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            lc = out[c].astype(str).str.strip().str.lower()
            mask_yes = lc.isin(yes_tokens)
            mask_no  = lc.isin(no_tokens)
            out.loc[mask_yes, c] = "Yes"
            out.loc[mask_no,  c] = "No"
    return out

def plot_bar_counts(series, title, out_png):
    plt.figure()
    series.value_counts(dropna=False).plot(kind="bar", title=title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def boxplot_by_diag(df, feat_col, diag_col, out_png):
    import numpy as np
    plt.figure()
    ok = False
    if feat_col and diag_col and df[feat_col].notna().sum() > 0:
        groups, labels = [], []
        for dlab in df[diag_col].dropna().unique():
            vals = pd.to_numeric(df.loc[df[diag_col] == dlab, feat_col], errors="coerce").dropna().values
            if len(vals) > 0:
                groups.append(vals)
                labels.append(str(dlab))
        if groups:
            plt.boxplot(groups, labels=labels)
            plt.title(f"{feat_col} by {diag_col}")
            plt.xlabel(diag_col)
            plt.ylabel(feat_col)
            ok = True
    if not ok:
        plt.title(f"{feat_col} by {diag_col} (no data)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def missingness_heatmap(df, out_png):
    plt.figure(figsize=(8, 6))
    msk = df.isna()
    plt.imshow(msk.values, aspect="auto", interpolation="nearest")
    plt.title("Missingness heatmap (white = missing)")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ---------- Main ----------

def main():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the clinical/cognitive/demographic Excel file")
    parser.add_argument("--sheet", default=None, help="Sheet name (optional). If not set, uses the first sheet.")
    parser.add_argument("--outdir", default="./out", help="Output directory")
    parser.add_argument("--impute", action="store_true", help="Run CogNID-style class-aware KNN imputation and save clinical_imputed.xlsx")
    parser.add_argument("--target", default=None, help="Diagnosis/target column name (auto-detected if not provided)")
    parser.add_argument("--idcol", default=None, help="PTID column name (auto-detected if not provided)")
    args = parser.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(in_path)
    if args.sheet:
        df = xl.parse(args.sheet)
        sheet_used = args.sheet
    else:
        df = xl.parse(xl.sheet_names[0])
        sheet_used = xl.sheet_names[0]

    df.columns = normalize_colnames(df.columns)

    ptid_col  = args.idcol or find_exact_col(df, PTID_TOKENS)
    visit_col = find_exact_col(df, VISIT_TOKENS)
    if ptid_col is None or visit_col is None:
        print("\n[ERROR] Required columns not found.")
        print("Need PTID and VISIT-like columns. Detected columns:")
        for c in df.columns: print(" -", c)
        sys.exit(1)

    gender_col = find_contains_col(df, GENDER_TOKENS)
    diag_col   = args.target or find_contains_col(df, DIAG_TOKENS)
    age_col    = find_contains_col(df, AGE_TOKENS)

    # Baseline filter
    work = df.copy()
    work["_visit_priority"] = work[visit_col].apply(parse_visit_priority)
    work_sorted = work.sort_values(by=["_visit_priority"]).copy()
    baseline = work_sorted.drop_duplicates(subset=[ptid_col], keep="first").copy()
    baseline.drop(columns=["_visit_priority"], inplace=True)

    # Light tidy
    baseline, dropped_empty = drop_empty_columns(baseline, keep_cols=[ptid_col, visit_col, gender_col, diag_col])
    baseline = standardize_yes_no(baseline)

    # Mappings for codes
    if gender_col:
        baseline[gender_col] = baseline[gender_col].map(lambda x: GENDER_MAP.get(x, x))
    if diag_col:
        baseline[diag_col] = baseline[diag_col].map(lambda x: DIAG_MAP.get(x, x))

    # Save baseline
    baseline_xlsx = outdir / "clinical_cognitive_demographic_baseline.xlsx"
    baseline.to_excel(baseline_xlsx, index=False)
    baseline.head(50).to_csv(outdir / "clinical_baseline_preview.csv", index=False)

    # Plots
    if age_col:
        series = pd.to_numeric(baseline[age_col], errors="coerce").dropna()
        plt.figure()
        series.plot(kind="hist", bins=30, title=f"Histogram of {age_col}")
        plt.xlabel(age_col); plt.tight_layout()
        plt.savefig(plots_dir / "age_hist.png", dpi=150); plt.close()

    if gender_col:
        plot_bar_counts(baseline[gender_col], f"Gender distribution ({gender_col})", plots_dir / "gender_bar.png")

    geno_col = find_contains_col(baseline, ["genotype", "apoe"])
    if geno_col:
        plot_bar_counts(baseline[geno_col], f"Genotype distribution ({geno_col})", plots_dir / "genotype_bar.png")

    for tokens, fname in [
        (MMSE_TOKENS, "mmse_by_diag.png"),
        (CDRSB_TOKENS, "cdrsb_by_diag.png"),
        (FAQ_TOKENS, "faq_by_diag.png"),
        (ADAS_TOKENS, "adas13_by_diag.png"),
    ]:
        feat_col = find_contains_col(baseline, tokens)
        boxplot_by_diag(baseline, feat_col, diag_col, plots_dir / fname)

    comorb_cols = [c for c in baseline.columns if any(tok in c.lower() for tok in COMORB_TOKENS)]
    if comorb_cols:
        counts = {}
        for c in comorb_cols:
            vc = baseline[c].value_counts(dropna=False)
            pos = 0
            if 1 in vc.index: pos = max(pos, int(vc.get(1, 0)))
            if "Yes" in vc.index: pos = max(pos, int(vc.get("Yes", 0)))
            counts[c] = pos
        ser = pd.Series(counts).sort_values(ascending=False) if counts else pd.Series(dtype=int)
        plt.figure()
        if not ser.empty:
            ser.plot(kind="bar", title="Comorbidity prevalence (heuristic positives)")
        else:
            plt.title("Comorbidity prevalence (no positives found)")
        plt.tight_layout(); plt.savefig(plots_dir / "comorbidities_bar.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 6))
    msk = baseline.isna()
    plt.imshow(msk.values, aspect="auto", interpolation="nearest")
    plt.title("Missingness heatmap (white = missing)"); plt.xlabel("Columns"); plt.ylabel("Rows")
    plt.colorbar(); plt.tight_layout(); plt.savefig(plots_dir / "missingness_heatmap.png", dpi=150); plt.close()

    if args.impute:
        if ClassAwareKNNImputer is None:
            print("[ERROR] DataImputation.py not found. Place it beside this script or in PYTHONPATH.")
            sys.exit(1)
        imputer = ClassAwareKNNImputer(
            target_col=diag_col if diag_col else "Diagnosis",
            id_cols=[ptid_col],
            n_neighbors=5,
            add_noise_frac=0.01,
            clip_strategy="iqr",
            percentile_bounds=(1, 99),
            min_class_size=5,
            random_state=42
        )
        df_imputed, report = imputer.fit_transform(baseline)
        imputed_xlsx = outdir / "clinical_imputed.xlsx"
        df_imputed.to_excel(imputed_xlsx, index=False)
        (outdir / "imputation_report.txt").write_text(str(report))

    print("\n=== Clinical baseline pipeline complete ===")
    print(f"Input:         {in_path}")
    print(f"Sheet used:    {sheet_used}")
    print(f"Baseline out:  {baseline_xlsx}")
    if args.impute:
        print(f"Imputed out:   {imputed_xlsx}")
        print(f"Report:        {outdir / 'imputation_report.txt'}")
    print(f"Plots dir:     {plots_dir.resolve()}")


if __name__ == '__main__':
    main()
