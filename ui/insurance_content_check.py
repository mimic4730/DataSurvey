# ui/insurance_content_check.py
from __future__ import annotations
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd
from datetime import datetime as _dt

from core.io_utils import CsvLoader
from core import inspection


class InsuranceContentChecker:
    """保険情報の【内容】検収（項目値の一致判定）。
    - 元CSV と 突合CSV（検収用/他システム出力）を読み、
      キー = (患者番号, 保険者番号) で突合し、以下の項目の一致/不一致を判定。
        * 患者負担割合
        * 保険開始日
        * 保険終了日
        * 保険証記号
        * 保険証番号
    - マッピングは検収生成時と同様（プリセット）を利用可能。
    """

    def __init__(self, logger=None, preset_colmap: dict | None = None, migration_date_yyyymmdd: str | None = None):
        self._logger = logger
        self._preset = preset_colmap or {}
        self._migration_yyyymmdd = migration_date_yyyymmdd

    def log(self, msg: str):
        if self._logger:
            try:
                self._logger(msg)
            except Exception:
                pass

    def _prepare_output_dir(self, in_path: str | Path, kind: str) -> Path:
        base = Path(in_path).resolve().parent
        tag = _dt.now().strftime("%Y%m%d")
        out_dir = base / "検収結果" / f"{kind}_内容_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # ---- Normalizers ----
    def _normalize_codes(self, s: pd.Series, width: int, mode: str = "zfill") -> pd.Series:
        import re, unicodedata
        digits = s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
        if mode == "zfill":
            return digits.map(lambda x: x.zfill(width) if x else "")
        elif mode == "lstrip":
            return digits.map(lambda x: x.lstrip("0"))
        return digits

    @staticmethod
    def _digits_only(s: pd.Series) -> pd.Series:
        import re, unicodedata
        return s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))

    @staticmethod
    def _text_norm(s: pd.Series) -> pd.Series:
        import unicodedata, re
        t = s.astype(str).map(lambda x: unicodedata.normalize("NFKC", x))
        return t.map(lambda x: re.sub(r"\s+", " ", x).strip())

    @staticmethod
    def _ratio_norm(s: pd.Series) -> pd.Series:
        # 3割/2割/1割 → 0.3/0.2/0.1
        def _map(v: str) -> str:
            t = str(v).strip()
            if t in {"3割", "３割", "0.3", "0.30", "30%", "３０％"}: return "0.3"
            if t in {"2割", "２割", "0.2", "0.20", "20%", "２０％"}: return "0.2"
            if t in {"1割", "１割", "0.1", "0.10", "10%", "１０％"}: return "0.1"
            return ""
        return s.astype(str).map(_map)

    @staticmethod
    def _date_norm(s: pd.Series) -> pd.Series:
        return s.map(lambda v: inspection._parse_date_any_to_yyyymmdd(v))

    @staticmethod
    def _payer_norm(s: pd.Series) -> pd.Series:
        # 7桁→8桁ゼロ埋め、5桁→6桁ゼロ埋め、他は数字のみ
        import re, unicodedata
        digits = s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
        def _pad(v: str) -> str:
            if not v: return ""
            if len(v) == 7: return v.zfill(8)
            if len(v) == 5: return v.zfill(6)
            return v
        return digits.map(_pad)

    # ---- Main ----
    def run(self, app):
        # 1) 元CSV
        in_path = filedialog.askopenfilename(title="保険情報（元CSV）を選択", filetypes=[("CSV files", "*.csv")])
        self.log(f"[保険-内容] 元CSV: {in_path}")
        if not in_path:
            return
        src = CsvLoader.read_csv_flex(in_path)
        out_dir = self._prepare_output_dir(in_path, "保険")

        # 2) マッピング（プリセット適用）
        required = ["患者番号", "保険者番号", "患者負担割合", "保険開始日", "保険終了日", "保険証記号", "保険証番号"]
        colmap = app._ask_inspection_colmap(src, required_cols=required, preset=self._preset)
        if colmap is None:
            return
        self.log(f"[保険-内容] 元マッピング: {colmap}")

        # 3) 突合CSV（検収用）
        cmp_path = filedialog.askopenfilename(title="突合用（検収CSV/他システム出力）を選択", filetypes=[("CSV files", "*.csv")])
        self.log(f"[保険-内容] 突合CSV: {cmp_path}")
        if not cmp_path:
            return
        cmp_df = CsvLoader.read_csv_flex(cmp_path)
        for need in ("患者番号", "保険者番号", "患者負担割合", "保険開始日", "保険終了日", "保険証記号", "保険証番号"):
            if need not in cmp_df.columns:
                messagebox.showerror("エラー", f"突合CSVに『{need}』列がありません。")
                return

        mig = self._migration_yyyymmdd
        if not mig and hasattr(getattr(app, "actions", None), "_get_migration_date"):
            try:
                mig = app.actions._get_migration_date()  # 既存の共通取得関数
            except Exception:
                mig = None
        if mig:
            mig = inspection._parse_date_any_to_yyyymmdd(mig) or None
        self.log(f"[保険-内容] 比較用の移行日(開始日空欄の埋め草): {mig or '(未設定)'}")        

        # 4) 幅決定（患者番号）
        import unicodedata, re
        def _digits_len_max(s: pd.Series) -> int:
            return int(s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x))).str.len().max() or 0)
        src_pat = src[colmap.get("患者番号")] if colmap.get("患者番号") in src.columns else pd.Series([], dtype="object")
        cmp_pat = cmp_df["患者番号"]
        width = max(_digits_len_max(src_pat) if len(src_pat) else 0, _digits_len_max(cmp_pat), 1)
        self.log(f"[保険-内容] 患者番号幅: {width}")

        # 5) 元CSV 正規化
        src_pat_norm   = self._normalize_codes(src[colmap.get("患者番号")].astype(str), width, mode="zfill") if colmap.get("患者番号") in src.columns else pd.Series([""]*len(src))
        src_payer_norm = self._payer_norm(src[colmap.get("保険者番号")]) if colmap.get("保険者番号") in src.columns else pd.Series([""]*len(src))
        src_ratio      = self._ratio_norm(src[colmap.get("患者負担割合")]) if colmap.get("患者負担割合") in src.columns else pd.Series([""]*len(src))
        src_start      = self._date_norm(src[colmap.get("保険開始日")]) if colmap.get("保険開始日") in src.columns else pd.Series([""]*len(src))
        src_end        = self._date_norm(src[colmap.get("保険終了日")]) if colmap.get("保険終了日") in src.columns else pd.Series([""]*len(src))
        src_kigo       = self._text_norm(src[colmap.get("保険証記号")]) if colmap.get("保険証記号") in src.columns else pd.Series([""]*len(src))
        src_bango      = self._digits_only(src[colmap.get("保険証番号")]) if colmap.get("保険証番号") in src.columns else pd.Series([""]*len(src))

        if mig:
            src_start = src_start.map(lambda v, m=mig: m if (not v or str(v).strip() == "") else v)

        src_norm = pd.DataFrame({
            "患者番号": src_pat_norm,
            "保険者番号": src_payer_norm,
            "患者負担割合": src_ratio,
            "保険開始日": src_start,
            "保険終了日": src_end,
            "保険証記号": src_kigo,
            "保険証番号": src_bango,
        })
        
        def _get_cmp(col: str) -> pd.Series:
            return cmp_df[col] if col in cmp_df.columns else pd.Series([""]*len(cmp_df))

        # 6) 突合側 正規化
        cmp_pat_norm   = self._normalize_codes(cmp_df["患者番号"].astype(str), width, mode="zfill")
        cmp_payer_norm = self._payer_norm(cmp_df["保険者番号"]) if "保険者番号" in cmp_df.columns else pd.Series([""]*len(cmp_df))

        cmp_norm = pd.DataFrame({
            "患者番号": cmp_pat_norm,
            "保険者番号": cmp_payer_norm,
            "患者負担割合": self._ratio_norm(cmp_df["患者負担割合"]),
            "保険開始日": self._date_norm(cmp_df["保険開始日"]),
            "保険終了日": self._date_norm(cmp_df["保険終了日"]),
            "保険証記号": self._text_norm(cmp_df["保険証記号"]),
            "保険証番号": self._digits_only(cmp_df["保険証番号"]),
        })
        
        
        # 6.5) 完全重複の除去（キー＋比較列が全て同一の行のみ）
        #  - 部分的に異なる重複（例：同じキーで値が違う）は残して不一致検出に回す
        dup_subset = ["患者番号", "保険者番号", "患者負担割合", "保険開始日", "保険終了日", "保険証記号", "保険証番号"]
        src_norm = src_norm.drop_duplicates(subset=dup_subset, keep="first")
        cmp_norm = cmp_norm.drop_duplicates(subset=dup_subset, keep="first")

        # 7) キー突合 & 未ヒット
        src_norm = src_norm[(src_norm["患者番号"] != "") & (src_norm["保険者番号"] != "")].copy()
        cmp_norm = cmp_norm[(cmp_norm["患者番号"] != "") & (cmp_norm["保険者番号"] != "")].copy()

        cmp_key_set = set(zip(cmp_norm["患者番号"], cmp_norm["保険者番号"]))
        src_keys = list(zip(src_norm["患者番号"], src_norm["保険者番号"]))
        missing_mask = pd.Series([k not in cmp_key_set for k in src_keys], index=src_norm.index)

        missing_df = src.loc[src_norm.index.intersection(src_norm[missing_mask].index)].copy()
        if not missing_df.empty:
            missing_df.insert(0, "__正規化保険者番号__", src_norm.loc[missing_df.index, "保険者番号"])
            missing_df.insert(0, "__正規化患者番号__", src_norm.loc[missing_df.index, "患者番号"])

        # 8) 一致/不一致
        merged = src_norm.merge(cmp_norm, on=["患者番号", "保険者番号"], how="inner", suffixes=("_src", "_cmp"))
        fields = ["患者負担割合", "保険開始日", "保険終了日", "保険証記号", "保険証番号"]

        all_eq_mask = pd.Series([True] * len(merged), index=merged.index)
        for f in fields:
            all_eq_mask &= (merged[f + "_src"] == merged[f + "_cmp"])

        matched_rows = merged.loc[all_eq_mask, ["患者番号", "保険者番号"] + [f + "_src" for f in fields]].copy()
        matched_rows.rename(columns={f + "_src": f for f in fields}, inplace=True)

        mismatches = []
        for f in fields:
            neq = merged.loc[merged[f + "_src"] != merged[f + "_cmp"], ["患者番号", "保険者番号", f + "_src", f + "_cmp"]].copy()
            if not neq.empty:
                neq.insert(2, "項目名", f)
                neq.rename(columns={f + "_src": "正規化_元", f + "_cmp": "正規化_突合"}, inplace=True)
                mismatches.append(neq)
        mismatch_df = pd.concat(mismatches, axis=0) if mismatches else pd.DataFrame(
            columns=["患者番号", "保険者番号", "項目名", "正規化_元", "正規化_突合"]
        )

        # 9) 出力
        tag = _dt.now().strftime("%Y%m%d")
        out_matched  = out_dir / f"保険_内容_一致_{tag}.csv"
        out_mismatch = out_dir / f"保険_内容_不一致_{tag}.csv"
        out_missing  = out_dir / f"保険_内容_未ヒット_{tag}.csv"
        inspection.to_csv(matched_rows, str(out_matched))
        inspection.to_csv(mismatch_df, str(out_mismatch))
        inspection.to_csv(missing_df, str(out_missing))

        self.log(
            f"[保険-内容] 一致: {len(matched_rows)} / 不一致明細行: {len(mismatch_df)} / 未ヒット: {len(missing_df)}\n"
            f"  (src={len(src_norm)}, cmp={len(cmp_norm)})"
        )
        messagebox.showinfo(
            "保険内容検収 完了",
            f"一致: {len(matched_rows)} 件\n不一致明細: {len(mismatch_df)} 行（項目単位）\n未ヒット: {len(missing_df)} 件\n\n出力先:\n{out_dir}"
        )
        return True


# ===== Entry points for InspectionActions =====
def run_insurance_content_check(app, logger=None, preset=None):
    checker = InsuranceContentChecker(logger=logger, preset_colmap=preset)
    return bool(checker.run(app))

def run(app, logger=None, preset=None):
    checker = InsuranceContentChecker(logger=logger, preset_colmap=preset)
    return bool(checker.run(app))

def main(app, logger=None, preset=None):
    checker = InsuranceContentChecker(logger=logger, preset_colmap=preset)
    return bool(checker.run(app))