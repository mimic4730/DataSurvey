# ui/public_content_check.py
from __future__ import annotations
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd
from datetime import datetime as _dt

from core.io_utils import CsvLoader
from core import inspection


class PublicContentChecker:
    """
    公費情報の【内容】検収（項目値の一致判定）。

    元CSV と 突合CSV（検収用/他システム出力）を読み、
    キー = (患者番号, 公費負担者番号) で突合し、以下の項目の一致/不一致を判定。
      * 公費受給者番号
      * 公費開始日
    対象スロット: 1 / 2 を縦持ち展開して比較。

    ・開始日が「移行日」より未来（超未来）の場合は、移行日に置換して比較。
    ・患者番号は両CSVから桁幅を推定して0埋め統一。
    ・負担者/受給者番号は数字だけ比較（桁の0埋め差は吸収）。
    """

    def __init__(self, logger=None, preset_colmap: dict | None = None, migration_date_yyyymmdd: str | None = None):
        self._logger = logger
        self._preset = preset_colmap or {}
        self._migration_yyyymmdd = migration_date_yyyymmdd

    # ---------- utils ----------
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

    @staticmethod
    def _digits_only_series(s: pd.Series) -> pd.Series:
        import re, unicodedata
        return s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))

    def _normalize_codes(self, s: pd.Series, width: int, mode: str = "zfill") -> pd.Series:
        digits = self._digits_only_series(s)
        if mode == "zfill":
            return digits.map(lambda x: x.zfill(width) if x else "")
        elif mode == "lstrip":
            return digits.map(lambda x: x.lstrip("0"))
        return digits

    @staticmethod
    def _date_norm_series(s: pd.Series) -> pd.Series:
        return s.map(lambda v: inspection._parse_date_any_to_yyyymmdd(v))

    @staticmethod
    def _min_with_cap_to_migration(date_yyyymmdd: str | None, migration_yyyymmdd: str | None) -> str:
        if not migration_yyyymmdd:
            return date_yyyymmdd or ""
        # 空欄なら移行日に補完
        if not date_yyyymmdd:
            return migration_yyyymmdd
        # 未来日は移行日に丸める
        try:
            return migration_yyyymmdd if date_yyyymmdd > migration_yyyymmdd else date_yyyymmdd
        except Exception:
            return date_yyyymmdd

    # ---------- reshape helpers ----------
    def _unpivot_public_slots(self, df: pd.DataFrame, mapping: dict[str, str], slot: int) -> pd.DataFrame:
        """
        mapping からスロット列を拾って縦持ち行へ変換。
        期待するキー:
         - f"公費負担者番号{slot}"
         - f"公費受給者番号{slot}"
         - f"公費開始日{slot}"
        （不足時は空欄列で補う）
        """
        pat_col = mapping.get("患者番号")
        payer_col = mapping.get(f"公費負担者番号{slot}")
        recip_col = mapping.get(f"公費受給者番号{slot}")
        start_col = mapping.get(f"公費開始日{slot}")

        def _col_or_empty(colname: str) -> pd.Series:
            return df[colname] if colname and colname in df.columns else pd.Series([""] * len(df), index=df.index, dtype="object")

        out = pd.DataFrame({
            "患者番号": _col_or_empty(pat_col),
            "公費負担者番号": _col_or_empty(payer_col),
            "公費受給者番号": _col_or_empty(recip_col),
            "公費開始日": _col_or_empty(start_col),
        }, index=df.index)
        return out

    def _build_mapping_with_alias_for_cmp(self, cmp_df: pd.DataFrame) -> dict[str, str]:
        """
        突合側（検収用CSV）の列名ゆれを吸収するための簡易マッピングを構築。
        まず固定名があれば優先し、なければエイリアス候補から拾う。
        """
        aliases = {
            "患者番号": ["患者番号", "患者コード", "No", "通番"],
            "公費負担者番号1": ["公費負担者番号1", "第一公費負担者番号", "第一公費番号", "公費負担者番号１"],
            "公費受給者番号1": ["公費受給者番号1", "第一公費受給者番号", "公費受給者番号１"],
            "公費開始日1": ["公費開始日1", "第一公費開始日", "公費開始日１"],
            "公費負担者番号2": ["公費負担者番号2", "第2公費負担者番号", "第二公費負担者番号", "第２公費負担者番号", "公費負担者番号２"],
            "公費受給者番号2": ["公費受給者番号2", "第2公費受給者番号", "第二公費受給者番号", "第２公費受給者番号", "公費受給者番号２"],
            "公費開始日2": ["公費開始日2", "第2公費開始日", "第二公費開始日", "第２公費開始日", "公費開始日２"],
        }

        def pick(cols: list[str]) -> str | None:
            for c in cols:
                if c in cmp_df.columns:
                    return c
            return None

        return {
            "患者番号": pick(aliases["患者番号"]),
            "公費負担者番号1": pick(aliases["公費負担者番号1"]),
            "公費受給者番号1": pick(aliases["公費受給者番号1"]),
            "公費開始日1": pick(aliases["公費開始日1"]),
            "公費負担者番号2": pick(aliases["公費負担者番号2"]),
            "公費受給者番号2": pick(aliases["公費受給者番号2"]),
            "公費開始日2": pick(aliases["公費開始日2"]),
        }

    # ---------- main ----------
    def run(self, app):
        # 1) 元CSV
        in_path = filedialog.askopenfilename(title="公費情報（元CSV）を選択", filetypes=[("CSV files", "*.csv")])
        self.log(f"[公費-内容] 元CSV: {in_path}")
        if not in_path:
            return
        src = CsvLoader.read_csv_flex(in_path)
        out_dir = self._prepare_output_dir(in_path, "公費")

        # 2) マッピング（プリセット適用）— 元CSV側
        required_src = ["患者番号", "公費負担者番号１", "公費受給者番号１", "公費開始日１",
                        "公費負担者番号２", "公費受給者番号２", "公費開始日２"]
        # UIのラベルは全角1/2のことがあるので内部キーは 1/2 で扱う
        # app 側のダイアログは required_cols の表記に従うため、ここではそのまま要求
        colmap_src = app._ask_inspection_colmap(src, required_cols=required_src, preset=self._preset)
        if colmap_src is None:
            return
        self.log(f"[公費-内容] 元マッピング: {colmap_src}")

        # 3) 突合CSV（検収用/他システム出力）
        cmp_path = filedialog.askopenfilename(title="突合用（検収CSV/他システム出力）を選択", filetypes=[("CSV files", "*.csv")])
        self.log(f"[公費-内容] 突合CSV: {cmp_path}")
        if not cmp_path:
            return
        cmp_df = CsvLoader.read_csv_flex(cmp_path)

        # 突合側の列名は固定/別名の可能性があるので、別名対応マッピングを生成
        cmp_map = self._build_mapping_with_alias_for_cmp(cmp_df)
        if cmp_map.get("患者番号") is None:
            messagebox.showerror("エラー", "突合CSVに『患者番号』列が見つかりません。")
            return

        # 4) 患者番号の桁幅決定
        import unicodedata, re
        def _digits_len_max(s: pd.Series) -> int:
            return int(s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x))).str.len().max() or 0)

        src_pat = src[colmap_src.get("患者番号")] if colmap_src.get("患者番号") in src.columns else pd.Series([], dtype="object")
        cmp_pat = cmp_df[cmp_map["患者番号"]]
        width = max(_digits_len_max(src_pat) if len(src_pat) else 0, _digits_len_max(cmp_pat), 1)
        self.log(f"[公費-内容] 患者番号幅: {width}")

        # 5) 元CSV → 縦持ち & 正規化
        #   - スロット1/2を展開
        #   - 数字列は digitsOnly（受給者/負担者）
        #   - 日付はyyyymmdd
        #   - 開始日が移行日より未来なら移行日に丸める
        mig = self._migration_yyyymmdd
        if not mig and hasattr(getattr(app, "actions", None), "_get_migration_date"):
            try:
                mig = app.actions._get_migration_date()
            except Exception:
                mig = None
        if mig:
            mig = inspection._parse_date_any_to_yyyymmdd(mig) or None
        self.log(f"[公費-内容] 比較用の移行日(未来開始日の置換基準): {mig or '(未設定)'}")

        # UIの required_src は全角の “１/２” だが、内部処理のスロット番号は 1/2 に合わせる
        # 一旦、処理用にマップし直す
        def _map_slot_names(colmap_src: dict) -> dict:
            m = dict(colmap_src)
            # 「１」→「1」、「２」→「2」 のエイリアス
            repl = {
                "公費負担者番号１": "公費負担者番号1",
                "公費受給者番号１": "公費受給者番号1",
                "公費開始日１": "公費開始日1",
                "公費負担者番号２": "公費負担者番号2",
                "公費受給者番号２": "公費受給者番号2",
                "公費開始日２": "公費開始日2",
            }
            for k_j, k_h in repl.items():
                if k_j in m and k_h not in m:
                    m[k_h] = m[k_j]
            return m

        src_map_proc = _map_slot_names(colmap_src)

        src_1 = self._unpivot_public_slots(src, {"患者番号": src_map_proc.get("患者番号"),
                                                 "公費負担者番号1": src_map_proc.get("公費負担者番号1"),
                                                 "公費受給者番号1": src_map_proc.get("公費受給者番号1"),
                                                 "公費開始日1": src_map_proc.get("公費開始日1")}, slot=1)
        src_2 = self._unpivot_public_slots(src, {"患者番号": src_map_proc.get("患者番号"),
                                                 "公費負担者番号2": src_map_proc.get("公費負担者番号2"),
                                                 "公費受給者番号2": src_map_proc.get("公費受給者番号2"),
                                                 "公費開始日2": src_map_proc.get("公費開始日2")}, slot=2)
        src_long = pd.concat([src_1, src_2], axis=0, ignore_index=True)

        src_long["患者番号"] = self._normalize_codes(src_long["患者番号"], width, mode="zfill")
        src_long["公費負担者番号"] = self._digits_only_series(src_long["公費負担者番号"])
        src_long["公費受給者番号"] = self._digits_only_series(src_long["公費受給者番号"])
        src_long["公費開始日"] = self._date_norm_series(src_long["公費開始日"])
        if mig:
            src_long["公費開始日"] = src_long["公費開始日"].map(lambda v, m=mig: self._min_with_cap_to_migration(v, m))

        # 空負担者は無効行として落とす（比較対象外）
        src_long = src_long[(src_long["患者番号"] != "") & (src_long["公費負担者番号"] != "")].copy()

        # 6) 突合側 → 縦持ち & 正規化
        # cmp_map を内部キーへ変換
        def _cmp_pick(name: str) -> pd.Series:
            col = cmp_map.get(name)
            return cmp_df[col] if col else pd.Series([""] * len(cmp_df))

        cmp_1 = pd.DataFrame({
            "患者番号": _cmp_pick("患者番号"),
            "公費負担者番号": _cmp_pick("公費負担者番号1"),
            "公費受給者番号": _cmp_pick("公費受給者番号1"),
            "公費開始日": _cmp_pick("公費開始日1"),
        })
        cmp_2 = pd.DataFrame({
            "患者番号": _cmp_pick("患者番号"),
            "公費負担者番号": _cmp_pick("公費負担者番号2"),
            "公費受給者番号": _cmp_pick("公費受給者番号2"),
            "公費開始日": _cmp_pick("公費開始日2"),
        })
        cmp_long = pd.concat([cmp_1, cmp_2], axis=0, ignore_index=True)

        cmp_long["患者番号"] = self._normalize_codes(cmp_long["患者番号"], width, mode="zfill")
        cmp_long["公費負担者番号"] = self._digits_only_series(cmp_long["公費負担者番号"])
        cmp_long["公費受給者番号"] = self._digits_only_series(cmp_long["公費受給者番号"])
        cmp_long["公費開始日"] = self._date_norm_series(cmp_long["公費開始日"])
        if mig:
            cmp_long["公費開始日"] = cmp_long["公費開始日"].map(lambda v, m=mig: self._min_with_cap_to_migration(v, m))

        cmp_long = cmp_long[(cmp_long["患者番号"] != "") & (cmp_long["公費負担者番号"] != "")].copy()

        # 完全重複（キー＋比較対象列がすべて同じ）だけを事前に落として直積膨張を防ぐ
        subset_cols = ["患者番号", "公費負担者番号", "公費受給者番号", "公費開始日"]
        src_long = src_long.drop_duplicates(subset=subset_cols, keep="first")
        cmp_long = cmp_long.drop_duplicates(subset=subset_cols, keep="first")

        # 7) 未ヒット（元にあるが突合に無いキー）
        cmp_key_set = set(zip(cmp_long["患者番号"], cmp_long["公費負担者番号"]))
        src_keys = list(zip(src_long["患者番号"], src_long["公費負担者番号"]))
        missing_mask = pd.Series([k not in cmp_key_set for k in src_keys], index=src_long.index)
        missing_norm = src_long.loc[missing_mask].copy()

        # 元CSVから元行を拾うため、キーで結び直し（可能なら）
        # ここでは正規化値そのままを出力（どのレコードが落ちているか把握しやすい）
        missing_out = missing_norm.copy()
        if not missing_out.empty:
            missing_out.insert(0, "__キー__", missing_out["患者番号"] + "-" + missing_out["公費負担者番号"])

        # 8) 一致/不一致
        merged = src_long.merge(cmp_long, on=["患者番号", "公費負担者番号"], how="inner", suffixes=("_src", "_cmp"))
        fields = ["公費受給者番号", "公費開始日"]

        all_eq = pd.Series([True] * len(merged), index=merged.index)
        for f in fields:
            all_eq &= (merged[f + "_src"] == merged[f + "_cmp"])

        matched_rows = merged.loc[all_eq, ["患者番号", "公費負担者番号"] + [f + "_src" for f in fields]].copy()
        matched_rows.rename(columns={f + "_src": f for f in fields}, inplace=True)

        mismatches = []
        for f in fields:
            neq = merged.loc[merged[f + "_src"] != merged[f + "_cmp"], ["患者番号", "公費負担者番号", f + "_src", f + "_cmp"]].copy()
            if not neq.empty:
                neq.insert(2, "項目名", f)
                neq.rename(columns={f + "_src": "正規化_元", f + "_cmp": "正規化_突合"}, inplace=True)
                mismatches.append(neq)
        mismatch_df = pd.concat(mismatches, axis=0) if mismatches else pd.DataFrame(
            columns=["患者番号", "公費負担者番号", "項目名", "正規化_元", "正規化_突合"]
        )

        # 9) 出力
        tag = _dt.now().strftime("%Y%m%d")
        out_matched  = out_dir / f"公費_内容_一致_{tag}.csv"
        out_mismatch = out_dir / f"公費_内容_不一致_{tag}.csv"
        out_missing  = out_dir / f"公費_内容_未ヒット_{tag}.csv"

        inspection.to_csv(matched_rows, str(out_matched))
        inspection.to_csv(mismatch_df, str(out_mismatch))
        inspection.to_csv(missing_out, str(out_missing))

        self.log(
            f"[公費-内容] 一致: {len(matched_rows)} / 不一致明細行: {len(mismatch_df)} / 未ヒット: {len(missing_out)}\n"
            f"  (src={len(src_long)}, cmp={len(cmp_long)})"
        )
        messagebox.showinfo(
            "公費内容検収 完了",
            f"一致: {len(matched_rows)} 件\n不一致明細: {len(mismatch_df)} 行（項目単位）\n未ヒット: {len(missing_out)} 件\n\n出力先:\n{out_dir}"
        )
        return True


# ===== Entry points for InspectionActions =====
def run_public_content_check(app, logger=None, preset=None):
    checker = PublicContentChecker(logger=logger, preset_colmap=preset)
    return bool(checker.run(app))

def run(app, logger=None, preset=None):
    checker = PublicContentChecker(logger=logger, preset_colmap=preset)
    return bool(checker.run(app))

def main(app, logger=None, preset=None):
    checker = PublicContentChecker(logger=logger, preset_colmap=preset)
    return bool(checker.run(app))