# ui/insurance_content_check.py
from __future__ import annotations
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd
from datetime import datetime as _dt
import threading

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

    def _ui_log(self, app, msg: str):
        # Safely log from background thread by scheduling on main thread
        app.after(0, lambda: self.log(msg))

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

    def _split_symbol_number(self, s: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        記号番号が1列に混在している場合に、
        ・記号=先頭の非数字（空白は除去、全角→半角）
        ・番号=末尾の数字列（全角→半角）
        へ自動分割して返す（Series2本）。
        """
        import unicodedata, re
        t = s.astype(str).map(lambda x: unicodedata.normalize("NFKC", x).replace(" ", "").replace("　", ""))
        sym = t.map(lambda x: re.match(r"^([^\d]+)", x).group(1) if re.match(r"^([^\d]+)", x) else "")
        num = t.map(lambda x: re.search(r"(\d+)$", x).group(1) if re.search(r"(\d+)$", x) else "")
        return sym, num

    def _make_key_series(self, pat: pd.Series, payer: pd.Series) -> pd.Series:
        """患者番号と保険者番号から突合キー（'患者番号|保険者番号'）を高速生成。
        - 事前に数字正規化済み（ゼロ埋め/桁合わせ後）の Series を渡す想定。
        - NaN を空文字に、型は object のまま連結（pandas のベクトル化で高速）。
        """
        a = pat.fillna("").astype(str)
        b = payer.fillna("").astype(str)
        # 文字列連結（ベクトル化）。`+` は Cython 実装で高速。
        return a + "|" + b

    # ---- Main ----
    def _process(self, app, src: pd.DataFrame, cmp_df: pd.DataFrame, colmap: dict, out_dir: Path, width: int, mig: str | None):
        """重い本処理。バックグラウンドスレッドで実行し、UI更新は after で行う。
        速度最適化点:
          - 早期の列絞り込み
          - 完全重複の削除（キー+比較列）
          - set/Index ベースの anti-join（未ヒット）
          - join( index ) による内部結合（一致/不一致判定）
        """
        try:
            import gc

            # 0) 必要列に絞り込み（突合側）
            needed = ["患者番号","保険者番号","患者負担割合","保険開始日","保険終了日","保険証記号","保険証番号"]
            cmp_df = cmp_df[[c for c in needed if c in cmp_df.columns]].copy()

            # 1) 元CSV 正規化
            self._ui_log(app, "[保険-内容] 正規化(元CSV) 開始")
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
            self._ui_log(app, "[保険-内容] 正規化(元CSV) 完了")

            # 2) 突合側 正規化
            self._ui_log(app, "[保険-内容] 正規化(突合CSV) 開始")
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
            self._ui_log(app, "[保険-内容] 正規化(突合CSV) 完了")

            # 3) 完全重複の除去（キー＋比較列が全て同一の行のみ）
            dup_subset = ["患者番号", "保険者番号", "患者負担割合", "保険開始日", "保険終了日", "保険証記号", "保険証番号"]
            src_norm = src_norm.drop_duplicates(subset=dup_subset, keep="first")
            cmp_norm = cmp_norm.drop_duplicates(subset=dup_subset, keep="first")
            self._ui_log(app, f"[保険-内容] 完全重複の除去 完了: src={len(src_norm)} cmp={len(cmp_norm)}")

            # 4) キー生成（ベクトル化文字列連結）
            src_norm = src_norm[(src_norm["患者番号"] != "") & (src_norm["保険者番号"] != "")].copy()
            cmp_norm = cmp_norm[(cmp_norm["患者番号"] != "") & (cmp_norm["保険者番号"] != "")].copy()
            src_norm["__key__"] = self._make_key_series(src_norm["患者番号"], src_norm["保険者番号"])
            cmp_norm["__key__"] = self._make_key_series(cmp_norm["患者番号"], cmp_norm["保険者番号"])

            # 5) 未ヒット（anti-join）: set で membership を取り、高速化
            self._ui_log(app, "[保険-内容] 未ヒット(anti-join) 算出中")
            cmp_key_set = set(cmp_norm["__key__"].tolist())
            missing_mask = ~src_norm["__key__"].isin(cmp_key_set)
            # 元の行を抽出（インデックス対応のために src の直値を参照）
            missing_df = src.loc[src_norm.index[missing_mask]].copy()
            if not missing_df.empty:
                missing_df.insert(0, "__正規化保険者番号__", src_norm.loc[missing_df.index, "保険者番号"].values)
                missing_df.insert(0, "__正規化患者番号__", src_norm.loc[missing_df.index, "患者番号"].values)
            self._ui_log(app, f"[保険-内容] 未ヒット(anti-join) 完了: {len(missing_df)}件")

            # 6) 一致/不一致：Index join による内部結合
            self._ui_log(app, "[保険-内容] 一致/不一致 join 中")
            src_idx = src_norm.set_index("__key__", drop=False)
            cmp_idx = cmp_norm.set_index("__key__", drop=False)
            merged = src_idx.join(cmp_idx, how="inner", lsuffix="_src", rsuffix="_cmp", sort=False)
            self._ui_log(app, f"[保険-内容] 一致/不一致 join 完了: merged={len(merged)}行")

            fields = ["患者負担割合", "保険開始日", "保険終了日", "保険証記号", "保険証番号"]
            # 完全一致判定
            all_eq_mask = pd.Series(True, index=merged.index)
            for f in fields:
                all_eq_mask &= (merged[f + "_src"] == merged[f + "_cmp"])

            # 一致行の抽出
            matched_cols = ["患者番号_src", "保険者番号_src"] + [f + "_src" for f in fields]
            matched_rows = merged.loc[all_eq_mask, matched_cols].copy()
            matched_rows.rename(columns={
                "患者番号_src": "患者番号",
                "保険者番号_src": "保険者番号",
                **{f + "_src": f for f in fields}
            }, inplace=True)

            # 不一致明細
            mismatches = []
            for f in fields:
                neq = merged.loc[merged[f + "_src"] != merged[f + "_cmp"], ["患者番号_src", "保険者番号_src", f + "_src", f + "_cmp"]].copy()
                if not neq.empty:
                    neq.insert(2, "項目名", f)
                    neq.rename(columns={
                        "患者番号_src": "患者番号",
                        "保険者番号_src": "保険者番号",
                        f + "_src": "正規化_元",
                        f + "_cmp": "正規化_突合",
                    }, inplace=True)
                    mismatches.append(neq)
            mismatch_df = pd.concat(mismatches, axis=0) if mismatches else pd.DataFrame(
                columns=["患者番号", "保険者番号", "項目名", "正規化_元", "正規化_突合"]
            )

            # 7) 出力
            self._ui_log(app, "[保険-内容] 出力中")
            tag = _dt.now().strftime("%Y%m%d")
            out_matched  = out_dir / f"保険_内容_一致_{tag}.csv"
            out_mismatch = out_dir / f"保険_内容_不一致_{tag}.csv"
            out_missing  = out_dir / f"保険_内容_未ヒット_{tag}.csv"
            inspection.to_csv(matched_rows, str(out_matched))
            inspection.to_csv(mismatch_df, str(out_mismatch))
            inspection.to_csv(missing_df, str(out_missing))
            self._ui_log(app, "[保険-内容] 出力完了")

            # 8) メモリ回収（次タスクが続く場合の安定性向上）
            del cmp_key_set, src_idx, cmp_idx, merged
            gc.collect()

            def _notify_ok():
                self.log(
                    f"[保険-内容] 一致: {len(matched_rows)} / 不一致明細行: {len(mismatch_df)} / 未ヒット: {len(missing_df)}\n"
                    f"  (src={len(src_norm)}, cmp={len(cmp_norm)})"
                )
                messagebox.showinfo(
                    "保険内容検収 完了",
                    f"一致: {len(matched_rows)} 件\n不一致明細: {len(mismatch_df)} 行（項目単位）\n未ヒット: {len(missing_df)} 件\n\n出力先:\n{out_dir}"
                )
            app.after(0, _notify_ok)
        except Exception as e:
            def _notify_err():
                messagebox.showerror("エラー", f"保険内容検収でエラーが発生しました。\n{e}")
            app.after(0, _notify_err)

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

        # --- 記号+番号 混在列の自動分割 ---
        sym_key = "保険証記号"
        num_key = "保険証番号"
        src_sym_col = colmap.get(sym_key)
        src_num_col = colmap.get(num_key)
        try:
            if (src_sym_col or src_num_col):
                base_col = src_sym_col or src_num_col
                # 片方未指定 or 同一列を参照している場合 → base_col を分割
                if base_col and base_col in src.columns and (src_sym_col == src_num_col or (src_sym_col is None or src_num_col is None)):
                    sym_s, num_s = self._split_symbol_number(src[base_col])
                    tmp_sym = "__split_保険証記号__"
                    tmp_num = "__split_保険証番号__"
                    src[tmp_sym] = sym_s
                    src[tmp_num] = num_s
                    colmap[sym_key] = tmp_sym
                    colmap[num_key] = tmp_num
                    self.log("[保険-内容] 記号+番号混在列を自動分割して使用しました")
        except Exception:
            # 分割失敗時は素通し
            pass

        # 3) 突合CSV（検収用）
        cmp_path = filedialog.askopenfilename(title="突合用（検収CSV/他システム出力）を選択", filetypes=[("CSV files", "*.csv")])
        self.log(f"[保険-内容] 突合CSV: {cmp_path}")
        if not cmp_path:
            return
        cmp_df = CsvLoader.read_csv_flex(cmp_path)
        needed = ["患者番号", "保険者番号", "患者負担割合", "保険開始日", "保険終了日", "保険証記号", "保険証番号"]
        for need in needed:
            if need not in cmp_df.columns:
                messagebox.showerror("エラー", f"突合CSVに『{need}』列がありません。")
                return
        # Immediately reduce cmp_df to necessary columns before passing to worker thread
        cmp_df = cmp_df[needed].copy()

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

        # バックグラウンドで突合・出力を実行
        self.log("[保険-内容] 突合処理を開始します（バックグラウンド実行）...")
        t = threading.Thread(
            target=self._process,
            args=(app, src, cmp_df, colmap, out_dir, width, mig),
            daemon=True
        )
        t.start()
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