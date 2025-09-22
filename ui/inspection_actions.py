# ui/inspection_actions.py
from __future__ import annotations
from tkinter import filedialog, messagebox
from datetime import datetime as _dt
import pandas as pd

from core.io_utils import CsvLoader
from core import inspection


class InspectionActions:
    """検収系のUIイベントを集約します。app（DataSurveyApp）に依存します。"""

    def __init__(self, app):
        self.app = app  # DataSurveyApp（_ask_inspection_colmap, _normalize_patient_number_for_match を利用）

    def _normalize_codes(self, s: pd.Series, width: int, mode: str = "zfill") -> pd.Series:
        """患者番号の正規化ヘルパ
        mode:
          - "zfill": 数字以外除去→ゼロ埋め
          - "lstrip": 数字以外除去→先頭0除去（長さ揃えない）
          - "rawdigits": 数字以外除去のみ
        """
        import re
        digits = s.astype(str).map(lambda x: re.sub(r"\D", "", x))
        if mode == "zfill":
            return digits.map(lambda x: x.zfill(width) if x else "")
        elif mode == "lstrip":
            return digits.map(lambda x: x.lstrip("0"))
        else:
            return digits

    # === 共通ユーティリティ ===
    def _ask_and_save_missing_and_matched(self, *, src: pd.DataFrame, colmap: dict,
                                          out_df: pd.DataFrame, cfg: inspection.InspectionConfig) -> None:
        """既存検収CSVを選び、未ヒット保存＆一致のみ保存（任意）を行う共通処理。"""
        from tkinter import messagebox as _mb

        if not _mb.askyesno("突合の確認", "既存の検収CSV（他システム出力等）と突合して、未ヒット患者を出力しますか？"):
            return

        cmp_path = filedialog.askopenfilename(
            title="突合対象の検収用CSV（固定カラム）を選択してください",
            filetypes=[("CSV files", "*.csv")]
        )
        if not cmp_path:
            return
        try:
            cmp_df = CsvLoader.read_csv_flex(cmp_path)
        except Exception as e:
            _mb.showerror("エラー", f"突合対象CSVの読み込みに失敗しました。\n{e}")
            return

        if "患者番号" not in cmp_df.columns:
            _mb.showerror("エラー", "突合対象CSVに『患者番号』列が見つかりません。仕様に沿ったCSVを選択してください。")
            return

        # 幅の決定（比較側優先）
        try:
            cmp_digits = cmp_df["患者番号"].astype(str).str.replace(r"\D", "", regex=True)
            width = int(cmp_digits.str.len().max()) if cmp_digits.notna().any() else cfg.patient_number_width
            if not width or width <= 0:
                width = cfg.patient_number_width
        except Exception:
            width = cfg.patient_number_width

        # out_df側の桁数も考慮（ゼロ埋め幅のズレ対策）
        try:
            if "患者番号" in out_df.columns:
                out_digits_only = out_df["患者番号"].astype(str).str.replace(r"\D", "", regex=True)
                out_max = int(out_digits_only.str.len().max()) if out_digits_only.notna().any() else 0
                width = max(width, out_max or 0) or width
        except Exception:
            pass

        # 未ヒット抽出（src -> cmp）
        missing_df = None
        src_code_col = colmap.get("患者番号")
        if src_code_col and src_code_col in src.columns:
            src_codes_norm = self._normalize_codes(src[src_code_col].astype(str), width, mode="zfill")
            cmp_codes_norm = self._normalize_codes(cmp_df["患者番号"].astype(str), width, mode="zfill")
            cmp_set = set(cmp_codes_norm.loc[cmp_codes_norm != ""])  # 空は除外
            mask_missing = (src_codes_norm != "") & (~src_codes_norm.isin(cmp_set))
            if mask_missing.any():
                missing_df = src.loc[mask_missing].copy()
                missing_df.insert(0, "__正規化患者番号__", src_codes_norm.loc[mask_missing])
                # フォールバック: 全件未ヒット/全件ヒットなど極端な結果なら先頭0無視で再判定
                try:
                    total = len(src)
                    miss_cnt = int(mask_missing.sum())
                    if miss_cnt in (0, total):
                        src_ls = self._normalize_codes(src[src_code_col].astype(str), width, mode="lstrip")
                        cmp_ls = self._normalize_codes(cmp_df["患者番号"].astype(str), width, mode="lstrip")
                        cmp_set_ls = set(cmp_ls.loc[cmp_ls != ""])  # 空は除外
                        mask2 = (src_ls != "") & (~src_ls.isin(cmp_set_ls))
                        if mask2.any() != mask_missing.any() or int(mask2.sum()) != miss_cnt:
                            missing_df = src.loc[mask2].copy()
                            missing_df.insert(0, "__正規化患者番号__", src_ls.loc[mask2])
                except Exception:
                    pass
        else:
            _mb.showwarning("注意", "患者番号のマッピングが不明です。未ヒット抽出をスキップします。")

        if missing_df is not None and not missing_df.empty:
            if _mb.askyesno("未ヒット出力", f"未ヒットの患者が {len(missing_df)} 件見つかりました。CSVとして保存しますか？"):
                default_name = f"未ヒット患者_{_dt.now().strftime('%Y%m%d')}.csv"
                miss_path = filedialog.asksaveasfilename(
                    title="未ヒット患者リストを保存",
                    defaultextension=".csv",
                    initialfile=default_name,
                    filetypes=[("CSV files", "*.csv")]
                )
                if miss_path:
                    with open(miss_path, "w", encoding="cp932", errors="replace", newline="") as f:
                        missing_df.to_csv(f, index=False)
                    _mb.showinfo("未ヒット出力", f"未ヒット {len(missing_df)} 件を保存しました。\n{miss_path}")
        else:
            _mb.showinfo("結果", "未ヒットの患者は見つかりませんでした。")

        # 検収CSVを一致のみで別名保存
        try:
            if "患者番号" in out_df.columns:
                out_codes_norm = self._normalize_codes(out_df["患者番号"].astype(str), width, mode="zfill")
                # cmp_set が上で定義されていない可能性があるため再作成
                cmp_codes_norm = self._normalize_codes(cmp_df["患者番号"].astype(str), width, mode="zfill")
                cmp_set = set(cmp_codes_norm.loc[cmp_codes_norm != ""])  # 空は除外
                matched_mask = (out_codes_norm != "") & (out_codes_norm.isin(cmp_set))
                filtered_out_df = out_df.loc[matched_mask].copy()
                # フォールバック: zfill一致が0件の場合、先頭0無視で再判定
                if filtered_out_df.empty:
                    out_lstrip = self._normalize_codes(out_df["患者番号"].astype(str), width, mode="lstrip")
                    cmp_lstrip = self._normalize_codes(cmp_df["患者番号"].astype(str), width, mode="lstrip")
                    cmp_set2 = set(cmp_lstrip.loc[cmp_lstrip != ""])  # 空は除外
                    matched2 = (out_lstrip != "") & (out_lstrip.isin(cmp_set2))
                    if matched2.any():
                        filtered_out_df = out_df.loc[matched2].copy()
                if _mb.askyesno(
                    "検収CSVの絞り込み",
                    f"検収CSVを『一致のみ』({len(filtered_out_df)}行)に絞って別名保存しますか？\n（元の検収CSVはそのまま残ります）",
                ):
                    default_name2 = f"検収_一致のみ_{_dt.now().strftime('%Y%m%d')}.csv"
                    filtered_path = filedialog.asksaveasfilename(
                        title="検収CSV（一致のみ）を保存",
                        defaultextension=".csv",
                        initialfile=default_name2,
                        filetypes=[("CSV files", "*.csv")]
                    )
                    if filtered_path:
                        inspection.to_csv(filtered_out_df, filtered_path)
                        _mb.showinfo("保存", f"一致のみの検収CSVを保存しました。\n{filtered_path}")
        except Exception as fe:
            _mb.showwarning("絞り込み保存エラー", f"一致のみの検収CSV保存に失敗しました。\n{fe}")

    # === 各アクション ===
    def run_patient(self):
        in_path = filedialog.askopenfilename(title="患者情報CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        if not in_path:
            return
        try:
            src = CsvLoader.read_csv_flex(in_path)
            colmap = self.app._ask_inspection_colmap(src, required_cols=list(inspection.COLUMNS_PATIENT))
            if colmap is None:
                return
            cfg = inspection.InspectionConfig(patient_number_width=10)
            out_df = inspection.build_inspection_df(src, colmap, cfg, target_columns=inspection.COLUMNS_PATIENT)

            out_path = filedialog.asksaveasfilename(title="検収CSVを保存", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not out_path:
                return
            inspection.to_csv(out_df, out_path)

            # 突合（任意）
            self._ask_and_save_missing_and_matched(src=src, colmap=colmap, out_df=out_df, cfg=cfg)

            messagebox.showinfo("完了", f"検収CSVを保存しました:\n{out_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"検収処理中に失敗しました。\n{e}")

    def run_insurance(self):
        in_path = filedialog.askopenfilename(title="保険情報の入力CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        if not in_path:
            return
        try:
            src = CsvLoader.read_csv_flex(in_path)
            colmap = self.app._ask_inspection_colmap(src, required_cols=list(inspection.COLUMNS_INSURANCE))
            if colmap is None:
                return
            cfg = inspection.InspectionConfig(patient_number_width=10)
            out_df = inspection.build_inspection_df(src, colmap, cfg, target_columns=inspection.COLUMNS_INSURANCE)

            out_path = filedialog.asksaveasfilename(title="保険情報 検収CSVを保存", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not out_path:
                return
            inspection.to_csv(out_df, out_path)

            # 突合（任意）
            self._ask_and_save_missing_and_matched(src=src, colmap=colmap, out_df=out_df, cfg=cfg)

            messagebox.showinfo("完了", f"保険情報の検収CSVを保存しました:\n{out_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"保険情報の検収処理に失敗しました。\n{e}")

    def run_public(self):
        in_path = filedialog.askopenfilename(title="公費情報の入力CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        if not in_path:
            return
        try:
            src = CsvLoader.read_csv_flex(in_path)
            colmap = self.app._ask_inspection_colmap(src, required_cols=list(inspection.COLUMNS_PUBLIC))
            if colmap is None:
                return
            cfg = inspection.InspectionConfig(patient_number_width=10)
            out_df = inspection.build_inspection_df(src, colmap, cfg, target_columns=inspection.COLUMNS_PUBLIC)

            out_path = filedialog.asksaveasfilename(title="公費情報 検収CSVを保存", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not out_path:
                return
            inspection.to_csv(out_df, out_path)

            # 突合（任意）
            self._ask_and_save_missing_and_matched(src=src, colmap=colmap, out_df=out_df, cfg=cfg)

            messagebox.showinfo("完了", f"公費情報の検収CSVを保存しました:\n{out_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"公費情報の検収処理に失敗しました。\n{e}")

    def run_missing(self):
        # 単発の未ヒット抽出（元CSV vs 検収CSV）
        src_path = filedialog.askopenfilename(title="検収元（無加工）CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        if not src_path:
            return
        try:
            src_df = CsvLoader.read_csv_flex(src_path)
        except Exception as e:
            messagebox.showerror("エラー", f"検収元CSVの読み込みに失敗しました。\n{e}")
            return

        # 患者番号列を選択
        cols = list(src_df.columns)
        from .dialogs import ColumnSelectDialog
        dlg = ColumnSelectDialog(self.app, cols, title=f"[{src_path.split('/')[-1]}] 検収元の患者番号列を選択")
        src_code_col = dlg.selected if hasattr(dlg, "selected") and dlg.selected else None
        if not src_code_col:
            messagebox.showwarning("中止", "患者番号列が選択されませんでした。")
            return

        insp_path = filedialog.askopenfilename(title="検収用CSV（固定カラム）を選択してください", filetypes=[("CSV files", "*.csv")])
        if not insp_path:
            return
        try:
            insp_df = CsvLoader.read_csv_flex(insp_path)
        except Exception as e:
            messagebox.showerror("エラー", f"検収用CSVの読み込みに失敗しました。\n{e}")
            return
        if "患者番号" not in insp_df.columns:
            messagebox.showerror("エラー", "検収用CSVに『患者番号』列が見つかりません。仕様に沿ったCSVを選択してください。")
            return

        # 幅の決定
        insp_digits = insp_df["患者番号"].astype(str).str.replace(r"\D", "", regex=True)
        try:
            width = int(insp_digits.str.len().max()) if insp_digits.notna().any() else 10
            if not width or width <= 0:
                width = 10
        except Exception:
            width = 10

        # 正規化して集合化
        src_codes_norm = self.app._normalize_patient_number_for_match(src_df[src_code_col], width)
        insp_codes_norm = self.app._normalize_patient_number_for_match(insp_df["患者番号"], width)
        insp_set = set(insp_codes_norm.loc[insp_codes_norm != ""].tolist())

        mask_missing = (src_codes_norm != "") & (~src_codes_norm.isin(insp_set))
        missing_df = src_df.loc[mask_missing].copy()
        missing_df.insert(0, "__正規化患者番号__", src_codes_norm.loc[mask_missing])

        if missing_df.empty:
            messagebox.showinfo("結果", "未ヒットの患者は見つかりませんでした。")
            return

        default_name = f"未ヒット患者_{_dt.now().strftime('%Y%m%d')}.csv"
        out_path = filedialog.asksaveasfilename(title="未ヒット患者リストを保存", defaultextension=".csv",
                                                initialfile=default_name, filetypes=[("CSV files", "*.csv")])
        if not out_path:
            return
        try:
            with open(out_path, "w", encoding="cp932", errors="replace", newline="") as f:
                missing_df.to_csv(f, index=False)
            messagebox.showinfo("完了", f"未ヒット {len(missing_df)} 件を出力しました。\n{out_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"未ヒットリストの保存に失敗しました。\n{e}")