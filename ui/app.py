# ui/app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from pathlib import Path

from core.io_utils import CsvLoader
from core.file_keys import (
    FILE_KEYS, PATIENT_CODE_CANDIDATES, SAMPLE_SIZE,
    COL_KUBUN, COL_LOCAL_PUB_UUID
)
from core.sampling import (
    normalize_patient_code_series, sample_unique_by_col, sample_by_patient_code
)
from .dialogs import ColumnSelectDialog
from .tables import DualTablesView

class DataSurveyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("データ移行後調査ツール")
        self.geometry("1100x700")
        self.resizable(True, True)
        self._build_widgets()

    def _build_widgets(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(
            top,
            text="4つのCSVを読み込み：各ファイルで抽出5件（特殊条件）/ 下段に他ファイル抽出コード一致レコードを表示。"
        ).pack(side="left")

        ttk.Button(top, text="CSVを選択（4ファイル）", command=self.load_and_display).pack(side="right")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=(0,10))

    def load_and_display(self):
        file_paths = filedialog.askopenfilenames(
            title="CSVファイルを4つ選択してください",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_paths:
            return
        if len(file_paths) != 4:
            messagebox.showerror("エラー", "必ず4つのCSVファイルを選択してください。")
            return

        for child in self.notebook.winfo_children():
            child.destroy()

        # 1) ロード & 患者コード列確定
        files_info = []
        for path in file_paths:
            name = Path(path).name
            try:
                df = CsvLoader.read_csv_flex(path)
            except Exception as e:
                tab = ttk.Frame(self.notebook)
                self.notebook.add(tab, text=name)
                ttk.Label(tab, text=f"読み込みエラー:\n{e}").pack(anchor="w", padx=10, pady=10)
                files_info.append({
                    "key": "unknown", "name": name,
                    "df": pd.DataFrame(), "code_col": None, "sample_df": pd.DataFrame()
                })
                continue

            # 患者コード列 推定 or ダイアログ
            code_col = None
            for cand in PATIENT_CODE_CANDIDATES:
                if cand in df.columns:
                    code_col = cand
                    break
            if code_col is None:
                cols = list(df.columns)
                dlg = ColumnSelectDialog(self, cols, title=f"[{name}] 患者コード列を選択")
                code_col = dlg.selected if dlg.selected else None

            files_info.append({
                "key": self._guess_file_key(name),
                "name": name,
                "df": df,
                "code_col": code_col,
                "sample_df": pd.DataFrame(),
            })

        # 2) サンプリング（特殊条件）
        for info in files_info:
            df = info["df"]; code_col = info["code_col"]; key = info["key"]
            if df.empty or not code_col:
                info["sample_df"] = pd.DataFrame(); continue

            df2 = df.copy()
            df2[code_col] = normalize_patient_code_series(df2[code_col])
            df2 = df2[df2[code_col] != ""]

            if df2.empty:
                info["sample_df"] = pd.DataFrame(); continue

            if key == "ceiling":
                sample_df = sample_unique_by_col(df2, COL_KUBUN, SAMPLE_SIZE)
            elif key == "subsidies":
                sample_df = sample_unique_by_col(df2, COL_LOCAL_PUB_UUID, SAMPLE_SIZE)
            else:
                sample_df = sample_by_patient_code(df2, code_col, SAMPLE_SIZE)

            info["sample_df"] = sample_df
            
        order_map = {
            "patients": 0,
            "health_ins": 1,
            "subsidies": 2,
            "ceiling": 3,
            "unknown": 99
        }
        files_info_sorted = sorted(files_info, key=lambda x: order_map.get(x["key"], 99))

        # 3) タブ作成（上下2段）
        for idx, info in enumerate(files_info_sorted):
            df = info["df"]; code_col = info["code_col"]; name = info["name"]; sample_df = info["sample_df"]
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=f"{idx+1}: {name}")

            if df.empty or not code_col:
                ttk.Label(tab, text="このファイルは読み込み/列選択に失敗しました。").pack(anchor="w", padx=10, pady=10)
                continue

            # 他ファイル抽出コード集合
            other_codes = set()
            for other in files_info_sorted:
                if other is info:
                    continue
                cc = other.get("code_col")
                sdf = other.get("sample_df", pd.DataFrame())
                if cc and not sdf.empty:
                    other_codes |= set(normalize_patient_code_series(sdf[cc]).tolist())

            top_df = sample_df.copy()
            df_work = df.copy()
            df_work[code_col] = normalize_patient_code_series(df_work[code_col])
            bottom_df = df_work[df_work[code_col].isin(other_codes)]
            if not top_df.empty:
                bottom_df = bottom_df.drop(index=bottom_df.index.intersection(top_df.index), errors="ignore")

            DualTablesView.build_dual_tables(tab, top_df, bottom_df)
            btn_frame = ttk.Frame(tab)
            btn_frame.pack(fill="x", padx=12, pady=(0,10))
            ttk.Button(
                btn_frame,
                text="上段の患者コードをコピー",
                command=lambda df=top_df, cc=code_col: self._copy_top_codes(df, cc),
            ).pack(side="left")

            status = ttk.Label(tab, text=(
                f"患者コード列: {code_col} / 抽出: {len(top_df)}件 / "
                f"他ファイル抽出コード一致: {len(bottom_df)}件"
            ))
            status.pack(anchor="w", padx=12, pady=(0,10))

    @staticmethod
    def _guess_file_key(fname: str) -> str:
        low = fname.lower()
        items = sorted(FILE_KEYS.items(), key=lambda kv: -len(kv[1]))
        for k, token in items:
            if token in low:
                return k
        return "unknown"
    
    def _copy_top_codes(self, top_df: pd.DataFrame, code_col: str):
        if top_df is None or top_df.empty or not code_col or code_col not in top_df.columns:
            messagebox.showwarning("コピー", "コピー対象がありません。")
            return
        # 正規化して重複・空を排除（念のため）
        codes = (
            top_df[code_col]
            .dropna()
            .map(lambda x: str(x).strip())
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )
        if not codes:
            messagebox.showwarning("コピー", "コピー対象がありません。")
            return
        # クリップボードへ
        try:
            self.clipboard_clear()
            self.clipboard_append("\n".join(codes))
            self.update()  # クリップボード反映
            messagebox.showinfo("コピー", f"{len(codes)}件の患者コードをコピーしました。")
        except Exception as e:
            messagebox.showerror("コピー失敗", f"クリップボードへのコピーに失敗しました。\n{e}")