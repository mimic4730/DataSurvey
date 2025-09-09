# ui/app.py
import threading
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
from .loading import LoadingOverlay


class DataSurveyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DataSurvey - 患者コードサンプル抽出（行全体 & クロス表示）")
        self.geometry("1100x700")
        self.resizable(True, True)
        self._build_widgets()
        self._loader = None  # LoadingOverlay
        self._bg_thread = None

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

    # === 追加：上段患者コードコピー ===
    def _copy_top_codes(self, top_df: pd.DataFrame, code_col: str):
        if top_df is None or top_df.empty or not code_col or code_col not in top_df.columns:
            messagebox.showwarning("コピー", "コピー対象がありません。")
            return
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
        try:
            self.clipboard_clear()
            self.clipboard_append("\n".join(codes))
            self.update()
            messagebox.showinfo("コピー", f"{len(codes)}件の患者コードをコピーしました。")
        except Exception as e:
            messagebox.showerror("コピー失敗", f"クリップボードへのコピーに失敗しました。\n{e}")

    # === 読み込みトリガ ===
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

        # 既存タブをクリア
        for child in self.notebook.winfo_children():
            child.destroy()

        # ローディング表示
        self._loader = LoadingOverlay(self, "読み込み中...")
        self._loader.show()

        # バックグラウンドで読み込み＆サンプリング
        self._bg_thread = threading.Thread(
            target=self._prepare_and_build_async, args=(file_paths,), daemon=True
        )
        self._bg_thread.start()

    # === バックグラウンド本体（IOと整形のみ） ===
    def _prepare_and_build_async(self, file_paths):
        try:
            files_info = self._prepare_files_info(file_paths)
            # 並び順固定
            order_map = {"patients": 0, "health_ins": 1, "subsidies": 2, "ceiling": 3, "unknown": 99}
            files_info_sorted = sorted(files_info, key=lambda x: order_map.get(x["key"], 99))
            # UIスレッドでタブ作成
            self.after(0, lambda: self._build_tabs(files_info_sorted))
        except Exception as e:
            self.after(0, lambda: self._show_async_error(e))
        finally:
            self.after(0, self._close_loader)

    def _close_loader(self):
        if self._loader:
            self._loader.close()
            self._loader = None

    def _show_async_error(self, e: Exception):
        messagebox.showerror("エラー", f"読み込み処理でエラーが発生しました。\n{e}")

    # === データ準備：ファイル読み込み→患者コード列確定→サンプリング ===
    def _prepare_files_info(self, file_paths):
        files_info = []
        for path in file_paths:
            name = Path(path).name
            try:
                df = CsvLoader.read_csv_flex(path)
            except Exception as e:
                files_info.append({
                    "key": "unknown", "name": name,
                    "df": pd.DataFrame(), "code_col": None, "sample_df": pd.DataFrame(),
                    "error": f"読み込みエラー: {e}"
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
                # ダイアログはメインスレッドで開く必要があるため after で同期的に取得
                code_col = self._ask_code_col_sync(name, cols)

            files_info.append({
                "key": self._guess_file_key(name),
                "name": name,
                "df": df,
                "code_col": code_col,
                "sample_df": pd.DataFrame(),
            })

        # サンプリング
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

        return files_info

    # === メインスレッドで列ダイアログを出すヘルパ ===
    def _ask_code_col_sync(self, filename: str, columns: list[str]) -> str | None:
        result = {"col": None}
        def _ask():
            dlg = ColumnSelectDialog(self, columns, title=f"[{filename}] 患者コード列を選択")
            result["col"] = dlg.selected if dlg.selected else None
        # after で呼び出し、完了まで待機
        done = threading.Event()
        self.after(0, lambda: (_ask(), done.set()))
        done.wait()
        return result["col"]

    # === タブ組み立て（UIスレッド） ===
    def _build_tabs(self, files_info_sorted):
        for idx, info in enumerate(files_info_sorted):
            df = info["df"]; code_col = info["code_col"]; name = info["name"]; sample_df = info["sample_df"]

            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=f"{idx+1}: {name}")

            if "error" in info:
                ttk.Label(tab, text=info["error"]).pack(anchor="w", padx=10, pady=10)
                continue

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

            # コピー操作ボタン（上段の患者コード5件をコピー）
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