# ui/app.py
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from pathlib import Path
from datetime import datetime

from core.io_utils import CsvLoader
from core.file_keys import (
    FILE_KEYS, PATIENT_CODE_CANDIDATES, SAMPLE_SIZE,
    COL_KUBUN, COL_LOCAL_PUB_UUID
)
from core.sampling import (
    normalize_patient_code_series, sample_unique_by_col, sample_by_patient_code
)
from core.naming import find_patient_code_column
from core import inspection
from .dialogs import ColumnSelectDialog
from .tables import DualTablesView
from .loading import LoadingOverlay
from .inspection_actions import InspectionActions

class DataSurveyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DataSurvey - 患者コードサンプル抽出（行全体 & クロス表示）")
        self.geometry("1300x700")
        self.resizable(True, True)
        self.actions = InspectionActions(self)
        self._build_widgets()
        self._loader = None  # LoadingOverlay
        self._bg_thread = None
        self._build_menu()

    def _build_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        inspection_menu = tk.Menu(menubar, tearoff=0)
        inspection_menu.add_command(label="患者情報検収CSVを生成", command=self.actions.run_patient)
        inspection_menu.add_command(label="保険情報検収CSVを生成", command=self.actions.run_insurance)
        inspection_menu.add_command(label="公費情報検収CSVを生成", command=self.actions.run_public)
        inspection_menu.add_separator()
        inspection_menu.add_command(label="未ヒット患者（検収元→検収用）をCSV出力", command=self.actions.run_missing)
        menubar.add_cascade(label="検収", menu=inspection_menu)

    def _build_widgets(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(
            top,
            text="4つのCSVを読み込み：各ファイルで抽出5件（特殊条件）/ 下段に他ファイル抽出コード一致レコードを表示。"
        ).pack(side="left")

        # 右側のボタン群
        right_btns = ttk.Frame(top)
        right_btns.pack(side="right")
        ttk.Button(right_btns, text="CSVを選択（4ファイル）", command=self.load_and_display).pack(side="left", padx=(0,6))
        ttk.Button(right_btns, text="検収CSV生成(患者)", command=self.actions.run_patient).pack(side="left", padx=(0,6))
        ttk.Button(right_btns, text="検収CSV生成(保険)", command=self.actions.run_insurance).pack(side="left", padx=(0,6))
        ttk.Button(right_btns, text="検収CSV生成(公費)", command=self.actions.run_public).pack(side="left")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=(0,10))

    def _ask_inspection_colmap(self, src_df: pd.DataFrame, required_cols: list[str] | None = None):
        """
        検収用の固定カラム（inspection.INSPECTION_COLUMNS）に対して、
        入力CSV(src_df)のカラムをユーザーに対応付けさせるモーダルダイアログ。
        戻り値: {固定カラム名: 入力側カラム名 or None} / キャンセル時は None
        """
        if required_cols is None:
            required_cols = list(inspection.INSPECTION_COLUMNS)
        cols = list(src_df.columns)
        # 表示用先頭に空欄を追加
        choices = ["(空欄)"] + cols

        win = tk.Toplevel(self)
        win.title("検収カラムの対応付け")
        win.transient(self)
        win.grab_set()
        win.resizable(False, True)

        container = ttk.Frame(win)
        container.pack(expand=True, fill="both")

        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        inner = ttk.Frame(canvas, padding=12)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_configure)

        def _on_resize(event):
            canvas.itemconfigure(inner_id, width=event.width)
        canvas.bind("<Configure>", _on_resize)

        canvas.pack(side="left", expand=True, fill="both")
        vscroll.pack(side="right", fill="y")

        ttk.Label(inner, text="検収で出力する各カラムに、元CSVのカラムを対応付けてください。").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,8))


        # コンボボックスを並べる
        widgets = {}
        for i, out_col in enumerate(required_cols, start=1):
            ttk.Label(inner, text=out_col, width=22).grid(row=i, column=0, sticky="e", padx=(0,8), pady=2)
            cb = ttk.Combobox(inner, values=choices, state="readonly", width=40)
            # 既定値：同名列があれば自動選択
            if out_col in cols:
                cb.set(out_col)
            else:
                cb.set("(空欄)")
            cb.grid(row=i, column=1, sticky="w", pady=2)
            widgets[out_col] = cb

        # ボタン行
        btn_frame = ttk.Frame(inner)
        btn_frame.grid(row=len(required_cols)+1, column=0, columnspan=2, pady=(12,0))

        result = {"map": None}

        def on_ok():
            mapping = {}
            for out_col in required_cols:
                cb = widgets[out_col]
                val = cb.get()
                mapping[out_col] = None if val == "(空欄)" else val
            result["map"] = mapping
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()

        def on_cancel():
            result["map"] = None
            try:
                win.grab_release()
            except Exception:
                pass
            win.destroy()

        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="キャンセル", command=on_cancel).pack(side="left", padx=6)
        # Protocol handler for window close (✕ button)
        win.protocol("WM_DELETE_WINDOW", on_cancel)

        # 位置調整（親の中央）
        win.update_idletasks()
        px, py = self.winfo_rootx(), self.winfo_rooty()
        pw, ph = self.winfo_width(), self.winfo_height()
        w, h = 640, min(600, 90 + 28 * len(required_cols))
        x, y = px + (pw - w)//2, py + (ph - h)//2
        win.geometry(f"{w}x{h}+{x}+{y}")

        win.wait_window()
        return result["map"]

    def _normalize_patient_number_for_match(self, s: pd.Series, width: int) -> pd.Series:
        """数字以外を除去し、指定桁で0埋め。空なら空欄のまま。"""
        def _norm(x):
            import re
            digits = re.sub(r"\D", "", str(x) if x is not None else "")
            return digits.zfill(width) if digits else ""
        return s.map(_norm)


    # === 上段患者コードコピー ===
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

            # ▼▼ 患者コード列 推定（順序：厳密一致 → 正規化一致 → ダイアログ） ▼▼
            code_col = None

            # (1) 厳密一致（候補にそのまま一致）
            for cand in PATIENT_CODE_CANDIDATES:
                if cand in df.columns:
                    code_col = cand
                    break

            # (2) 正規化マッチ（モジバケ・揺れ対応）
            if code_col is None:
                code_col = find_patient_code_column(df.columns)

            # (3) ダイアログ（最後の手段）
            if code_col is None:
                cols = list(df.columns)
                code_col = self._ask_code_col_sync(name, cols)

            files_info.append({
                "key": self._guess_file_key(name),
                "name": name,
                "df": df,
                "code_col": code_col,
                "sample_df": pd.DataFrame(),
            })

        # （以下サンプリング処理はそのまま）
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