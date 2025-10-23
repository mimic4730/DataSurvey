# ui/inspection_actions.py
from __future__ import annotations
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
class _ModalProgress(tk.Toplevel):
    def __init__(self, parent, title="処理中", message="処理中です…しばらくお待ちください"):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()  # modal
        self.protocol("WM_DELETE_WINDOW", lambda: None)  # 閉じる無効
        self._label = ttk.Label(self, text=message, padding=(16, 12))
        self._label.pack(anchor="w")
        self._bar = ttk.Progressbar(self, mode="indeterminate", length=320)
        self._bar.pack(fill="x", padx=16, pady=(0, 16))
        try:
            self._bar.start(12)
        except Exception:
            pass
        self.update_idletasks()
        # 画面中央へ
        try:
            self.update_idletasks()
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            w = 360; h = 100
            x = px + (pw - w)//2
            y = py + (ph - h)//2
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    def set_message(self, message: str):
        try:
            self._label.configure(text=message)
            self.update_idletasks()
        except Exception:
            pass

    def close(self):
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

from datetime import datetime as _dt
import pandas as pd
from pathlib import Path
import json, os

from core.io_utils import CsvLoader
from core import inspection


class InspectionActions:
    """検収系のUIイベントを集約します。app（DataSurveyApp）に依存します。"""
    COLMAP_FILE = Path.home() / ".datasurvey" / "colmaps.json"

    def __init__(self, app):
        self.app = app  # DataSurveyApp（_ask_inspection_colmap, _normalize_patient_number_for_match を利用）
        self.public_migration_yyyymmdd: str | None = None
        self.insurance_migration_yyyymmdd: str | None = None
        self._logger = None
        self._migration_provider = None  # callable that returns raw user input for migration date
        self._migration_yyyymmdd: str | None = None  # cached normalized yyyymmdd (shared for 保険/公費)

    # プリセット保存・ロード
    def _load_colmaps(self) -> dict:
        try:
            if self.COLMAP_FILE.exists():
                with open(self.COLMAP_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_colmaps(self, data: dict) -> None:
        try:
            self.COLMAP_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.COLMAP_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ▼ 追加: app から受け取ったロガーを保持
    def set_logger(self, logger_callable):
        self._logger = logger_callable

    # ▼ 追加: 検収ページの「データ移行日」入力欄から値を取得するためのプロバイダを登録
    def set_migration_provider(self, provider_callable):
        """
        provider_callable: 呼び出し時に文字列を返す関数（例: lambda: entry.get()）
        """
        self._migration_provider = provider_callable

    # ▼ 追加: コード側から直接移行日を更新したい場合（手動設定用）
    def set_migration_date(self, raw_value: str | None):
        """
        raw_value: 'YYYYMMDD' や '2024/07/10', 'R6.7.10' のような表記を許容。
        正しく解釈できた場合は共通キャッシュに保存し、保険/公費の個別値にも反映する。
        """
        if not raw_value:
            return
        yyyymmdd = inspection._parse_date_any_to_yyyymmdd(str(raw_value))
        if yyyymmdd:
            self._migration_yyyymmdd = yyyymmdd
            # 後方互換（既存プロパティにも反映）
            self.insurance_migration_yyyymmdd = yyyymmdd
            self.public_migration_yyyymmdd = yyyymmdd
            self._log(f"[共通] 移行日を設定: {yyyymmdd}")

    # ▼ 追加: 現在有効な移行日を取得（UI → 解析 → キャッシュ）
    def _get_migration_date(self) -> str:
        """
        1) UIのプロバイダから取得して解釈成功ならそれを採用
        2) キャッシュ済みがあればそれを採用
        3) どちらも無ければ本日を採用し、キャッシュする
        """
        # 1) UIプロバイダ優先
        if self._migration_provider:
            try:
                raw = self._migration_provider()
                if raw is not None:
                    yyyymmdd = inspection._parse_date_any_to_yyyymmdd(str(raw))
                    if yyyymmdd:
                        if yyyymmdd != self._migration_yyyymmdd:
                            self._log(f"[共通] UIから移行日取得: {yyyymmdd}")
                        self._migration_yyyymmdd = yyyymmdd
                        # 後方互換
                        self.insurance_migration_yyyymmdd = yyyymmdd
                        self.public_migration_yyyymmdd = yyyymmdd
                        return yyyymmdd
            except Exception:
                pass
        # 2) キャッシュ
        if self._migration_yyyymmdd:
            return self._migration_yyyymmdd
        # 3) 本日
        today = _dt.now().strftime("%Y%m%d")
        self._migration_yyyymmdd = today
        self.insurance_migration_yyyymmdd = today
        self.public_migration_yyyymmdd = today
        self._log(f"[共通] 移行日未設定のため本日を採用: {today}")
        return today
    
    # ▼ 追加: 共通ログ関数（ロガー未設定なら何もしない）
    def _log(self, msg: str):
        try:
            if self._logger:
                self._logger(msg)
        except Exception:
            pass

    def _prog_open(self, msg: str = "処理中…"):
        """
        すでにモーダルが開いている場合は新規に作らず、メッセージだけ更新して再利用する。
        ネストした処理（外側で開いた後、内側の処理でも _prog_open を呼ぶケース）で
        ダイアログが取り残されるのを防ぐ。
        """
        try:
            # 既存モーダルが生きていれば再利用
            if getattr(self, "_prog", None):
                try:
                    # winfo_exists() が 1 を返す場合はウィンドウが存続
                    if self._prog.winfo_exists():
                        self._prog.set_message(msg)
                        self._safe_pump()
                        return
                except Exception:
                    # winfo_exists で例外時は作り直し
                    self._prog = None
            # ここまでで既存が無ければ新規作成
            self._prog = _ModalProgress(self.app, title="実行中", message=msg)
        except Exception:
            self._prog = None
        self._safe_pump()

    def _prog_set(self, msg: str):
        try:
            if getattr(self, "_prog", None):
                self._prog.set_message(msg)
        except Exception:
            pass
        self._safe_pump()

    def _prog_close(self):
        try:
            if getattr(self, "_prog", None):
                self._prog.close()
        finally:
            self._prog = None
        self._safe_pump()

    def _safe_pump(self):
        try:
            self.app.update_idletasks()
            self.app.update()
        except Exception:
            pass

    def _prepare_output_dir(self, in_path: str, key_mode: str) -> Path:
        """入力CSV(in_path)のあるフォルダ直下に、検収種別ごとの出力先フォルダを作成して返す。
        key_mode: "patient" | "insurance" | "public"
        例) /path/to/input.csv → /path/to/検収_患者 など
        """
        base = Path(in_path).resolve().parent
        folder_map = {
            "patient": "検収_患者",
            "insurance": "検収_保険",
            "public": "検収_公費",
        }
        sub = folder_map.get(key_mode, "検収_その他")
        outdir = base / sub
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir

    def _normalize_codes(self, s: pd.Series, width: int, mode: str = "zfill") -> pd.Series:
        """患者番号の正規化ヘルパ
        mode:
          - "zfill": 数字以外除去→ゼロ埋め
          - "lstrip": 数字以外除去→先頭0除去（長さ揃えない）
          - "rawdigits": 数字以外除去のみ
        """
        import re, unicodedata
        digits = s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
        if mode == "zfill":
            return digits.map(lambda x: x.zfill(width) if x else "")
        elif mode == "lstrip":
            return digits.map(lambda x: x.lstrip("0"))
        else:
            return digits

    def _make_composite_keys(self, df: pd.DataFrame, pat_col: str, sub_col: str,
                             width_pat: int, width_sub: int, mode: str = "zfill"):
        pat = self._normalize_codes(df[pat_col].astype(str), width_pat, mode=mode)
        sub = self._normalize_codes(df[sub_col].astype(str), width_sub, mode=mode)
        return list(zip(pat, sub))

    def _extract_excluded(self, src: pd.DataFrame, colmap: dict, key_mode: str,
                        migration_yyyymmdd: str | None = None) -> pd.DataFrame:
        """
        仕様に基づき「移行対象外」の行を抽出して返す。
        key_mode: "patient" | "insurance" | "public"
        戻り値: 対象外行のDataFrame（理由列 '__対象外理由__' を先頭に付与）
        """
        def _digits_or_empty(series: pd.Series) -> pd.Series:
            import re, unicodedata
            return series.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x))).map(lambda x: x if x else "")

        try:
            df = src.copy()
            payer_norm = None
            reasons = pd.Series([""] * len(df), index=df.index, dtype="object")

            if key_mode == "patient":
                # 患者ルールは core.rules.patient に集約（公費と同じ設計）
                try:
                    from core.rules.patient import evaluate_patient_exclusions, PatientRuleConfig
                except Exception as e:
                    # ルール読込失敗時は空を返す（上位で素通し扱い）
                    self._log(f"[patient] ルール読込エラー: {type(e).__name__}: {e}")
                    return src.head(0)

                # UI マッピング（colmap）はそのまま渡せば OK（患者番号/氏名/カナ/生年月日のキーを含む想定）
                cfg_rules = PatientRuleConfig()

                try:
                    remains, excluded = evaluate_patient_exclusions(df, colmap, cfg_rules)
                except Exception as e:
                    self._log(f"[patient] 対象外抽出 例外: {type(e).__name__}: {e}")
                    excluded = pd.DataFrame(columns=["__対象外理由__"])  # 空で返す

                return excluded

            elif key_mode == "insurance":
                # ルール実装を core.rules.insurance に一本化
                try:
                    from core.rules.insurance import evaluate_insurance_exclusions, InsuranceRuleConfig
                except Exception as e:
                    self._log(f"[insurance] ルール読込エラー: {type(e).__name__}: {e}")
                    return src.head(0)

                # UIマッピング → ルール側が期待するキー名へ調整
                # 期待キー: '患者番号', '保険者番号', '保険証番号', '保険終了日', '生年月日'
                proc_colmap = {
                    "患者番号": colmap.get("患者番号"),
                    "保険者番号": colmap.get("保険者番号"),
                    # 記号+番号を使っている環境では UI 側で『保険証番号』に『記号番号』を割当て済み（ログで確認済み）
                    "保険証番号": colmap.get("保険証番号") or colmap.get("記号番号"),
                    "保険終了日": colmap.get("保険終了日") or colmap.get("終了日"),
                    "生年月日": colmap.get("生年月日"),
                }

                # 移行日（YYYYMMDD）を設定
                mig = migration_yyyymmdd or self._get_migration_date()
                cfg_rules = InsuranceRuleConfig(migration_yyyymmdd=mig)

                try:
                    remains, excluded = evaluate_insurance_exclusions(src, proc_colmap, cfg_rules)
                except Exception as e:
                    self._log(f"[insurance] 対象外抽出 例外: {type(e).__name__}: {e}")
                    excluded = pd.DataFrame(columns=["__対象外理由__"])  # 空で返す

                return excluded

            elif key_mode == "public":
                # まず必要列を縦持ちフラットにしてから共通ルール適用
                # ここでは、既存の src / colmap のまま “1/2スロット” を縦持ちにする簡易展開を行う
                def _pick_first(names: list[str]):
                    # Return the first available column series among the given logical names.
                    for nm in names:
                        col = colmap.get(nm)
                        if col and col in src.columns:
                            return src[col]
                    # fallback: empty series with the same index
                    return pd.Series([""] * len(src), index=src.index, dtype="object")

                p1 = pd.DataFrame({
                    "患者番号": _pick_first(["患者番号"]),
                    "公費負担者番号": _pick_first(["公費負担者番号１", "公費負担者番号1", "負担者番号", "第一公費負担者番号", "第１公費負担者番号", "第一公費負担者番号"]),
                    "公費受給者番号": _pick_first(["公費受給者番号１", "公費受給者番号1"]),
                    "公費開始日": _pick_first(["公費開始日１", "公費開始日1"]),
                    "公費終了日": _pick_first(["公費終了日１", "公費終了日1"]),
                })

                p2 = pd.DataFrame({
                    "患者番号": _pick_first(["患者番号"]),
                    "公費負担者番号": _pick_first(["公費負担者番号２", "公費負担者番号2"]),
                    "公費受給者番号": _pick_first(["公費受給者番号２", "公費受給者番号2"]),
                    "公費開始日": _pick_first(["公費開始日２", "公費開始日2"]),
                    "公費終了日": _pick_first(["公費終了日２", "公費終了日2"]),
                })
                flat = pd.concat([p1, p2], axis=0, ignore_index=True)
                try:
                    flat = flat.fillna("").astype(str)
                except Exception:
                    pass
                self._log(f"[public] 縦持ち化: p1={len(p1)}, p2={len(p2)}, flat={len(flat)}")

                # --- 開始日が空欄の場合は移行月初(YYYYMM01)で補完（内容検収と同一ロジック） ---
                try:
                    mig = None
                    try:
                        mig = self._get_migration_date()
                    except Exception:
                        mig = None
                    if mig:
                        mig_month_first = str(mig)[:6] + "01"
                        # 日付正規化のうえ、空欄は移行月初へ
                        def _to_yyyymmdd_or_empty(v):
                            try:
                                d = inspection._parse_date_any_to_yyyymmdd(v)
                                return d if d else ""
                            except Exception:
                                return ""
                        start_norm = flat["公費開始日"].map(_to_yyyymmdd_or_empty)
                        flat["公費開始日"] = start_norm.map(lambda s: mig_month_first if s == "" else s)
                except Exception:
                    pass

                from core.rules.public import evaluate_public_exclusions, PublicRuleConfig
                mig = None
                try:
                    mig = self._get_migration_date()
                except Exception:
                    pass
                cfg_rules = PublicRuleConfig(migration_yyyymmdd=mig)

                proc_colmap = {
                    "患者番号": "患者番号",
                    "公費負担者番号": "公費負担者番号",
                    "公費受給者番号": "公費受給者番号",
                    "公費開始日": "公費開始日",
                    "公費終了日": "公費終了日",
                }
                try:
                    remains, excluded = evaluate_public_exclusions(flat, proc_colmap, cfg_rules)
                except Exception as e:
                    self._log(f"[public] 対象外抽出 例外: {type(e).__name__}: {e}")
                    excluded = pd.DataFrame(columns=["__対象外理由__"])  # 空で返す
                return excluded

            else:
                # 不明モード
                return src.head(0)

            excluded = df.loc[mask].copy()
            if not excluded.empty:
                excluded.insert(0, "__対象外理由__", reasons.loc[excluded.index])
                # 保険者番号の正規化は保険モードのときのみ挿入
                if key_mode == "insurance" and payer_norm is not None:
                    excluded.insert(1, "__正規化保険者番号__", payer_norm.loc[excluded.index])
            self._log(f"[{key_mode}] 対象外抽出 完了: {len(excluded)}件")
            return excluded
        except Exception as e:
            # 失敗時は空のDF
            self._log(f"[{key_mode}] 対象外抽出でエラー: {type(e).__name__}: {e}。空データを返します")
            return src.head(0)

    # === 共通ユーティリティ ===
    def _ask_and_save_missing_and_matched(self, *, src: pd.DataFrame, colmap: dict,
                                          out_df: pd.DataFrame, cfg: inspection.InspectionConfig,
                                          key_mode: str = "patient", out_dir: Path | None = None) -> dict:
        """
        既存検収CSVを選び、未ヒット/対象外/一致のみ を自動保存し、件数とパスを返す。
        大規模CSVでも固まらないように、比較側は usecols + chunksize で段階突合する。
        - 比較側は必要列のみをチャンク読み（患者番号と副キー）
        - キーは Python タプルではなく '患者|副' の文字列で作成（オブジェクト生成を削減）
        - 同一列の正規化はキャッシュして再利用（digits / zfill / lstrip）
        """
        summary = {
            "matched_count": 0,
            "missing_count": 0,
            "excluded_count": 0,
            "matched_path": None,
            "missing_path": None,
            "excluded_path": None,
            # 追加: 日付（開始・終了）不一致レポート
            "dates_diff_count": 0,
            "dates_diff_path": None,
        }

        # 出力用の共通タグ/パス関数（関数前半で定義してどこからでも使えるように）
        today_tag = _dt.now().strftime("%Y%m%d")
        def _path_in_dir(name: str) -> Path:
            return (out_dir / name) if out_dir else Path(name)

        import re, unicodedata
        self._log(f"[{key_mode}] 突合開始")
        self._prog_open("検収中…（突合CSVの確認）")

        # ---- 1) 比較CSVの選択 & ヘッダ取得（列名判定） ----
        from tkinter import messagebox as _mb
        cmp_path = filedialog.askopenfilename(
            title="突合対象の検収用CSV（固定カラム）を選択してください",
            filetypes=[("CSV files", "*.csv")]
        )
        if not cmp_path:
            self._log(f"[{key_mode}] 突合CSV未選択のためスキップ")
            self._prog_close()
            return summary
        self._log(f"[{key_mode}] 突合CSV: {cmp_path}")

        # まずはヘッダだけ読んで列を確認
        try:
            import pandas as _pd
            try:
                _hdr = _pd.read_csv(cmp_path, nrows=0, dtype=str, encoding="utf-8", engine="python")
            except Exception:
                _hdr = _pd.read_csv(cmp_path, nrows=0, dtype=str, encoding="cp932", engine="python")
            cmp_columns = list(_hdr.columns)
            self._log(f"[{key_mode}] 突合CSV 列ヘッダ: {cmp_columns}")
        except Exception:
            # 旧方式フォールバック（丸読み）—失敗時は従来挙動に戻す
            try:
                cmp_df = CsvLoader.read_csv_flex(cmp_path)
                cmp_columns = list(cmp_df.columns)
            except Exception:
                self._log(f"[{key_mode}] 突合CSV読込失敗")
                self._prog_close()
                return summary

        if "患者番号" not in cmp_columns:
            self._log(f"[{key_mode}] 突合CSVに患者番号がないためスキップ")
            self._prog_close()
            return summary

        # ---- 2) 比較側で使う副キー列名決定（保険/公費のみ） ----
        sub_key_name_cmp = None   # 比較(CMP)側の副キー名
        sub_key_name_src = None   # 元(SRC)側の副キー名（マッピング）
        out_sub_col = None        # 出力(OUT)側の副キー名（固定仕様名）

        if key_mode == "insurance":
            # SRC（元CSV）側の論理列 → 実列名
            payer_src = colmap.get("保険者番号")
            sym_src   = colmap.get("保険証記号")
            cno_src   = colmap.get("保険証番号")
            comb_src  = colmap.get("記号番号")

            # 比較（CMP）側の固定列名（検収CSV）
            payer_cmp = "保険者番号" if "保険者番号" in cmp_columns else None
            sym_cmp   = "保険証記号" if "保険証記号" in cmp_columns else None
            cno_cmp   = "保険証番号" if "保険証番号" in cmp_columns else None
            comb_cmp  = "記号番号" if "記号番号" in cmp_columns else None

            # 出力（OUT）側（検収CSV生成結果）
            payer_out = "保険者番号" if "保険者番号" in out_df.columns else None
            sym_out   = "保険証記号" if "保険証記号" in out_df.columns else None
            cno_out   = "保険証番号" if "保険証番号" in out_df.columns else None
            comb_out  = "記号番号" if "記号番号" in out_df.columns else None

            if not payer_src or payer_src not in src.columns or not payer_cmp:
                self._log(f"[{key_mode}] 副キー（保険者番号）不足のためスキップ")
                self._prog_close()
                return summary

        elif key_mode == "public":
            sub_key_name_src = colmap.get("公費負担者番号１") or colmap.get("負担者番号")
            out_sub_col = "公費負担者番号１"
            public_aliases_cmp = [
                "公費負担者番号１", "公費負担者番号1", "第１公費負担者番号", "第一公費負担者番号", "負担者番号",
            ]
            for cand in public_aliases_cmp:
                if cand in cmp_columns:
                    sub_key_name_cmp = cand
                    break
            if sub_key_name_cmp is None:
                # 最後の手段で選択ダイアログ
                try:
                    from .dialogs import ColumnSelectDialog
                    from pathlib import Path as _Path
                    dlg = ColumnSelectDialog(self.app, cmp_columns, title=f"[{_Path(cmp_path).name}] 突合用の公費負担者番号列を選択")
                    sub_key_name_cmp = dlg.selected if hasattr(dlg, "selected") and dlg.selected else None
                except Exception:
                    sub_key_name_cmp = None
            if not sub_key_name_src or sub_key_name_src not in src.columns or not sub_key_name_cmp:
                self._log(f"[{key_mode}] 副キー不足のためスキップ")
                self._prog_close()
                return summary

        # ---- 3) 正規化ユーティリティ（キャッシュ付き） ----
        _cache = {}
        def _digits(series: _pd.Series, tag: str):
            key = (tag, "digits", id(series))
            if key in _cache:
                return _cache[key]
            # 数字以外を除去（全角→半角を含む正規化）
            s = series.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
            # すべてが '0' の値は実質的に空欄扱いにする（例: "0", "00000000" → ""）
            s = s.map(lambda d: "" if d == "" or set(d) == {"0"} else d)
            _cache[key] = s
            return s

        def _zfill(series: _pd.Series, width: int, tag: str):
            key = (tag, "zfill", width, id(series))
            if key in _cache:
                return _cache[key]
            d = _digits(series, tag)
            z = d.map(lambda x: x.zfill(width) if x else "")
            _cache[key] = z
            return z

        def _lstrip(series: _pd.Series, tag: str):
            key = (tag, "lstrip", id(series))
            if key in _cache:
                return _cache[key]
            d = _digits(series, tag)
            ls = d.map(lambda x: x.lstrip("0"))
            _cache[key] = ls
            return ls

        def _make_key_str(pat_s: _pd.Series, sub_s: _pd.Series | None, width_pat: int, width_sub: int,
                          mode: str, tag_pat: str, tag_sub: str | None) -> _pd.Series:
            """複合キーを '患者|副' の文字列で返す（サロゲートの生成コストを抑制）"""
            if mode == "zfill":
                p = _zfill(pat_s, width_pat, tag_pat)
                if sub_s is None:
                    return p
                s = _zfill(sub_s, width_sub, tag_sub or "sub")
            else:  # lstrip
                p = _lstrip(pat_s, tag_pat)
                if sub_s is None:
                    return p
                s = _lstrip(sub_s, tag_sub or "sub")
            if sub_s is None:
                return p
            return p.astype(str).str.cat(s.astype(str), sep="|")

        # ---- 保険者番号の桁補正（7→8、5→6）: insurance.py と同等 ----
        def _payer_pad(series: _pd.Series, tag: str):
            key = (tag, "payer_pad", id(series))
            if key in _cache:
                return _cache[key]
            d = _digits(series, tag)
            def _pad(v: str) -> str:
                if v == "":
                    return ""
                n = len(v)
                if n == 7:  # 7→8
                    return v.zfill(8)
                if n == 5:  # 5→6
                    return v.zfill(6)
                return v
            out = d.map(_pad)
            _cache[key] = out
            return out

        def _payer_for_mode(series: _pd.Series, mode: str, tag: str):
            """まず _payer_pad で桁補正 → zfill: そのまま / lstrip: 先頭0除去"""
            p = _payer_pad(series, tag)
            if mode == "lstrip":
                return p.map(lambda x: x.lstrip("0"))
            return p

        # ---- 保険用: 記号番号の正規化・合成・トリプルキー ----
        def _alnum(series: _pd.Series, tag: str):
            key = (tag, "alnum", id(series))
            if key in _cache:
                return _cache[key]
            import re, unicodedata
            # 1) NFKC 正規化 + 前後空白除去
            s = series.astype(str).map(lambda x: unicodedata.normalize("NFKC", x).strip())
            # 2) NaN/NULL 文字列や空を空扱い
            null_tokens = {"", "nan", "none", "null", "n/a", "na"}
            s = s.map(lambda x: "" if x.lower() in null_tokens else x)
            # 3) 英数字のみ残して大文字化
            s = s.map(lambda x: re.sub(r"[^A-Za-z0-9]", "", x)).str.upper()
            # 4) 変換結果が空なら空のまま
            _cache[key] = s
            return s

        def _compose_symcard(sym_s: _pd.Series | None, cno_s: _pd.Series | None, tag_sym: str, tag_cno: str):
            if sym_s is None and cno_s is None:
                return _pd.Series([""] * len(src), index=src.index, dtype="object")
            if sym_s is None:
                return _alnum(cno_s, tag_cno)
            if cno_s is None:
                return _alnum(sym_s, tag_sym)
            a = _alnum(sym_s, tag_sym)
            b = _alnum(cno_s, tag_cno)
            return a.str.cat(b, na_rep="")

        def _compose_symcard_any(combined: _pd.Series | None,
                                 sym_s: _pd.Series | None,
                                 cno_s: _pd.Series | None,
                                 tag_combined: str,
                                 tag_sym: str,
                                 tag_cno: str):
            if combined is not None:
                return _alnum(combined, tag_combined)
            return _compose_symcard(sym_s, cno_s, tag_sym, tag_cno)

        def _make_key_triple(pat_s: _pd.Series,
                              payer_s: _pd.Series,
                              symcard_s: _pd.Series,
                              width_pat: int,
                              width_payer: int,
                              mode: str,
                              tag_pat: str,
                              tag_payer: str,
                              tag_sc: str) -> _pd.Series:
            if mode == "zfill":
                p = _zfill(pat_s, width_pat, tag_pat)
                y = _zfill(payer_s, width_payer, tag_payer)
                sc = _alnum(symcard_s, tag_sc)
            else:
                p = _lstrip(pat_s, tag_pat)
                y = _lstrip(payer_s, tag_payer)
                sc = _alnum(symcard_s, tag_sc)
            return p.astype(str).str.cat(y.astype(str), sep="|").str.cat(sc.astype(str), sep="|")

        # ---- 4) 患者番号/副キーの幅の決定（src/out 基準で決める） ----
        #   ※ 比較側はチャンク読みのため、幅推定をここで確定させる（十分）
        try:
            width_pat = cfg.patient_number_width
            src_code_col = colmap.get("患者番号")
            if src_code_col and src_code_col in src.columns:
                src_pat_digits = _digits(src[src_code_col], "src_pat")
                width_pat = max(width_pat, int(src_pat_digits.str.len().max() or 0))
            if "患者番号" in out_df.columns:
                out_pat_digits = _digits(out_df["患者番号"], "out_pat")
                width_pat = max(width_pat, int(out_pat_digits.str.len().max() or 0))
            if width_pat <= 0:
                width_pat = cfg.patient_number_width
        except Exception:
            width_pat = cfg.patient_number_width
        self._log(f"[{key_mode}] 患者番号幅: {width_pat}")

        width_sub = 0
        if key_mode in ("insurance", "public"):
            try:
                if sub_key_name_src and sub_key_name_src in src.columns:
                    src_sub_digits = _digits(src[sub_key_name_src], "src_sub")
                    width_sub = max(width_sub, int(src_sub_digits.str.len().max() or 0))
                if out_sub_col and out_sub_col in out_df.columns:
                    out_sub_digits = _digits(out_df[out_sub_col], "out_sub")
                    width_sub = max(width_sub, int(out_sub_digits.str.len().max() or 0))
            except Exception:
                pass
        self._log(f"[{key_mode}] 副キ―幅: {width_sub}")

        # ---- 5) 比較側キー集合をチャンクで構築 ----
        usecols = ["患者番号"]
        if key_mode == "insurance":
            for c in ["保険者番号", "保険証記号", "保険証番号", "記号番号"]:
                if c in cmp_columns:
                    usecols.append(c)
            # 開始/終了日も比較用に読み込む
            for c in ["保険開始日", "保険終了日"]:
                if c in cmp_columns and c not in usecols:
                    usecols.append(c)
        elif key_mode == "public" and sub_key_name_cmp:
            usecols.append(sub_key_name_cmp)

        cmp_keys_zfill_set = set()
        cmp_keys_lstrip_set = set()
        total_rows = 0
        self._prog_set("検収中…（突合キーを構築中）")

        # 比較側: 開始/終了日を zfill キーごとに保持
        cmp_dates_zfill: dict[str, tuple[str, str]] = {}
        # --- 比較側に同一キーの複数レコードがある場合、どれを日付比較の基準にするかを決定する ---
        try:
            _mig_for_cmp = self._get_migration_date()
        except Exception:
            _mig_for_cmp = None

        def _is_expired(end_yyyymmdd: str) -> bool:
            if not end_yyyymmdd or not _mig_for_cmp:
                return False
            try:
                return str(end_yyyymmdd) < str(_mig_for_cmp)
            except Exception:
                return False

        def _prefer_dates(new_pair: tuple[str, str], old_pair: tuple[str, str]) -> bool:
            """採用基準: 非期限切れ > 期限切れ。次に終了日の遅い方。さらに開始日の遅い方。"""
            new_s, new_e = (new_pair[0] or ""), (new_pair[1] or "")
            old_s, old_e = (old_pair[0] or ""), (old_pair[1] or "")
            new_exp = _is_expired(new_e)
            old_exp = _is_expired(old_e)
            if new_exp != old_exp:
                return not new_exp  # 非期限切れを優先
            if new_e != old_e:
                return new_e > old_e
            if new_s != old_s:
                return new_s > old_s
            return False  # 既存維持

        def _read_cmp_chunks(path: str, usecols: list[str], chunksize: int = 200_000):
            import pandas as _pd
            # まず UTF-8 を試し、ダメなら cp932
            try:
                for chunk in _pd.read_csv(path, usecols=usecols, dtype=str, chunksize=chunksize, encoding="utf-8", engine="python"):
                    yield chunk
            except Exception:
                for chunk in _pd.read_csv(path, usecols=usecols, dtype=str, chunksize=chunksize, encoding="cp932", engine="python", errors="ignore"):
                    yield chunk

        try:
            for i, chunk in enumerate(_read_cmp_chunks(cmp_path, usecols), start=1):
                total_rows += len(chunk)
                # ベクタで文字列キー作成
                pat = chunk["患者番号"]
                if key_mode == "insurance":
                    payer_c = chunk[payer_cmp] if (payer_cmp and payer_cmp in chunk.columns) else _pd.Series([""]*len(chunk))
                    sym_c   = chunk[sym_cmp]   if (sym_cmp and sym_cmp in chunk.columns)   else None
                    cno_c   = chunk[cno_cmp]   if (cno_cmp and cno_cmp in chunk.columns)   else None
                    comb_c  = chunk[comb_cmp]  if (comb_cmp and comb_cmp in chunk.columns)  else None

                    symcard = _compose_symcard_any(comb_c, sym_c, cno_c, "cmp_comb", "cmp_sym", "cmp_cno")
                    kz = _make_key_triple(pat, _payer_for_mode(payer_c, "zfill", "cmp_payer"), symcard, width_pat, width_sub, "zfill", "cmp_pat", "cmp_payer", "cmp_sc")
                    kl = _make_key_triple(pat, _payer_for_mode(payer_c, "lstrip", "cmp_payer"), symcard, width_pat, width_sub, "lstrip", "cmp_pat", "cmp_payer", "cmp_sc")

                    valid = (~_digits(pat, "cmp_pat").eq("")) & (~_payer_for_mode(payer_c, "zfill", "cmp_payer").eq("")) & (~_alnum(symcard, "cmp_sc").eq(""))
                    # 期限切れ（終了日 < 移行日）は比較集合から除外する（非期限切れ優先）
                    # ※ s_norm / e_norm の計算より後で mask を確定する

                    # 比較側: 開始/終了日を正規化してキーごとに保存（zfillキーに紐付け）
                    try:
                        def _to_yyyymmdd(v):
                            try:
                                d = inspection._parse_date_any_to_yyyymmdd(v)
                                if not d:
                                    return ""
                                d = str(d)
                                if set(d) == {"9"}:
                                    return ""
                                return d
                            except Exception:
                                return ""
                        if "保険開始日" in chunk.columns or "保険終了日" in chunk.columns:
                            s_cmp = chunk["保険開始日"] if "保険開始日" in chunk.columns else None
                            e_cmp = chunk["保険終了日"] if "保険終了日" in chunk.columns else None
                            if s_cmp is not None or e_cmp is not None:
                                s_norm = s_cmp.map(_to_yyyymmdd) if s_cmp is not None else pd.Series([""]*len(chunk))
                                e_norm = e_cmp.map(_to_yyyymmdd) if e_cmp is not None else pd.Series([""]*len(chunk))
                                kz_valid = kz.loc[valid]
                                s_take = s_norm.loc[valid]
                                e_take = e_norm.loc[valid]
                                # 非期限切れのみ採用（終了日が空、または移行日以降）
                                if _mig_for_cmp:
                                    non_exp_mask = ~((e_take != "") & (e_take < str(_mig_for_cmp)))
                                else:
                                    non_exp_mask = pd.Series([True]*len(e_take), index=e_take.index)
                                kz_valid = kz_valid.loc[non_exp_mask]
                                s_take = s_take.loc[non_exp_mask]
                                e_take = e_take.loc[non_exp_mask]
                                for k, sv, ev in zip(kz_valid.tolist(), s_take.tolist(), e_take.tolist()):
                                    cand = (sv or "", ev or "")
                                    prev = cmp_dates_zfill.get(k)
                                    if prev is None or _prefer_dates(cand, prev):
                                        cmp_dates_zfill[k] = cand
                    except Exception:
                        pass
                else:
                    sub = chunk[sub_key_name_cmp] if (key_mode in ("public",) and sub_key_name_cmp in chunk.columns) else None
                    kz = _make_key_str(pat, sub, width_pat, width_sub, "zfill", "cmp_pat", "cmp_sub" if sub is not None else None)
                    kl = _make_key_str(pat, sub, width_pat, width_sub, "lstrip", "cmp_pat", "cmp_sub" if sub is not None else None)

                if key_mode == "insurance":
                    # 期限切れは集合から除外（上のフィルタで kz/kl を valid→non-exp でスライス済みの場合に対応）
                    try:
                        cmp_keys_zfill_set.update(kz_valid.tolist())
                        cmp_keys_lstrip_set.update(kl.loc[kz_valid.index].tolist())
                    except Exception:
                        # フォールバック（全件）
                        cmp_keys_zfill_set.update(kz.tolist())
                        cmp_keys_lstrip_set.update(kl.tolist())
                else:
                    cmp_keys_zfill_set.update(kz.tolist())
                    cmp_keys_lstrip_set.update(kl.tolist())

                if i % 5 == 0:
                    self._log(f"[{key_mode}] 突合キー構築中… {total_rows:,} 行処理")
                    self._prog_set(f"検収中…（突合キー構築 {total_rows:,} 行）")
        except Exception as e:
            self._log(f"[{key_mode}] 突合キー構築でエラー: {e}")

        self._log(f"[{key_mode}] 突合キー集合 構築完了: zfill={len(cmp_keys_zfill_set):,} / lstrip={len(cmp_keys_lstrip_set):,}")

        # ---- 6) 未ヒット算出（src → cmp）※集合比較のみで高速化 ----
        self._prog_set("検収中…（未ヒット/対象外/一致の算出）")
        src_code_col = colmap.get("患者番号")
        missing_df = pd.DataFrame()
        if src_code_col and src_code_col in src.columns:
            if key_mode == "patient":
                src_z = _make_key_str(src[src_code_col], None, width_pat, 0, "zfill", "src_pat", None)
                mask_missing = (src_z != "") & (~src_z.isin(cmp_keys_zfill_set))
                if mask_missing.any():
                    missing_df = src.loc[mask_missing].copy()
                    missing_df.insert(0, "__正規化患者番号__", _zfill(src[src_code_col], width_pat, "src_pat").loc[mask_missing])
                else:
                    # フォールバック: lstrip
                    src_l = _make_key_str(src[src_code_col], None, width_pat, 0, "lstrip", "src_pat", None)
                    mask2 = (src_l != "") & (~src_l.isin(cmp_keys_lstrip_set))
                    if mask2.any():
                        missing_df = src.loc[mask2].copy()
                        missing_df.insert(0, "__正規化患者番号__", _lstrip(src[src_code_col], "src_pat").loc[mask2])
            else:
                if key_mode == "insurance":
                    pat_s   = src[src_code_col]
                    payer_s = src[payer_src] if (payer_src and payer_src in src.columns) else None
                    sym_s   = src[sym_src]   if (sym_src and sym_src in src.columns)   else None
                    cno_s   = src[cno_src]   if (cno_src and cno_src in src.columns)   else None
                    comb_s  = src[comb_src]  if (comb_src and comb_src in src.columns)  else None

                    if payer_s is None:
                        mask_missing = pd.Series([False]*len(src), index=src.index)
                    else:
                        symcard_s = _compose_symcard_any(comb_s, sym_s, cno_s, "src_comb", "src_sym", "src_cno")
                        kz = _make_key_triple(pat_s, _payer_for_mode(payer_s, "zfill", "src_payer"), symcard_s, width_pat, width_sub, "zfill", "src_pat", "src_payer", "src_sc")
                        valid = (~_digits(pat_s, "src_pat").eq("")) & (~_payer_for_mode(payer_s, "zfill", "src_payer").eq("")) & (~_alnum(symcard_s, "src_sc").eq(""))
                        mask_missing = valid & (~kz.isin(cmp_keys_zfill_set))
                    if mask_missing.any():
                        missing_df = src.loc[mask_missing].copy()
                        try:
                            missing_df.insert(0, "__正規化記号番号__", _alnum(_compose_symcard_any(comb_s, sym_s, cno_s, "src_comb", "src_sym", "src_cno"), "src_sc").loc[mask_missing])
                            missing_df.insert(0, "__正規化保険者番号__", _payer_for_mode(payer_s, "zfill", "src_payer").loc[mask_missing] if payer_s is not None else "")
                            missing_df.insert(0, "__正規化患者番号__", _zfill(pat_s, width_pat, "src_pat").loc[mask_missing])
                        except Exception:
                            pass
                    if missing_df.empty and payer_s is not None:
                        kl = _make_key_triple(pat_s, payer_s, _compose_symcard_any(comb_s, sym_s, cno_s, "src_comb", "src_sym", "src_cno"), width_pat, width_sub, "lstrip", "src_pat", "src_payer", "src_sc")
                        valid2 = (~_lstrip(pat_s, "src_pat").eq("")) & (~_lstrip(payer_s, "src_payer").eq("")) & (~_alnum(_compose_symcard_any(comb_s, sym_s, cno_s, "src_comb", "src_sym", "src_cno"), "src_sc").eq(""))
                        mask2 = valid2 & (~kl.isin(cmp_keys_lstrip_set))
                        if mask2.any():
                            missing_df = src.loc[mask2].copy()
                            try:
                                missing_df.insert(0, "__正規化記号番号__", _alnum(_compose_symcard_any(comb_s, sym_s, cno_s, "src_comb", "src_sym", "src_cno"), "src_sc").loc[mask2])
                                missing_df.insert(0, "__正規化保険者番号__", _payer_for_mode(payer_s, "lstrip", "src_payer").loc[mask2])
                                missing_df.insert(0, "__正規化患者番号__", _lstrip(pat_s, "src_pat").loc[mask2])
                            except Exception:
                                pass
                else:
                    # 複合キー
                    sub_src = sub_key_name_src
                    if sub_src and sub_src in src.columns:
                        kz = _make_key_str(src[src_code_col], src[sub_src], width_pat, width_sub, "zfill", "src_pat", "src_sub")
                        valid = kz.str.contains(r"\|") & (~kz.str.startswith("|")) & (~kz.str.endswith("|"))
                        mask_missing = valid & (~kz.isin(cmp_keys_zfill_set))
                        if mask_missing.any():
                            missing_df = src.loc[mask_missing].copy()
                            missing_df.insert(0, "__正規化副キー__", _zfill(src[sub_src], width_sub, "src_sub").loc[mask_missing])
                            missing_df.insert(0, "__正規化患者番号__", _zfill(src[src_code_col], width_pat, "src_pat").loc[mask_missing])
                        if missing_df.empty:
                            kl = _make_key_str(src[src_code_col], src[sub_src], width_pat, width_sub, "lstrip", "src_pat", "src_sub")
                            valid2 = kl.str.contains(r"\|") & (~kl.str.startswith("|")) & (~kl.str.endswith("|"))
                            mask2 = valid2 & (~kl.isin(cmp_keys_lstrip_set))
                            if mask2.any():
                                missing_df = src.loc[mask2].copy()
                                missing_df.insert(0, "__正規化副キー__", _lstrip(src[sub_src], "src_sub").loc[mask2])
                                missing_df.insert(0, "__正規化患者番号__", _lstrip(src[src_code_col], "src_pat").loc[mask2])

        self._log(f"[{key_mode}] 未ヒット算出: {len(missing_df)}件")

        # ---- (診断) 未ヒットの原因内訳を簡易出力（insurance のみ） ----
        try:
            if key_mode == "insurance" and isinstance(missing_df, pd.DataFrame) and not missing_df.empty:
                # 元CSV側の参照列（存在しない場合は空列を作る）
                src_code_col = colmap.get("患者番号")
                payer_src = colmap.get("保険者番号") if colmap.get("保険者番号") in src.columns else None
                comb_src  = colmap.get("記号番号") if colmap.get("記号番号") in src.columns else None
                sym_src   = colmap.get("保険証記号") if colmap.get("保険証記号") in src.columns else None
                cno_src   = colmap.get("保険証番号") if colmap.get("保険証番号") in src.columns else None

                # 正規化（患者/保険者/記号番号）
                pat_norm = missing_df["__正規化患者番号__"] if "__正規化患者番号__" in missing_df.columns else _zfill(src[src_code_col], width_pat, "diag_src_pat").loc[missing_df.index]
                payer_norm = missing_df["__正規化保険者番号__"] if "__正規化保険者番号__" in missing_df.columns else (_payer_for_mode(src[payer_src], "zfill", "diag_src_payer").loc[missing_df.index] if payer_src else pd.Series([""]*len(missing_df), index=missing_df.index))
                sc_norm = missing_df["__正規化記号番号__"] if "__正規化記号番号__" in missing_df.columns else (
                    _alnum(_compose_symcard_any(src.get(comb_src), src.get(sym_src), src.get(cno_src), "diag_src_comb", "diag_src_sym", "diag_src_cno"), "diag_src_sc").loc[missing_df.index]
                )

                # 空判定
                pat_empty = pat_norm.eq("")
                payer_empty = payer_norm.eq("")
                sc_empty = sc_norm.eq("")

                # CMP 側のキー集合に無いか（zfill キー）
                src_key = pat_norm.astype(str).str.cat(payer_norm.astype(str), sep="|").str.cat(sc_norm.astype(str), sep="|")
                in_cmp = src_key.isin(cmp_keys_zfill_set)

                # まとめ
                diag = pd.DataFrame({
                    "__正規化患者番号__": pat_norm,
                    "__正規化保険者番号__": payer_norm,
                    "__正規化記号番号__": sc_norm,
                    "__診断:患者空__": pat_empty,
                    "__診断:保険者空__": payer_empty,
                    "__診断:記号番号空__": sc_empty,
                    "__診断:集合に存在__": in_cmp,
                }, index=missing_df.index)

                # 集計ログ
                try:
                    total = len(diag)
                    c_payer_empty = int(payer_empty.sum())
                    c_sc_empty = int(sc_empty.sum())
                    c_pat_empty = int(pat_empty.sum())
                    c_in_cmp = int(in_cmp.sum())
                    self._log(f"[insurance][診断] 未ヒット {total} 件 / 患者空 {c_pat_empty} / 保険者空 {c_payer_empty} / 記号番号空 {c_sc_empty} / (正規化キーが比較集合に存在) {c_in_cmp}")
                except Exception:
                    pass

                # サンプル 200 件だけ抜粋を保存（全件は未ヒットCSVに含まれるため）
                try:
                    sample = diag.head(200).copy()
                    diag_path = (out_dir / f"保険_未ヒット_診断_{_dt.now().strftime('%Y%m%d')}.csv") if out_dir else Path(f"保険_未ヒット_診断_{_dt.now().strftime('%Y%m%d')}.csv")
                    inspection.to_csv(sample, str(diag_path))
                    self._log(f"[insurance][診断] 未ヒット診断サンプルを出力: {diag_path}")
                except Exception:
                    pass
        except Exception:
            pass

        # ---- 7) 対象外算出（仕様ルール + 未分類編入）----
        excluded_df = pd.DataFrame()
        try:
            mig = self._get_migration_date()
            excluded_df = self._extract_excluded(src=src, colmap=colmap, key_mode=key_mode, migration_yyyymmdd=mig)

            # 一致（src基準）判定を「集合」で実施してから対象外から除外
            matched_mask_src = pd.Series([False] * len(src), index=src.index)
            try:
                if src_code_col and src_code_col in src.columns:
                    if key_mode == "patient":
                        src_z_m = _make_key_str(src[src_code_col], None, width_pat, 0, "zfill", "src_pat", None)
                        matched_mask_src = (src_z_m != "") & (src_z_m.isin(cmp_keys_zfill_set))
                    else:
                        sub_src = sub_key_name_src
                        if sub_src and sub_src in src.columns:
                            kz = _make_key_str(src[src_code_col], src[sub_src], width_pat, width_sub, "zfill", "src_pat", "src_sub")
                            valid = kz.str.contains(r"\|") & (~kz.str.startswith("|")) & (~kz.str.endswith("|"))
                            matched_mask_src = valid & kz.isin(cmp_keys_zfill_set)
                if excluded_df is not None and not excluded_df.empty and matched_mask_src.any():
                    excluded_df = excluded_df.loc[~excluded_df.index.isin(src.index[matched_mask_src])]
            except Exception:
                pass

            # Eligible 未ヒットを『未分類（未ヒット・要ルール）』として "未ヒット" に表示
            try:
                excluded_idx_now = set(excluded_df.index) if isinstance(excluded_df, pd.DataFrame) and not excluded_df.empty else set()
                eligible_mask_now = ~src.index.to_series().isin(excluded_idx_now)
                unmatched_eligible_mask = eligible_mask_now & (~matched_mask_src)

                if unmatched_eligible_mask.any():
                    # 正規化キー列が未作成で、missing_df が空の場合はここで作成
                    if isinstance(missing_df, pd.DataFrame) and missing_df.empty:
                        unmatched_df = src.loc[unmatched_eligible_mask].copy()
                        # 正規化キー付与
                        try:
                            if src_code_col and src_code_col in src.columns:
                                pat_norm = _zfill(src[src_code_col], width_pat, "src_pat")
                                unmatched_df.insert(0, "__正規化患者番号__", pat_norm.loc[unmatched_eligible_mask])
                            if key_mode in ("insurance", "public"):
                                sub_src = sub_key_name_src
                                if sub_src and sub_src in src.columns:
                                    sub_norm = _zfill(src[sub_src], width_sub, "src_sub")
                                    unmatched_df.insert(1, "__正規化副キー__", sub_norm.loc[unmatched_eligible_mask])
                        except Exception:
                            pass
                        missing_df = unmatched_df

                    # 未ヒット理由カラムを付与/更新
                    if isinstance(missing_df, pd.DataFrame):
                        if "__未ヒット理由__" not in missing_df.columns:
                            missing_df.insert(0, "__未ヒット理由__", "")
                        target_idx = missing_df.index.intersection(src.index[unmatched_eligible_mask])
                        if len(target_idx) > 0:
                            missing_df.loc[target_idx, "__未ヒット理由__"] = "未分類（未ヒット・要ルール）"
            except Exception:
                pass
        except Exception:
            excluded_df = pd.DataFrame()
        self._log(f"[{key_mode}] 対象外算出: {len(excluded_df)}件")

        # ---- 8) 未ヒットから対象外を除去（重複計上防止） ----
        try:
            if isinstance(missing_df, pd.DataFrame) and not missing_df.empty and isinstance(excluded_df, pd.DataFrame) and not excluded_df.empty:
                before_cnt = len(missing_df)
                missing_df = missing_df.loc[~missing_df.index.isin(excluded_df.index)].copy()
                self._log(f"[{key_mode}] 未ヒット⇔対象外の重複を除外: {before_cnt} → {len(missing_df)}")
        except Exception:
            pass

        # ---- 9) 一致のみ（out_df 基準：出力用DFのキーと比較側集合の積集合） ----
        filtered_out_df = pd.DataFrame()
        try:
            if "患者番号" in out_df.columns:
                if key_mode == "patient":
                    out_k = _make_key_str(out_df["患者番号"], None, width_pat, 0, "zfill", "out_pat", None)
                    mask_matched = (out_k != "") & (out_k.isin(cmp_keys_zfill_set))
                    filtered_out_df = out_df.loc[mask_matched].copy()
                    if filtered_out_df.empty:
                        out_k2 = _make_key_str(out_df["患者番号"], None, width_pat, 0, "lstrip", "out_pat", None)
                        mask2 = (out_k2 != "") & (out_k2.isin(cmp_keys_lstrip_set))
                        if mask2.any():
                            filtered_out_df = out_df.loc[mask2].copy()
                else:
                    if key_mode == "insurance":
                        if "患者番号" in out_df.columns and payer_out:
                            symcard_out = _compose_symcard_any(out_df.get(comb_out), out_df.get(sym_out), out_df.get(cno_out), "out_comb", "out_sym", "out_cno")
                            out_k = _make_key_triple(out_df["患者番号"], _payer_for_mode(out_df.get(payer_out), "zfill", "out_payer"), symcard_out, width_pat, width_sub, "zfill", "out_pat", "out_payer", "out_sc")
                            valid = (~_digits(out_df["患者番号"], "out_pat").eq("")) & (~_payer_for_mode(out_df.get(payer_out), "zfill", "out_payer").eq("")) & (~_alnum(symcard_out, "out_sc").eq(""))
                            mask = valid & out_k.isin(cmp_keys_zfill_set)
                            filtered_out_df = out_df.loc[mask].copy()
                            if filtered_out_df.empty:
                                out_k2 = _make_key_triple(out_df["患者番号"], _payer_for_mode(out_df.get(payer_out), "lstrip", "out_payer"), symcard_out, width_pat, width_sub, "lstrip", "out_pat", "out_payer", "out_sc")
                                mask2 = valid & out_k2.isin(cmp_keys_lstrip_set)
                                if mask2.any():
                                    filtered_out_df = out_df.loc[mask2].copy()
                    else:
                        if out_sub_col and out_sub_col in out_df.columns:
                            out_k = _make_key_str(out_df["患者番号"], out_df[out_sub_col], width_pat, width_sub, "zfill", "out_pat", "out_sub")
                            valid = out_k.str.contains(r"\|") & (~out_k.str.startswith("|")) & (~out_k.str.endswith("|"))
                            mask = valid & out_k.isin(cmp_keys_zfill_set)
                            filtered_out_df = out_df.loc[mask].copy()
                            if filtered_out_df.empty:
                                out_k2 = _make_key_str(out_df["患者番号"], out_df[out_sub_col], width_pat, width_sub, "lstrip", "out_pat", "out_sub")
                                valid2 = out_k2.str.contains(r"\|") & (~out_k2.str.startswith("|")) & (~out_k2.str.endswith("|"))
                                mask2 = valid2 & out_k2.isin(cmp_keys_lstrip_set)
                                if mask2.any():
                                    filtered_out_df = out_df.loc[mask2].copy()

                # 正規化キーで重複除去（完全重複のみ落とす）
                if not filtered_out_df.empty:
                    try:
                        if key_mode == "insurance":
                            # --- 重複キーがある場合: 非期限切れ > 終了日が遅い > 開始日が遅い を優先して1件に絞る ---
                            # 1) zfillキーを算出
                            symcard_out = _compose_symcard_any(filtered_out_df.get(comb_out), filtered_out_df.get(sym_out), filtered_out_df.get(cno_out), "out_comb", "out_sym", "out_cno")
                            key_z = _make_key_triple(
                                filtered_out_df["患者番号"],
                                _payer_for_mode(filtered_out_df.get(payer_out), "zfill", "out_payer"),
                                symcard_out,
                                width_pat, width_sub, "zfill", "out_pat", "out_payer", "out_sc"
                            )

                            # 2) out側の開始・終了日を正規化
                            def _to_yyyymmdd(v):
                                try:
                                    d = inspection._parse_date_any_to_yyyymmdd(v)
                                    if not d:
                                        return ""
                                    d = str(d)
                                    if set(d) == {"9"}:
                                        return ""
                                    return d
                                except Exception:
                                    return ""
                            s_norm = (filtered_out_df.get("保険開始日").map(_to_yyyymmdd)
                                      if "保険開始日" in filtered_out_df.columns else pd.Series([""]*len(filtered_out_df), index=filtered_out_df.index))
                            e_norm = (filtered_out_df.get("保険終了日").map(_to_yyyymmdd)
                                      if "保険終了日" in filtered_out_df.columns else pd.Series([""]*len(filtered_out_df), index=filtered_out_df.index))
                            # 3) 元の開始日が空欄なら移行月初(YYYYMM01)で比較する
                            try:
                                _mig_local = self._get_migration_date()
                                _mig_month_first = str(_mig_local)[:6] + "01" if _mig_local else None
                                if _mig_month_first:
                                    s_norm = s_norm.map(lambda v: _mig_month_first if (v == "") else v)
                            except Exception:
                                pass

                            # 4) キー毎に優先レコードを選択
                            selected_idx = {}
                            for idx, k, so, eo in zip(filtered_out_df.index, key_z.values, s_norm.values, e_norm.values):
                                if k not in selected_idx:
                                    selected_idx[k] = idx
                                else:
                                    prev_idx = selected_idx[k]
                                    so_prev = s_norm.loc[prev_idx]
                                    eo_prev = e_norm.loc[prev_idx]
                                    # 既存ヘルパーの優先規則を使用（非期限切れ > 終了日遅い > 開始日遅い）
                                    if _prefer_dates((so or "", eo or ""), (so_prev or "", eo_prev or "")):
                                        selected_idx[k] = idx
                            keep = list(selected_idx.values())
                            filtered_out_df = filtered_out_df.loc[keep].copy()
                        elif key_mode == "public":
                            if out_sub_col in filtered_out_df.columns:
                                key_l = _make_key_str(filtered_out_df["患者番号"], filtered_out_df[out_sub_col], width_pat, width_sub, "lstrip", "out_pat", "out_sub")
                                keep_mask = ~key_l.duplicated(keep="first")
                                filtered_out_df = filtered_out_df.loc[keep_mask].copy()
                        else:
                            key_l = _make_key_str(filtered_out_df["患者番号"], None, width_pat, width_sub, "lstrip", "out_pat", None)
                            keep_mask = ~key_l.duplicated(keep="first")
                            filtered_out_df = filtered_out_df.loc[keep_mask].copy()
                    except Exception:
                        # フォールバック
                        if key_mode == "insurance":
                            filtered_out_df = filtered_out_df.drop_duplicates(subset=["患者番号", payer_out, comb_out, sym_out, cno_out], keep="first")
                        elif key_mode == "public" and out_sub_col in filtered_out_df.columns:
                            filtered_out_df = filtered_out_df.drop_duplicates(subset=["患者番号", out_sub_col], keep="first")
                        elif key_mode == "patient":
                            filtered_out_df = filtered_out_df.drop_duplicates(subset=["患者番号"], keep="first")

                # ---- 保険開始日・終了日の不一致レポート作成 ----
                if key_mode == "insurance":
                    try:
                        diff_rows = []
                        if isinstance(filtered_out_df, pd.DataFrame) and not filtered_out_df.empty and cmp_dates_zfill:
                            # out 側のキー（zfill）を再計算
                            symcard_out2 = _compose_symcard_any(filtered_out_df.get(comb_out), filtered_out_df.get(sym_out), filtered_out_df.get(cno_out), "out_comb", "out_sym", "out_cno")
                            out_k_z = _make_key_triple(
                                filtered_out_df["患者番号"],
                                _payer_for_mode(filtered_out_df.get(payer_out), "zfill", "out_payer"),
                                symcard_out2,
                                width_pat, width_sub, "zfill", "out_pat", "out_payer", "out_sc"
                            )
                            # out 側の日付を正規化
                            def _to_yyyymmdd(v):
                                try:
                                    d = inspection._parse_date_any_to_yyyymmdd(v)
                                    if not d:
                                        return ""
                                    d = str(d)
                                    if set(d) == {"9"}:
                                        return ""
                                    return d
                                except Exception:
                                    return ""
                            s_out = filtered_out_df.get("保険開始日")
                            e_out = filtered_out_df.get("保険終了日")
                            s_out = s_out.map(_to_yyyymmdd) if s_out is not None else pd.Series([""]*len(filtered_out_df), index=filtered_out_df.index)
                            e_out = e_out.map(_to_yyyymmdd) if e_out is not None else pd.Series([""]*len(filtered_out_df), index=filtered_out_df.index)
                            # 元の開始日が空欄なら、移行月初(YYYYMM01)で比較する
                            try:
                                mig = self._get_migration_date()
                                mig_month_first = str(mig)[:6] + "01" if mig else None
                                if mig_month_first:
                                    s_out = s_out.map(lambda v: mig_month_first if (v == "") else v)
                            except Exception:
                                pass

                            for row_i, key_k in zip(out_k_z.index, out_k_z.values):
                                if key_k in cmp_dates_zfill:
                                    s_cmp, e_cmp = cmp_dates_zfill.get(key_k, ("", ""))
                                    so = s_out.loc[row_i]
                                    eo = e_out.loc[row_i]
                                    # 期限切れ（元側）が比較に紛れないように、元の終了日が移行日より前なら差分判定をスキップ
                                    try:
                                        _mig_for_diff = self._get_migration_date()
                                    except Exception:
                                        _mig_for_diff = None
                                    if _mig_for_diff and (eo != "") and (eo < str(_mig_for_diff)):
                                        continue

                                    # --- 値の単純一致ではなく、移行日(YYYYMMDD)時点の有効性で差分判定する ---
                                    try:
                                        mig_eff = self._get_migration_date()
                                    except Exception:
                                        mig_eff = None

                                    def _eff_start(s):
                                        # 開始日が空なら「昔から有効」とみなす（= 移行日基準で True）
                                        if not s:
                                            return True
                                        if not mig_eff:
                                            return True
                                        return s <= str(mig_eff)

                                    def _eff_end(e):
                                        # 終了日が空なら「期限なし（有効）」とみなす
                                        if not e:
                                            return True
                                        if not mig_eff:
                                            return True
                                        return e >= str(mig_eff)

                                    start_eff_out = _eff_start(so)
                                    start_eff_cmp = _eff_start(s_cmp)
                                    end_eff_out   = _eff_end(eo)
                                    end_eff_cmp   = _eff_end(e_cmp)

                                    start_eff_diff = (start_eff_out != start_eff_cmp)
                                    end_eff_diff   = (end_eff_out != end_eff_cmp)

                                    # 有効性が同じなら差分としては扱わない（移行ルールに則った比較）
                                    if not (start_eff_diff or end_eff_diff):
                                        continue

                                    # ここから下は「有効性が異なる」場合のみ差分として出力
                                    if start_eff_diff and end_eff_diff:
                                        diff_kind = "開始・終了(有効性)"
                                    elif start_eff_diff:
                                        diff_kind = "開始(有効性)"
                                    else:
                                        diff_kind = "終了(有効性)"

                                    # 可能なら記号/番号/合成も出力（既存と同様）
                                    sym_val = filtered_out_df.get("保険証記号").loc[row_i] if (sym_out and sym_out in filtered_out_df.columns) else ""
                                    cno_val = filtered_out_df.get("保険証番号").loc[row_i] if (cno_out and cno_out in filtered_out_df.columns) else ""
                                    comb_val = filtered_out_df.get("記号番号").loc[row_i] if ("記号番号" in filtered_out_df.columns) else ""
                                    payer_val = filtered_out_df.get(payer_out).loc[row_i] if (payer_out and payer_out in filtered_out_df.columns) else ""
                                    diff_rows.append({
                                        "患者番号": filtered_out_df["患者番号"].loc[row_i],
                                        "保険者番号": payer_val,
                                        "保険証記号": sym_val,
                                        "保険証番号": cno_val,
                                        "記号番号": comb_val,
                                        # 元（= out_df）値も併記
                                        "元:保険開始日": so,
                                        "元:保険終了日": eo,
                                        # 検収（= cmp）値
                                        "検収:保険開始日": s_cmp,
                                        "検収:保険終了日": e_cmp,
                                        "差分種別": diff_kind,
                                    })
                        # CSV 出力
                        if diff_rows:
                            diff_df = pd.DataFrame(diff_rows)
                            diff_path = _path_in_dir(f"保険_差異_開始終了_{today_tag}.csv")
                            inspection.to_csv(diff_df, str(diff_path))
                            summary["dates_diff_count"] = len(diff_df)
                            summary["dates_diff_path"] = str(diff_path)
                            self._log(f"[insurance] 開始/終了の不一致: {len(diff_df)}件 → {diff_path}")
                        else:
                            self._log("[insurance] 開始/終了の不一致: 0件")
                    except Exception as e:
                        self._log(f"[insurance] 開始/終了差分の算出でエラー: {type(e).__name__}: {e}")
        except Exception:
            filtered_out_df = pd.DataFrame()
        self._log(f"[{key_mode}] 一致のみ算出: {len(filtered_out_df)}件")

        # ---- 10) 出力 ----
        self._prog_set("書き出し中…（CSV出力）")
        prefix = "保険" if key_mode == "insurance" else ("公費" if key_mode == "public" else "患者")

        try:
            miss_path = _path_in_dir(f"{prefix}_未ヒット_{today_tag}.csv")
            inspection.to_csv(missing_df, str(miss_path))
            summary["missing_count"] = len(missing_df)
            summary["missing_path"]  = str(miss_path)
            self._log(f"[{key_mode}] 未ヒット出力: {miss_path}")
        except Exception:
            self._log(f"[{key_mode}] 未ヒット出力に失敗しました")

        try:
            ex_path = _path_in_dir(f"{prefix}_対象外_{today_tag}.csv")
            inspection.to_csv(excluded_df, str(ex_path))
            summary["excluded_count"] = len(excluded_df)
            summary["excluded_path"]  = str(ex_path)
            self._log(f"[{key_mode}] 対象外出力: {ex_path}")
        except Exception:
            self._log(f"[{key_mode}] 対象外出力に失敗しました")

        try:
            matched_path = _path_in_dir(f"{prefix}_検収_一致のみ_{today_tag}.csv")
            inspection.to_csv(filtered_out_df, str(matched_path))
            summary["matched_count"] = len(filtered_out_df)
            summary["matched_path"]  = str(matched_path)
            self._log(f"[{key_mode}] 一致のみ出力: {matched_path}")
        except Exception:
            self._log(f"[{key_mode}] 一致のみ出力に失敗しました")

        # ---- 10.1) サマリテキスト出力（件数 + 対象外内訳 + 開始/終了差分） ----
        try:
            summary_path = _path_in_dir(f"{prefix}_検収_サマリ_{today_tag}.txt")
            lines: list[str] = []
            try:
                ts = _dt.now().strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                ts = ''
            lines.append(f"検収サマリ ({key_mode}) {ts}")
            try:
                lines.append(f"比較CSV: {cmp_path}")
            except Exception:
                pass
            lines.append(f"一致のみ: {summary.get('matched_count', 0)}件 -> {summary.get('matched_path')}")
            lines.append(f"未ヒット: {summary.get('missing_count', 0)}件 -> {summary.get('missing_path')}")
            lines.append(f"対象外: {summary.get('excluded_count', 0)}件 -> {summary.get('excluded_path')}")
            if key_mode == "insurance":
                lines.append(f"開始/終了の不一致: {summary.get('dates_diff_count', 0)}件 -> {summary.get('dates_diff_path')}")

            # 対象外の理由内訳（理由は「 / 」区切りの複数可）
            try:
                if isinstance(excluded_df, pd.DataFrame) and not excluded_df.empty and ("__対象外理由__" in excluded_df.columns):
                    from collections import Counter
                    counter = Counter()
                    for s in excluded_df["__対象外理由__"].astype(str):
                        if not s:
                            continue
                        parts = [x for x in s.split(" / ") if x]
                        for r in parts:
                            counter[r] += 1
                    if counter:
                        lines.append("")
                        lines.append("[対象外 内訳]")
                        for reason, cnt in counter.most_common():
                            lines.append(f"{reason}\t{cnt}")
            except Exception:
                pass

            try:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines) + "\n")
                self._log(f"[{key_mode}] サマリ出力: {summary_path}")
            except Exception as e:
                self._log(f"[{key_mode}] サマリ出力に失敗しました: {type(e).__name__}: {e}")
        except Exception:
            self._log(f"[{key_mode}] サマリ出力に失敗しました")

        self._prog_close()
        return summary

    # === 各アクション ===
    def run_patient(self):
        in_path = filedialog.askopenfilename(title="患者情報CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        self._log(f"[患者] 入力CSV: {in_path}")
        if not in_path:
            return False
        try:
            try:
                self._prog_open("CSV読み込み中…")
                src = CsvLoader.read_csv_flex(in_path)
            finally:
                self._prog_close()
            out_dir = self._prepare_output_dir(in_path, "patient")
            self._log(f"[患者] 出力先ディレクトリ: {out_dir}")
            colmap = self.app._ask_inspection_colmap(src, required_cols=list(inspection.COLUMNS_PATIENT))
            if colmap is None:
                return False
            
            self._log(f"[患者] マッピング完了: {colmap}")
            colmaps = self._load_colmaps()
            colmaps["patient"] = colmap
            self._save_colmaps(colmaps)
            cfg = inspection.InspectionConfig(patient_number_width=10)
            out_df = inspection.build_inspection_df(src, colmap, cfg, target_columns=inspection.COLUMNS_PATIENT)

            default_name = f"患者_検収_{_dt.now().strftime('%Y%m%d')}.csv"
            out_path = filedialog.asksaveasfilename(
                title="検収CSVを保存",
                defaultextension=".csv",
                initialfile=default_name,
                initialdir=str(out_dir),
                filetypes=[("CSV files", "*.csv")]
            )
            if not out_path:
                return False
            self._prog_open("検収CSVを書き出し中…")
            inspection.to_csv(out_df, out_path)
            self._prog_close()
            self._log(f"[患者] 検収CSV出力: {out_path} (行数 {len(out_df)})")

            self._prog_open("検収中…（突合とリスト作成）")
            summary = self._ask_and_save_missing_and_matched(
                src=src, colmap=colmap, out_df=out_df, cfg=cfg, key_mode="patient", out_dir=out_dir
            )
            self._prog_close()
            self._log(f"[患者] 一致のみ: {summary.get('matched_count', 0)} → {summary.get('matched_path')}")
            self._log(f"[患者] 未ヒット: {summary.get('missing_count', 0)} → {summary.get('missing_path')}")
            self._log(f"[患者] 対象外: {summary.get('excluded_count', 0)} → {summary.get('excluded_path')}")
            messagebox.showinfo(
                "完了",
                f"出力が完了しました。\n一致: {summary.get('matched_count', 0)} 件 / 未ヒット: {summary.get('missing_count', 0)} 件 / 対象外: {summary.get('excluded_count', 0)} 件"
            )
            return True
        except Exception as e:
            messagebox.showerror("エラー", f"検収処理中に失敗しました。\n{e}")
            return False
        finally:
            self._prog_close()

    def run_insurance(self):
        in_path = filedialog.askopenfilename(title="保険情報の入力CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        self._log(f"[保険] 入力CSV: {in_path}")
        if not in_path:
            return False
        try:
            try:
                self._prog_open("CSV読み込み中…")
                src = CsvLoader.read_csv_flex(in_path)
            finally:
                self._prog_close()
            out_dir = self._prepare_output_dir(in_path, "insurance")
            self._log(f"[保険] 出力先ディレクトリ: {out_dir}")
            required_cols = list(inspection.COLUMNS_INSURANCE)
            colmap = self.app._ask_inspection_colmap(src, required_cols=required_cols)
            if colmap is None:
                return False
            
            self._log(f"[保険] マッピング完了: {colmap}")
            colmaps = self._load_colmaps()
            colmaps["insurance"] = colmap
            self._save_colmaps(colmaps)
            mig = self._get_migration_date()
            self.insurance_migration_yyyymmdd = mig  # 後方互換
            self._log(f"[保険] 移行日: {mig}")
            cfg = inspection.InspectionConfig(patient_number_width=10)
            out_df = inspection.build_inspection_df(src, colmap, cfg, target_columns=inspection.COLUMNS_INSURANCE)

            default_name = f"保険_検収_{_dt.now().strftime('%Y%m%d')}.csv"
            out_path = filedialog.asksaveasfilename(
                title="保険情報 検収CSVを保存",
                defaultextension=".csv",
                initialfile=default_name,
                initialdir=str(out_dir),
                filetypes=[("CSV files", "*.csv")]
            )
            if not out_path:
                return False
            self._prog_open("検収CSVを書き出し中…")
            inspection.to_csv(out_df, out_path)
            self._prog_close()
            self._log(f"[保険] 検収CSV出力: {out_path} (行数 {len(out_df)})")

            self._prog_open("検収中…（突合とリスト作成）")
            summary = self._ask_and_save_missing_and_matched(
                src=src, colmap=colmap, out_df=out_df, cfg=cfg, key_mode="insurance", out_dir=out_dir
            )
            self._prog_close()
            self._log(f"[保険] 一致のみ: {summary.get('matched_count', 0)} → {summary.get('matched_path')}")
            self._log(f"[保険] 未ヒット: {summary.get('missing_count', 0)} → {summary.get('missing_path')}")
            self._log(f"[保険] 対象外: {summary.get('excluded_count', 0)} → {summary.get('excluded_path')}")
            messagebox.showinfo(
                "完了",
                f"出力が完了しました。\n一致: {summary.get('matched_count', 0)} 件 / 未ヒット: {summary.get('missing_count', 0)} 件 / 対象外: {summary.get('excluded_count', 0)} 件"
            )
            return True
        except Exception as e:
            messagebox.showerror("エラー", f"保険情報の検収処理に失敗しました。\n{e}")
            return False
        finally:
            self._prog_close()

    def run_public(self):
        in_path = filedialog.askopenfilename(title="公費情報の入力CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        self._log(f"[公費] 入力CSV: {in_path}")
        if not in_path:
            return False
        try:
            try:
                self._prog_open("CSV読み込み中…")
                src = CsvLoader.read_csv_flex(in_path)
            finally:
                self._prog_close()
            out_dir = self._prepare_output_dir(in_path, "public")
            self._log(f"[公費] 出力先ディレクトリ: {out_dir}")
            required_cols = list(inspection.COLUMNS_PUBLIC) + ["公費終了日１", "公費終了日２"]
            colmap = self.app._ask_inspection_colmap(src, required_cols=required_cols)
            if colmap is None:
                return False
            self._log(f"[公費] マッピング完了: {colmap}")
            colmaps = self._load_colmaps()
            colmaps["public"] = colmap
            self._save_colmaps(colmaps)
            # 共通の移行日（検収ページ右上の入力欄から取得／キャッシュ）
            mig = self._get_migration_date()
            self.public_migration_yyyymmdd = mig  # 後方互換
            self._log(f"[公費] 移行日: {mig}")
            cfg = inspection.InspectionConfig(patient_number_width=10)
            out_df = inspection.build_inspection_df(
                src,
                colmap,
                cfg,
                target_columns=list(inspection.COLUMNS_PUBLIC) + ["公費終了日１", "公費終了日２"]
            )

            default_name = f"公費_検収_{_dt.now().strftime('%Y%m%d')}.csv"
            out_path = filedialog.asksaveasfilename(
                title="公費情報 検収CSVを保存",
                defaultextension=".csv",
                initialfile=default_name,
                initialdir=str(out_dir),
                filetypes=[("CSV files", "*.csv")]
            )
            if not out_path:
                return False
            self._prog_open("検収CSVを書き出し中…")
            inspection.to_csv(out_df, out_path)
            self._prog_close()
            self._log(f"[公費] 検収CSV出力: {out_path} (行数 {len(out_df)})")

            self._prog_open("検収中…（突合とリスト作成）")
            summary = self._ask_and_save_missing_and_matched(
                src=src, colmap=colmap, out_df=out_df, cfg=cfg, key_mode="public", out_dir=out_dir
            )
            self._prog_close()
            self._log(f"[公費] 一致のみ: {summary.get('matched_count', 0)} → {summary.get('matched_path')}")
            self._log(f"[公費] 未ヒット: {summary.get('missing_count', 0)} → {summary.get('missing_path')}")
            self._log(f"[公費] 対象外: {summary.get('excluded_count', 0)} → {summary.get('excluded_path')}")
            messagebox.showinfo(
                "完了",
                f"出力が完了しました。\n一致: {summary.get('matched_count', 0)} 件 / 未ヒット: {summary.get('missing_count', 0)} 件 / 対象外: {summary.get('excluded_count', 0)} 件"
            )
            return True
        except Exception as e:
            messagebox.showerror("エラー", f"公費情報の検収処理に失敗しました。\n{e}")
            return False
        finally:
            self._prog_close()

    def run_ceiling(self):
        in_path = filedialog.askopenfilename(title="限度額情報CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        self._log(f"[限度額] 入力CSV: {in_path}")
        if not in_path:
            return False
        try:
            # ここではまず器だけ用意（後で COLUMNS_CEILING 等に差し替え）
            messagebox.showinfo("準備中", "限度額の検収CSV生成は現在実装中です。次のステップでロジックを追加します。")
            return False  # 生成未完了のため内容検収ボタンはまだ有効化しない
        except Exception as e:
            messagebox.showerror("エラー", f"限度額情報の検収処理に失敗しました。\n{e}")
            return False

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
            inspection.to_csv(missing_df, out_path)
            messagebox.showinfo("完了", f"未ヒット {len(missing_df)} 件を出力しました。\n{out_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"未ヒットリストの保存に失敗しました。\n{e}")

    def run_patient_content_check(self) -> bool:
        """患者情報の内容検収を実行（patient_content_check.py を呼び出す）。"""
        try:
            # 遅延インポート（起動時の ImportError を回避）
            from . import patient_content_check as _pcc
        except Exception as e:
            messagebox.showerror(
                "エラー",
                "患者情報の内容検収モジュール(ui/patient_content_check.py)の読み込みに失敗しました。\n"
                f"{e}\n\nファイル名・配置パスを確認してください。"
            )
            return False

        # 実行関数を候補名から解決
        func = None
        for name in ("run_patient_content_check", "run", "main"):
            f = getattr(_pcc, name, None)
            if callable(f):
                func = f
                break

        if func is None:
            messagebox.showerror(
                "エラー",
                "ui/patient_content_check.py に実行関数が見つかりません。\n"
                "次のいずれかの関数名で定義してください: run_patient_content_check / run / main"
            )
            return False

        try:
            maps = self._load_colmaps()
            preset = maps.get("patient")
            try:
                # 関数シグネチャに 'logger' 引数があるか簡易判定
                import inspect
                sig = inspect.signature(func)
                kwargs = {}
                if 'logger' in sig.parameters:
                    kwargs['logger'] = self._logger
                if 'preset' in sig.parameters:
                    kwargs['preset'] = preset
                ok = func(self.app, **kwargs) if kwargs else func(self.app)
            except Exception:
                # シグネチャ取得に失敗した場合は素直に2引数トライ→1引数トライ
                try:
                    ok = func(self.app, logger=self._logger, preset=preset)
                except Exception:
                    ok = func(self.app)
            return bool(ok)
        except Exception as e:
            messagebox.showerror("エラー", f"患者情報の内容検収でエラーが発生しました。\n{e}")
            return False

    def run_insurance_content_check(self) -> bool:
        try:
            # 遅延インポート
            from . import insurance_content_check as _icc
        except Exception as e:
            messagebox.showerror("エラー", f"保険情報の内容検収モジュールの読み込みに失敗しました。\n{e}")
            return False

        # 実行関数を解決
        func = None
        for name in ("run_insurance_content_check", "run", "main"):
            f = getattr(_icc, name, None)
            if callable(f):
                func = f; break
        if func is None:
            messagebox.showerror("エラー", "ui/insurance_content_check.py に実行関数が見つかりません。")
            return False

        # マッピングのプリセットをロード
        maps = self._load_colmaps()
        preset = maps.get("insurance")

        # 検収ページの移行日（共通入力）を取得
        mig = None
        try:
            mig = self._get_migration_date()  # 既存の共通関数（yyyymmdd を返す実装ならそのまま）
        except Exception:
            mig = None

        # 呼び出し（logger/preset/migration_date を渡す）
        try:
            return bool(func(self.app, logger=self._logger, preset=preset, migration_date=mig))
        except TypeError:
            # 旧シグネチャ互換
            return bool(func(self.app, logger=self._logger, preset=preset))

    def run_public_content_check(self) -> bool:
        """公費情報の内容検収を実行（public_content_check.py を呼び出す）。"""
        try:
            # 遅延インポート
            from . import public_content_check as _pcc
        except Exception as e:
            messagebox.showerror("エラー", f"公費情報の内容検収モジュールの読み込みに失敗しました。\n{e}")
            return False

        # 実行関数を解決（run_public_content_check / run / main の順で探す）
        func = None
        for name in ("run_public_content_check", "run", "main"):
            f = getattr(_pcc, name, None)
            if callable(f):
                func = f
                break
        if func is None:
            messagebox.showerror("エラー", "ui/public_content_check.py に実行関数が見つかりません。")
            return False

        # マッピングのプリセットをロード
        maps = self._load_colmaps()
        preset = maps.get("public")

        # 検収ページの移行日（共通入力）を取得（ファイル側が受け取らない実装でも後方互換で呼び出し可）
        mig = None
        try:
            mig = self._get_migration_date()
        except Exception:
            mig = None

        # 呼び出し（logger / preset / migration_date を可能なら渡す）
        try:
            return bool(func(self.app, logger=self._logger, preset=preset, migration_date=mig))
        except TypeError:
            # migration_date を受け取らない旧シグネチャ互換
            try:
                return bool(func(self.app, logger=self._logger, preset=preset))
            except TypeError:
                # logger/preset も受け取らない場合
                return bool(func(self.app))

    def run_ceiling_content_check(self) -> bool:
        messagebox.showinfo("未実装", "限度額の内容検収は現在未実装です。生成ロジックと併せて順次対応します。")
        return False