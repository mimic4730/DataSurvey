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
                payer_col = colmap.get("保険者番号")
                cardno_col = colmap.get("保険証番号")
                # --- 新ルール: 患者番号が空欄 ---
                code_col = colmap.get("患者番号")
                mask_code_empty = pd.Series([False] * len(df), index=df.index)
                if code_col and code_col in df.columns:
                    mask_code_empty = _digits_or_empty(df[code_col]).map(lambda s: s == "")
                # ---------------------------
                # --- 新ルール: 患者番号の重複（先頭0を無視して比較） ---
                mask_code_dup = pd.Series([False] * len(df), index=df.index)
                if code_col and code_col in df.columns:
                    code_digits = _digits_or_empty(df[code_col]).str.lstrip("0")
                    mask_code_dup = code_digits.duplicated(keep=False) & (code_digits != "")
                # ---------------------------
                # 保険者番号の正規化: 7桁→頭0で8桁, 5桁→頭0で6桁。他は桁数維持（数字のみ残す）
                payer_norm = None
                if payer_col and payer_col in df.columns:
                    import unicodedata, re
                    payer_digits = df[payer_col].astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
                    def _pad_payer(s: str) -> str:
                        if not s:
                            return ""
                        n = len(s)
                        if n == 7:
                            return s.zfill(8)
                        if n == 5:
                            return s.zfill(6)
                        return s
                    payer_norm = payer_digits.map(_pad_payer)

                mask_payer_empty = pd.Series([False] * len(df), index=df.index)
                mask_payer_invalid_len = pd.Series([False] * len(df), index=df.index)
                mask_cardno_empty = pd.Series([False] * len(df), index=df.index)

                if payer_norm is not None:
                    mask_payer_empty = payer_norm.map(lambda s: s == "")
                    # 空欄以外で 6桁・8桁以外は対象外
                    mask_payer_invalid_len = (payer_norm != "") & (~payer_norm.map(lambda s: len(s) in (6, 8)))

                if cardno_col and cardno_col in df.columns:
                    mask_cardno_empty = (
                        df[cardno_col].isna() |
                        df[cardno_col].astype(str).str.strip().str.lower().isin(["", "nan", "none", "null"])
                    )

                # --- 新ルール: 保険者番号8桁で頭が39の場合、保険証番号は8桁のみ対象 ---
                cardno_digits = None
                if cardno_col and cardno_col in df.columns:
                    import unicodedata, re
                    cardno_digits = df[cardno_col].astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
                else:
                    cardno_digits = pd.Series([""] * len(df), index=df.index, dtype="object")

                mask_payer_39_cardno_not8 = pd.Series([False] * len(df), index=df.index)
                if payer_norm is not None and cardno_digits is not None:
                    mask_payer_39_cardno_not8 = (
                        (payer_norm.str.len() == 8)
                        & (payer_norm.str.startswith("39"))
                        & (cardno_digits.str.len() != 8)
                    )
                # ----------------------------------------------------------

                # --- 新ルール: 保険者番号8桁の場合の先頭2桁チェック ---
                mask_payer_prefix_ng = pd.Series([False] * len(df), index=df.index)
                if payer_norm is not None:
                    allowed_prefixes = {"01", "02", "03", "04", "06", "07", "39", "31", "32", "33", "34", "63", "72", "73", "74", "75"}
                    mask_payer_prefix_ng = (
                        (payer_norm.str.len() == 8) &
                        (~payer_norm.str[:2].isin(allowed_prefixes))
                    )
                # ----------------------------------------------------------

                # --- 新ルール: 保険者番号8桁の3-4桁目(都道府県コード)が01..47以外は対象外 ---
                mask_prefcode_invalid = pd.Series([False] * len(df), index=df.index)
                if payer_norm is not None:
                    allowed_pref_codes = {
                        "01","02","03","04","05","06","07","08","09","10",
                        "11","12","13","14","15","16","17","18","19","20",
                        "21","22","23","24","25","26","27","28","29","30",
                        "31","32","33","34","35","36","37","38","39","40",
                        "41","42","43","44","45","46","47"
                    }
                    mask_prefcode_invalid = (
                        (payer_norm.str.len() == 8) &
                        (~payer_norm.str[2:4].isin(allowed_pref_codes))
                    )
                # ----------------------------------------------------------

                # --- 新ルール: 保険終了日による期限切れ判定 ---
                end_col = colmap.get("保険終了日") or colmap.get("終了日")
                mask_expired = pd.Series([False] * len(df), index=df.index)
                mig_yyyymmdd = migration_yyyymmdd or self._get_migration_date()
                if end_col and end_col in df.columns and mig_yyyymmdd:
                    end_norm = df[end_col].map(lambda s: inspection._parse_date_any_to_yyyymmdd(str(s)) if s is not None else "")
                    mask_expired = (end_norm != "") & (end_norm < str(mig_yyyymmdd))
                # ----------------------------------------------------------

                # --- 新ルール: 75歳以上 & 法別≠39 を対象外（移行日基準の年齢） ---
                mask_age75_law_not39 = pd.Series([False] * len(df), index=df.index)
                try:
                    birth_col = colmap.get("生年月日")
                    mig_yyyymmdd = migration_yyyymmdd or self._get_migration_date()
                    if birth_col and birth_col in df.columns and mig_yyyymmdd:
                        # 生年月日を YYYYMMDD に正規化
                        birth_norm = df[birth_col].map(lambda v: inspection._parse_date_any_to_yyyymmdd(v))
                        # 年齢計算（移行日基準）
                        from datetime import date
                        my, mm, md = int(mig_yyyymmdd[0:4]), int(mig_yyyymmdd[4:6]), int(mig_yyyymmdd[6:8])
                        mig_day = date(my, mm, md)

                        def _is_75_or_over(b: str) -> bool:
                            if not b or len(b) != 8:
                                return False
                            try:
                                by, bm, bd = int(b[0:4]), int(b[4:6]), int(b[6:8])
                                bd = date(by, bm, bd)
                                age = mig_day.year - bd.year - ((mig_day.month, mig_day.day) < (bd.month, bd.day))
                                return age >= 75
                            except Exception:
                                return False

                        is75 = birth_norm.map(_is_75_or_over)

                        # 保険者番号先頭2桁が "39" 以外
                        if payer_norm is not None:
                            law2 = payer_norm.map(lambda s: s[:2] if isinstance(s, str) and len(s) >= 2 else "")
                        else:
                            # フォールバック（正規化が無ければその場で数字抽出）
                            import re, unicodedata
                            payer_col_fallback = colmap.get("保険者番号")
                            if payer_col_fallback and payer_col_fallback in df.columns:
                                payer_digits = df[payer_col_fallback].astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
                                law2 = payer_digits.map(lambda s: s[:2] if s else "")
                            else:
                                law2 = pd.Series([""] * len(df), index=df.index, dtype="object")

                        not39 = law2.map(lambda s: (s != "") and (s != "39"))

                        mask_age75_law_not39 = is75 & not39
                except Exception:
                    # 判定に失敗した場合はこのルールを適用しない
                    mask_age75_law_not39 = pd.Series([False] * len(df), index=df.index)

                mask = (
                    mask_code_empty
                    | mask_code_dup
                    | mask_payer_empty
                    | mask_payer_invalid_len
                    | mask_cardno_empty
                    | mask_payer_39_cardno_not8
                    | mask_payer_prefix_ng
                    | mask_prefcode_invalid
                    | mask_expired
                    | mask_age75_law_not39
                )
                reasons.loc[mask_payer_empty] = reasons.loc[mask_payer_empty].astype(str).str.cat(
                    pd.Series(["保険者番号空欄"] * int(mask_payer_empty.sum()), index=reasons.loc[mask_payer_empty].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_payer_invalid_len] = reasons.loc[mask_payer_invalid_len].astype(str).str.cat(
                    pd.Series(["保険者番号桁数不正(6または8桁以外)"] * int(mask_payer_invalid_len.sum()), index=reasons.loc[mask_payer_invalid_len].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_cardno_empty] = reasons.loc[mask_cardno_empty].astype(str).str.cat(
                    pd.Series(["保険証番号空欄"] * int(mask_cardno_empty.sum()), index=reasons.loc[mask_cardno_empty].index),
                    sep=" / "
                ).str.strip(" /")
                # 新ルール理由付与: mask_payer_39_cardno_not8
                reasons.loc[mask_payer_39_cardno_not8] = reasons.loc[mask_payer_39_cardno_not8].astype(str).str.cat(
                    pd.Series(["保険者番号(39xxxxxx)は保険証番号8桁のみ対象"] * int(mask_payer_39_cardno_not8.sum()),
                              index=reasons.loc[mask_payer_39_cardno_not8].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_code_empty] = reasons.loc[mask_code_empty].astype(str).str.cat(
                    pd.Series(["患者番号空欄"] * int(mask_code_empty.sum()), index=reasons.loc[mask_code_empty].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_code_dup] = reasons.loc[mask_code_dup].astype(str).str.cat(
                    pd.Series(["患者番号重複"] * int(mask_code_dup.sum()), index=reasons.loc[mask_code_dup].index),
                    sep=" / "
                ).str.strip(" /")
                # 新ルール理由付与: mask_payer_prefix_ng
                reasons.loc[mask_payer_prefix_ng] = reasons.loc[mask_payer_prefix_ng].astype(str).str.cat(
                    pd.Series(["保険者番号先頭2桁が対象外"] * int(mask_payer_prefix_ng.sum()), index=reasons.loc[mask_payer_prefix_ng].index),
                    sep=" / "
                ).str.strip(" /")
                # 新ルール理由付与: mask_prefcode_invalid
                reasons.loc[mask_prefcode_invalid] = reasons.loc[mask_prefcode_invalid].astype(str).str.cat(
                    pd.Series(["保険者番号都道府県コード不正(3-4桁)"] * int(mask_prefcode_invalid.sum()),
                              index=reasons.loc[mask_prefcode_invalid].index),
                    sep=" / "
                ).str.strip(" /")
                # 新ルール理由付与: mask_expired
                reasons.loc[mask_expired] = reasons.loc[mask_expired].astype(str).str.cat(
                    pd.Series(["保険終了日が移行日より前(期限切れ)"] * int(mask_expired.sum()),
                              index=reasons.loc[mask_expired].index),
                    sep=" / "
                ).str.strip(" /")
                # 新ルール理由付与: mask_age75_law_not39
                reasons.loc[mask_age75_law_not39] = reasons.loc[mask_age75_law_not39].astype(str).str.cat(
                    pd.Series(["75歳以上・法別≠39"] * int(mask_age75_law_not39.sum()), index=reasons.loc[mask_age75_law_not39].index),
                    sep=" / "
                ).str.strip(" /")

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
        }

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
            sub_key_name_src = colmap.get("保険者番号")
            out_sub_col = "保険者番号"
            sub_key_name_cmp = "保険者番号" if "保険者番号" in cmp_columns else None
            if sub_key_name_cmp is None:
                self._log(f"[{key_mode}] 突合CSVに保険者番号が見つかりません")
            if not sub_key_name_src or sub_key_name_src not in src.columns or not sub_key_name_cmp:
                self._log(f"[{key_mode}] 副キー不足のためスキップ")
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
        if key_mode in ("insurance", "public") and sub_key_name_cmp:
            usecols.append(sub_key_name_cmp)

        cmp_keys_zfill_set = set()
        cmp_keys_lstrip_set = set()
        total_rows = 0
        self._prog_set("検収中…（突合キーを構築中）")

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
                sub = chunk[sub_key_name_cmp] if (key_mode in ("insurance", "public") and sub_key_name_cmp in chunk.columns) else None

                kz = _make_key_str(pat, sub, width_pat, width_sub, "zfill", "cmp_pat", "cmp_sub" if sub is not None else None)
                kl = _make_key_str(pat, sub, width_pat, width_sub, "lstrip", "cmp_pat", "cmp_sub" if sub is not None else None)

                # 空キーを除去
                if sub is None:
                    kz = kz.loc[kz != ""]
                    kl = kl.loc[kl != ""]
                else:
                    # 複合キーは患者 or 副のどちらかが空なら除外
                    # （zfill/lstrip 両方で同じロジック）
                    mask_valid = (~_digits(pat, "cmp_pat").eq("")) & (~_digits(sub, "cmp_sub").eq(""))
                    kz = kz.loc[mask_valid]
                    kl = kl.loc[mask_valid]

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
                        if key_mode in ("insurance", "public"):
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
                        if key_mode in ("insurance", "public") and out_sub_col in filtered_out_df.columns:
                            filtered_out_df = filtered_out_df.drop_duplicates(subset=["患者番号", out_sub_col], keep="first")
                        elif key_mode == "patient":
                            filtered_out_df = filtered_out_df.drop_duplicates(subset=["患者番号"], keep="first")
        except Exception:
            filtered_out_df = pd.DataFrame()
        self._log(f"[{key_mode}] 一致のみ算出: {len(filtered_out_df)}件")

        # ---- 10) 出力 ----
        self._prog_set("書き出し中…（CSV出力）")
        prefix = "保険" if key_mode == "insurance" else ("公費" if key_mode == "public" else "患者")
        today_tag = _dt.now().strftime("%Y%m%d")

        def _path_in_dir(name: str) -> Path:
            return (out_dir / name) if out_dir else Path(name)

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
            required_cols = list(inspection.COLUMNS_INSURANCE) + ["保険終了日", "生年月日"]
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