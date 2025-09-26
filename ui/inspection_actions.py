# ui/inspection_actions.py
from __future__ import annotations
from tkinter import filedialog, messagebox
from datetime import datetime as _dt
import pandas as pd
from pathlib import Path

from core.io_utils import CsvLoader
from core import inspection


class InspectionActions:
    """検収系のUIイベントを集約します。app（DataSurveyApp）に依存します。"""

    def __init__(self, app):
        self.app = app  # DataSurveyApp（_ask_inspection_colmap, _normalize_patient_number_for_match を利用）
        self.public_migration_yyyymmdd: str | None = None
        self.insurance_migration_yyyymmdd: str | None = None
        self._logger = None
        self._migration_provider = None  # callable that returns raw user input for migration date
        self._migration_yyyymmdd: str | None = None  # cached normalized yyyymmdd (shared for 保険/公費)

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
                # 氏名・カナ氏名両方空欄、カナ氏名が記号のみ、もしくは生年月日<1900-01-01、または患者番号が空欄 → 対象外
                name_col = colmap.get("患者氏名") or colmap.get("氏名")
                kana_col = colmap.get("患者氏名カナ") or colmap.get("カナ氏名")
                mask_name_empty = pd.Series([False] * len(df), index=df.index)
                mask_kana_empty = pd.Series([False] * len(df), index=df.index)
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
                if name_col and name_col in df.columns:
                    s_name = df[name_col]
                    mask_name_empty = (
                        s_name.isna() |
                        s_name.astype(str).str.strip().str.lower().isin(["", "nan", "none", "null"])
                    )
                if kana_col and kana_col in df.columns:
                    s_kana = df[kana_col]
                    mask_kana_empty = (
                        s_kana.isna() |
                        s_kana.astype(str).str.strip().str.lower().isin(["", "nan", "none", "null"])
                    )
                mask_both_empty = mask_name_empty & mask_kana_empty
                # カナ氏名が記号のみ（= カナ/ひらがな/長音符が1文字も無く、非空）
                import re, unicodedata
                mask_kana_symbols = pd.Series([False] * len(df), index=df.index)
                if kana_col and kana_col in df.columns:
                    s_kana = df[kana_col]
                    def _is_symbols_only(val: object) -> bool:
                        if val is None:
                            return False
                        s = unicodedata.normalize("NFKC", str(val)).strip()
                        if s == "":
                            return False  # 空欄は別ロジックで判定
                        # カナ/ひらがな/長音符のみを残す（スペースは除去）
                        kana_kept = re.sub(r"[^ーｰ\u30A0-\u30FF\u3040-\u309F]", "", s)
                        # 残らなければカナ成分ゼロ → 記号等のみ
                        return kana_kept == ""
                    mask_kana_symbols = s_kana.map(_is_symbols_only)
                # 生年月日 < 1900-01-01 → 対象外
                birth_col = colmap.get("生年月日")
                mask_birth_old = pd.Series([False] * len(df), index=df.index)
                if birth_col and birth_col in df.columns:
                    norm_birth = df[birth_col].map(lambda s: inspection._parse_date_any_to_yyyymmdd(str(s)) if s is not None else "")
                    def _is_old(yyyymmdd: str) -> bool:
                        if not yyyymmdd or len(yyyymmdd) != 8:
                            return False
                        return int(yyyymmdd[:4]) < 1900
                    mask_birth_old = norm_birth.map(_is_old)
                mask = mask_code_empty | mask_both_empty | mask_birth_old | mask_kana_symbols | mask_code_dup
                # 理由を付与
                reasons.loc[mask_code_empty] = reasons.loc[mask_code_empty].astype(str).str.cat(
                    pd.Series(["患者番号空欄"] * int(mask_code_empty.sum()), index=reasons.loc[mask_code_empty].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_both_empty] = reasons.loc[mask_both_empty].astype(str).str.cat(
                    pd.Series(["氏名・カナ氏名空欄"] * int(mask_both_empty.sum()), index=reasons.loc[mask_both_empty].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_kana_symbols] = reasons.loc[mask_kana_symbols].astype(str).str.cat(
                    pd.Series(["カナ氏名記号のみ"] * int(mask_kana_symbols.sum()), index=reasons.loc[mask_kana_symbols].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_birth_old] = reasons.loc[mask_birth_old].astype(str).str.cat(
                    pd.Series(["生年月日1900年未満"] * int(mask_birth_old.sum()), index=reasons.loc[mask_birth_old].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_code_dup] = reasons.loc[mask_code_dup].astype(str).str.cat(
                    pd.Series(["患者番号重複"] * int(mask_code_dup.sum()), index=reasons.loc[mask_code_dup].index),
                    sep=" / "
                ).str.strip(" /")

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

            elif key_mode == "public":
                # ルールは「負担者番号/受給者番号の空欄」で対象外（一次キー想定として '１' を優先）
                payer1_col = colmap.get("公費負担者番号１") or colmap.get("負担者番号")
                recip1_col = colmap.get("公費受給者番号１") or colmap.get("受給者番号")
                # --- 新ルール: 患者番号が空欄 ---
                code_col = colmap.get("患者番号")
                mask_code_empty = pd.Series([False] * len(df), index=df.index)
                if code_col and code_col in df.columns:
                    mask_code_empty = _digits_or_empty(df[code_col]).map(lambda s: s == "")
                # ---------------------------
                mask_payer1_empty = pd.Series([False] * len(df), index=df.index)
                mask_recip1_empty = pd.Series([False] * len(df), index=df.index)
                if payer1_col and payer1_col in df.columns:
                    mask_payer1_empty = _digits_or_empty(df[payer1_col]).map(lambda s: s == "")
                if recip1_col and recip1_col in df.columns:
                    mask_recip1_empty = _digits_or_empty(df[recip1_col]).map(lambda s: s == "")
                # --- 新ルール: 公費負担者番号 先頭2桁が 27 / 28（未使用の方別番号）は対象外 ---
                payer1_digits = pd.Series([""] * len(df), index=df.index, dtype="object")
                if payer1_col and payer1_col in df.columns:
                    payer1_digits = _digits_or_empty(df[payer1_col])
                mask_payer1_prefix_27_28 = payer1_digits.str[:2].isin(["27", "28"])
                # --- 新ルール: 公費の重複は（患者番号 + 公費負担者番号）で判定（先頭0無視） ---
                mask_combo_dup = pd.Series([False] * len(df), index=df.index)
                if code_col and code_col in df.columns and payer1_col and payer1_col in df.columns:
                    pat_norm_l = _digits_or_empty(df[code_col]).str.lstrip("0")
                    payer_norm_l = _digits_or_empty(df[payer1_col]).str.lstrip("0")
                    combo_keys = pd.Series(list(zip(pat_norm_l, payer_norm_l)), index=df.index)
                    mask_combo_dup = combo_keys.duplicated(keep=False) & (pat_norm_l != "") & (payer_norm_l != "")
                end1_col = (
                    colmap.get("公費終了日１")
                    or colmap.get("公費終了日")
                    or colmap.get("終了日")
                )

                # 期限切れ（データ移行日より前に終了している）を対象外に
                mask_expired = pd.Series([False] * len(df), index=df.index)
                mig_yyyymmdd = migration_yyyymmdd or self._get_migration_date()
                if end1_col and end1_col in df.columns and mig_yyyymmdd:
                    end_norm = df[end1_col].map(
                        lambda s: inspection._parse_date_any_to_yyyymmdd(str(s)) if s is not None else ""
                    )
                    # 「移行日より前」なら期限切れ扱い（= 移行時点で有効でない）
                    mask_expired = (end_norm != "") & (end_norm < str(mig_yyyymmdd))
                # --------------------------------------------------------------
                mask = (
                    mask_code_empty
                    | mask_payer1_empty
                    | mask_recip1_empty
                    | mask_combo_dup
                    | mask_payer1_prefix_27_28
                    | mask_expired
                )
                reasons.loc[mask_payer1_empty] = reasons.loc[mask_payer1_empty].astype(str).str.cat(pd.Series(["公費負担者番号空欄"] * int(mask_payer1_empty.sum()), index=reasons.loc[mask_payer1_empty].index), sep=" / ").str.strip(" /")
                reasons.loc[mask_recip1_empty] = reasons.loc[mask_recip1_empty].astype(str).str.cat(pd.Series(["公費受給者番号空欄"] * int(mask_recip1_empty.sum()), index=reasons.loc[mask_recip1_empty].index), sep=" / ").str.strip(" /")
                reasons.loc[mask_code_empty] = reasons.loc[mask_code_empty].astype(str).str.cat(
                    pd.Series(["患者番号空欄"] * int(mask_code_empty.sum()), index=reasons.loc[mask_code_empty].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_combo_dup] = reasons.loc[mask_combo_dup].astype(str).str.cat(
                    pd.Series(["患者番号+公費負担者番号重複"] * int(mask_combo_dup.sum()), index=reasons.loc[mask_combo_dup].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_payer1_prefix_27_28] = reasons.loc[mask_payer1_prefix_27_28].astype(str).str.cat(
                    pd.Series(["公費負担者番号先頭2桁が27/28(対象外)"] * int(mask_payer1_prefix_27_28.sum()),
                              index=reasons.loc[mask_payer1_prefix_27_28].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_expired] = reasons.loc[mask_expired].astype(str).str.cat(
                    pd.Series(["公費終了日が移行日より前(期限切れ)"] * int(mask_expired.sum()),
                            index=reasons.loc[mask_expired].index),
                    sep=" / "
                ).str.strip(" /")

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
        except Exception:
            # 失敗時は空のDF
            self._log(f"[{key_mode}] 対象外抽出でエラーが発生しました。空データを返します")
            return src.head(0)

    # === 共通ユーティリティ ===
    def _ask_and_save_missing_and_matched(self, *, src: pd.DataFrame, colmap: dict,
                                        out_df: pd.DataFrame, cfg: inspection.InspectionConfig,
                                        key_mode: str = "patient", out_dir: Path | None = None) -> dict:
        """
        既存検収CSVを選び、未ヒット/対象外/一致のみ を自動保存し、件数とパスを返す。
        途中での確認ダイアログは出さない。最後に呼び出し側でまとめて完了ダイアログを出す。
        """
        summary = {
            "matched_count": 0,
            "missing_count": 0,
            "excluded_count": 0,
            "matched_path": None,
            "missing_path": None,
            "excluded_path": None,
        }

        self._log(f"[{key_mode}] 突合開始")

        # 突合ファイル選択（これだけは必要）
        cmp_path = filedialog.askopenfilename(
            title="突合対象の検収用CSV（固定カラム）を選択してください",
            filetypes=[("CSV files", "*.csv")]
        )
        if not cmp_path:
            self._log(f"[{key_mode}] 突合CSV未選択のためスキップ")
            return summary
        else:
            self._log(f"[{key_mode}] 突合CSV: {cmp_path}")

        try:
            cmp_df = CsvLoader.read_csv_flex(cmp_path)
            self._log(f"[{key_mode}] 突合CSV読込: 列 {list(cmp_df.columns)} / 行数 {len(cmp_df)}")
        except Exception:
            return summary

        if "患者番号" not in cmp_df.columns:
            self._log(f"[{key_mode}] 突合CSVに患者番号がないためスキップ")
            return summary

        # 出力用ユーティリティ
        prefix = "保険" if key_mode == "insurance" else ("公費" if key_mode == "public" else "患者")
        today_tag = _dt.now().strftime("%Y%m%d")

        def _path_in_dir(name: str) -> Path:
            if out_dir:
                return (out_dir / name)
            return Path(name)

        # ------------- ここから算出ロジック（既存を流用/微調整） -------------
        # 照合キー設定
        sub_key_name_cmp = None  # 比較(CMP)側の副キー
        sub_key_name_src = None  # 元(SRC)側の副キー（マッピング）
        out_sub_col = None       # 出力(OUT)側の副キー（固定仕様名）

        from tkinter import messagebox as _mb  # 既存の警告にだけ使用

        if key_mode == "insurance":
            sub_key_name_src = colmap.get("保険者番号")
            out_sub_col = "保険者番号"
            sub_key_name_cmp = "保険者番号"
            if sub_key_name_cmp not in cmp_df.columns or not sub_key_name_src or sub_key_name_src not in src.columns:
                self._log(f"[{key_mode}] 副キー不足のためスキップ")
                return summary
        elif key_mode == "public":
            sub_key_name_src = colmap.get("公費負担者番号１") or colmap.get("負担者番号")
            out_sub_col = "公費負担者番号１"
            public_aliases_cmp = [
                "公費負担者番号１", "公費負担者番号1", "第１公費負担者番号", "第一公費負担者番号", "負担者番号",
            ]
            for cand in public_aliases_cmp:
                if cand in cmp_df.columns:
                    sub_key_name_cmp = cand
                    break
            if sub_key_name_cmp is None:
                # 最後の手段で選択ダイアログ（選択なしなら突合スキップ）
                try:
                    from .dialogs import ColumnSelectDialog
                    from pathlib import Path as _Path
                    dlg = ColumnSelectDialog(
                        self.app,
                        list(cmp_df.columns),
                        title=f"[{_Path(cmp_path).name}] 突合用の公費負担者番号列を選択"
                    )
                    sub_key_name_cmp = dlg.selected if hasattr(dlg, "selected") and dlg.selected else None
                except Exception:
                    sub_key_name_cmp = None
            if not sub_key_name_src or sub_key_name_src not in src.columns or not sub_key_name_cmp or sub_key_name_cmp not in cmp_df.columns:
                self._log(f"[{key_mode}] 副キー不足のためスキップ")
                return summary

        self._log(f"[{key_mode}] 副キー: src={sub_key_name_src} / out={out_sub_col} / cmp={sub_key_name_cmp}")

        # 患者番号の幅決定
        try:
            import unicodedata, re
            cmp_digits_pat = cmp_df["患者番号"].astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
            width_pat = int(cmp_digits_pat.str.len().max()) if cmp_digits_pat.notna().any() else cfg.patient_number_width
            if not width_pat or width_pat <= 0:
                width_pat = cfg.patient_number_width
        except Exception:
            width_pat = cfg.patient_number_width
        try:
            if "患者番号" in out_df.columns:
                import unicodedata, re
                out_digits_pat = out_df["患者番号"].astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
                out_max_pat = int(out_digits_pat.str.len().max()) if out_digits_pat.notna().any() else 0
                width_pat = max(width_pat, out_max_pat or 0) or width_pat
        except Exception:
            pass
        self._log(f"[{key_mode}] 患者番号幅: {width_pat}")

        # 副キーの幅
        width_sub = 0
        if key_mode in ("insurance", "public"):
            try:
                import unicodedata, re
                cmp_digits_sub = cmp_df[sub_key_name_cmp].astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
                width_sub = int(cmp_digits_sub.str.len().max()) if cmp_digits_sub.notna().any() else 0
            except Exception:
                width_sub = 0
            try:
                if out_sub_col and out_sub_col in out_df.columns:
                    import unicodedata, re
                    out_digits_sub = out_df[out_sub_col].astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
                    out_max_sub = int(out_digits_sub.str.len().max()) if out_digits_sub.notna().any() else 0
                    width_sub = max(width_sub, out_max_sub)
            except Exception:
                pass
        self._log(f"[{key_mode}] 副キ―幅: {width_sub}")

        # === 未ヒット（src -> cmp）
        missing_df = pd.DataFrame()
        src_code_col = colmap.get("患者番号")
        if src_code_col and src_code_col in src.columns:
            if key_mode == "patient":
                src_codes_norm = self._normalize_codes(src[src_code_col].astype(str), width_pat, mode="zfill")
                cmp_codes_norm = self._normalize_codes(cmp_df["患者番号"].astype(str), width_pat, mode="zfill")
                cmp_set = set(cmp_codes_norm.loc[cmp_codes_norm != ""])
                mask_missing = (src_codes_norm != "") & (~src_codes_norm.isin(cmp_set))
                if mask_missing.any():
                    missing_df = src.loc[mask_missing].copy()
                    missing_df.insert(0, "__正規化患者番号__", src_codes_norm.loc[mask_missing])
                else:
                    # フォールバック：先頭0無視
                    src_ls = self._normalize_codes(src[src_code_col].astype(str), width_pat, mode="lstrip")
                    cmp_ls = self._normalize_codes(cmp_df["患者番号"].astype(str), width_pat, mode="lstrip")
                    cmp_set_ls = set(cmp_ls.loc[cmp_ls != ""])
                    mask2 = (src_ls != "") & (~src_ls.isin(cmp_set_ls))
                    if mask2.any():
                        missing_df = src.loc[mask2].copy()
                        missing_df.insert(0, "__正規化患者番号__", src_ls.loc[mask2])
            else:
                # 複合キー厳密
                src_keys_z = set(self._make_composite_keys(src, src_code_col, sub_key_name_src, width_pat, width_sub, mode="zfill"))
                cmp_keys_z = set(self._make_composite_keys(cmp_df, "患者番号", sub_key_name_cmp, width_pat, width_sub, mode="zfill"))
                miss_keys = [k for k in src_keys_z if k[0] != "" and k[1] != "" and k not in cmp_keys_z]
                if miss_keys:
                    src_pat_z = self._normalize_codes(src[src_code_col].astype(str), width_pat, mode="zfill")
                    src_sub_z = self._normalize_codes(src[sub_key_name_src].astype(str), width_sub, mode="zfill")
                    key_series = list(zip(src_pat_z, src_sub_z))
                    mask_missing = pd.Series([ks in miss_keys for ks in key_series], index=src.index)
                    if mask_missing.any():
                        missing_df = src.loc[mask_missing].copy()
                        missing_df.insert(0, "__正規化副キー__", src_sub_z.loc[mask_missing])
                        missing_df.insert(0, "__正規化患者番号__", src_pat_z.loc[mask_missing])
                # フォールバック：先頭0無視
                if missing_df.empty:
                    src_keys_l = set(self._make_composite_keys(src, src_code_col, sub_key_name_src, width_pat, width_sub, mode="lstrip"))
                    cmp_keys_l = set(self._make_composite_keys(cmp_df, "患者番号", sub_key_name_cmp, width_pat, width_sub, mode="lstrip"))
                    miss_keys2 = [k for k in src_keys_l if k[0] != "" and k[1] != "" and k not in cmp_keys_l]
                    if miss_keys2:
                        src_pat_l = self._normalize_codes(src[src_code_col].astype(str), width_pat, mode="lstrip")
                        src_sub_l = self._normalize_codes(src[sub_key_name_src].astype(str), width_sub, mode="lstrip")
                        key_series2 = list(zip(src_pat_l, src_sub_l))
                        mask_missing2 = pd.Series([ks in miss_keys2 for ks in key_series2], index=src.index)
                        if mask_missing2.any():
                            missing_df = src.loc[mask_missing2].copy()
                            missing_df.insert(0, "__正規化副キー__", src_sub_l.loc[mask_missing2])
                            missing_df.insert(0, "__正規化患者番号__", src_pat_l.loc[mask_missing2])
        self._log(f"[{key_mode}] 未ヒット算出: {len(missing_df)}件")

        # === 対象外（仕様ルール + 未ヒットEligible編入）
        excluded_df = pd.DataFrame()
        try:
            mig = self._get_migration_date()
            excluded_df = self._extract_excluded(src=src, colmap=colmap, key_mode=key_mode, migration_yyyymmdd=mig)

            # 一致マスク（src基準）
            matched_mask_src = pd.Series([False] * len(src), index=src.index)
            try:
                src_code_col = colmap.get("患者番号")
                if src_code_col and src_code_col in src.columns:
                    if key_mode == "patient":
                        src_codes_norm_m = self._normalize_codes(src[src_code_col].astype(str), width_pat, mode="zfill")
                        cmp_codes_norm_m = self._normalize_codes(cmp_df["患者番号"].astype(str), width_pat, mode="zfill")
                        cmp_set_m = set(cmp_codes_norm_m.loc[cmp_codes_norm_m != ""])
                        matched_mask_src = (src_codes_norm_m != "") & (src_codes_norm_m.isin(cmp_set_m))
                    else:
                        if key_mode == "insurance":
                            sub_src = colmap.get("保険者番号")
                            sub_cmp = "保険者番号"
                        else:
                            sub_src = colmap.get("公費負担者番号１") or colmap.get("負担者番号")
                            sub_cmp = sub_key_name_cmp
                        if sub_src and sub_src in src.columns and sub_cmp in cmp_df.columns:
                            src_keys_z = self._make_composite_keys(src, src_code_col, sub_src, width_pat, width_sub, mode="zfill")
                            cmp_keys_z = self._make_composite_keys(cmp_df, "患者番号", sub_cmp, width_pat, width_sub, mode="zfill")
                            src_keys_z = [(p, s) for (p, s) in src_keys_z if p and s]
                            cmp_key_set = set([(p, s) for (p, s) in cmp_keys_z if p and s])
                            matched_mask_src = pd.Series([ (k[0] and k[1] and (k in cmp_key_set)) for k in self._make_composite_keys(src, src_code_col, sub_src, width_pat, width_sub, mode="zfill") ], index=src.index)
                # 対象外から一致分を除外
                if excluded_df is not None and not excluded_df.empty and matched_mask_src.any():
                    excluded_df = excluded_df.loc[~excluded_df.index.isin(src.index[matched_mask_src])]
            except Exception:
                pass

            # Eligible 未ヒットを「未分類（未ヒット・要ルール）」として対象外に編入
            try:
                excluded_idx_now = set(excluded_df.index) if excluded_df is not None else set()
                eligible_mask_now = ~src.index.to_series().isin(excluded_idx_now)
                unmatched_eligible_mask = eligible_mask_now & (~matched_mask_src)
                if unmatched_eligible_mask.any():
                    unmatched_df = src.loc[unmatched_eligible_mask].copy()
                    # 正規化キー付与
                    try:
                        src_code_col = colmap.get("患者番号")
                        if src_code_col and src_code_col in src.columns:
                            pat_norm = self._normalize_codes(src[src_code_col].astype(str), width_pat, mode="zfill")
                            unmatched_df.insert(0, "__正規化患者番号__", pat_norm.loc[unmatched_eligible_mask])
                        if key_mode in ("insurance", "public"):
                            if key_mode == "insurance":
                                sub_src = colmap.get("保険者番号")
                            else:
                                sub_src = colmap.get("公費負担者番号１") or colmap.get("負担者番号")
                            if sub_src and sub_src in src.columns:
                                sub_norm = self._normalize_codes(src[sub_src].astype(str), width_sub, mode="zfill")
                                unmatched_df.insert(1, "__正規化副キー__", sub_norm.loc[unmatched_eligible_mask])
                    except Exception:
                        pass
                    unmatched_df.insert(0, "__対象外理由__", "未分類（未ヒット・要ルール）")
                    if excluded_df is None or excluded_df.empty:
                        excluded_df = unmatched_df
                    else:
                        excluded_df = pd.concat([excluded_df, unmatched_df], axis=0)
            except Exception:
                pass
        except Exception:
            excluded_df = pd.DataFrame()
        self._log(f"[{key_mode}] 対象外算出: {len(excluded_df)}件")

        # === 一致のみ（out_df基準でフィルタ）
        filtered_out_df = pd.DataFrame()
        try:
            if "患者番号" in out_df.columns:
                if key_mode == "patient":
                    out_codes_norm = self._normalize_codes(out_df["患者番号"].astype(str), width_pat, mode="zfill")
                    cmp_codes_norm = self._normalize_codes(cmp_df["患者番号"].astype(str), width_pat, mode="zfill")
                    cmp_set = set(cmp_codes_norm.loc[cmp_codes_norm != ""])
                    matched_mask = (out_codes_norm != "") & (out_codes_norm.isin(cmp_set))
                    filtered_out_df = out_df.loc[matched_mask].copy()
                    if filtered_out_df.empty:
                        out_lstrip = self._normalize_codes(out_df["患者番号"].astype(str), width_pat, mode="lstrip")
                        cmp_lstrip = self._normalize_codes(cmp_df["患者番号"].astype(str), width_pat, mode="lstrip")
                        cmp_set2 = set(cmp_lstrip.loc[cmp_lstrip != ""])
                        matched2 = (out_lstrip != "") & (out_lstrip.isin(cmp_set2))
                        if matched2.any():
                            filtered_out_df = out_df.loc[matched2].copy()
                else:
                    if not out_sub_col or out_sub_col not in out_df.columns:
                        self._log(f"[{key_mode}] 副キー不足のためスキップ")
                        return summary
                    out_keys_z = self._make_composite_keys(out_df, "患者番号", out_sub_col, width_pat, width_sub, mode="zfill")
                    cmp_keys_z = self._make_composite_keys(cmp_df, "患者番号", sub_key_name_cmp, width_pat, width_sub, mode="zfill")
                    out_keys_z = [(p, s) for (p, s) in out_keys_z if p and s]
                    cmp_keys_z = [(p, s) for (p, s) in cmp_keys_z if p and s]
                    matched_keys = set(out_keys_z) & set(cmp_keys_z)
                    if matched_keys:
                        out_pat_z = self._normalize_codes(out_df["患者番号"].astype(str), width_pat, mode="zfill")
                        out_sub_z = self._normalize_codes(out_df[out_sub_col].astype(str), width_sub, mode="zfill")
                        key_series = list(zip(out_pat_z, out_sub_z))
                        matched_mask = pd.Series([ks in matched_keys for ks in key_series], index=out_df.index)
                        filtered_out_df = out_df.loc[matched_mask].copy()
                    else:
                        out_keys_l = self._make_composite_keys(out_df, "患者番号", out_sub_col, width_pat, width_sub, mode="lstrip")
                        cmp_keys_l = self._make_composite_keys(cmp_df, "患者番号", sub_key_name_cmp, width_pat, width_sub, mode="lstrip")
                        out_keys_l = [(p, s) for (p, s) in out_keys_l if p and s]
                        cmp_keys_l = [(p, s) for (p, s) in cmp_keys_l if p and s]
                        matched_keys2 = set(out_keys_l) & set(cmp_keys_l)
                        if matched_keys2:
                            out_pat_l = self._normalize_codes(out_df["患者番号"].astype(str), width_pat, mode="lstrip")
                            out_sub_l = self._normalize_codes(out_df[out_sub_col].astype(str), width_sub, mode="lstrip")
                            key_series2 = list(zip(out_pat_l, out_sub_l))
                            matched_mask2 = pd.Series([ks in matched_keys2 for ks in key_series2], index=out_df.index)
                            filtered_out_df = out_df.loc[matched_mask2].copy()

                # 重複除去
                if not filtered_out_df.empty:
                    if key_mode in ("insurance", "public"):
                        dedup_cols = ["患者番号", out_sub_col]
                        if all(c in filtered_out_df.columns for c in dedup_cols):
                            filtered_out_df = filtered_out_df.drop_duplicates(subset=dedup_cols, keep="first")
                    elif key_mode == "patient" and "患者番号" in filtered_out_df.columns:
                        filtered_out_df = filtered_out_df.drop_duplicates(subset=["患者番号"], keep="first")
        except Exception:
            filtered_out_df = pd.DataFrame()
        self._log(f"[{key_mode}] 一致のみ算出: {len(filtered_out_df)}件")
        # ------------- 算出ここまで -------------

        # ------------- 自動保存 -------------
        try:
            miss_path = _path_in_dir(f"{prefix}_未ヒット_{today_tag}.csv")
            inspection.to_csv(missing_df, str(miss_path))
            summary["missing_count"] = len(missing_df)
            summary["missing_path"]  = str(miss_path)
            self._log(f"[{key_mode}] 未ヒット出力: {miss_path}")
        except Exception:
            self._log(f"[{key_mode}] 未ヒット出力に失敗しました")
            pass

        try:
            ex_path = _path_in_dir(f"{prefix}_対象外_{today_tag}.csv")
            inspection.to_csv(excluded_df, str(ex_path))
            summary["excluded_count"] = len(excluded_df)
            summary["excluded_path"]  = str(ex_path)
            self._log(f"[{key_mode}] 対象外出力: {ex_path}")
        except Exception:
            self._log(f"[{key_mode}] 対象外出力に失敗しました")
            pass

        try:
            matched_path = _path_in_dir(f"{prefix}_検収_一致のみ_{today_tag}.csv")
            inspection.to_csv(filtered_out_df, str(matched_path))
            summary["matched_count"] = len(filtered_out_df)
            summary["matched_path"]  = str(matched_path)
            self._log(f"[{key_mode}] 一致のみ出力: {matched_path}")
        except Exception:
            self._log(f"[{key_mode}] 一致のみ出力に失敗しました")
            pass
        # ------------- 自動保存ここまで -------------

        return summary

    # === 各アクション ===
    def run_patient(self):
        in_path = filedialog.askopenfilename(title="患者情報CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        self._log(f"[患者] 入力CSV: {in_path}")
        if not in_path:
            return
        try:
            src = CsvLoader.read_csv_flex(in_path)
            out_dir = self._prepare_output_dir(in_path, "patient")
            self._log(f"[患者] 出力先ディレクトリ: {out_dir}")
            colmap = self.app._ask_inspection_colmap(src, required_cols=list(inspection.COLUMNS_PATIENT))
            if colmap is None:
                return
            self._log(f"[患者] マッピング完了: {colmap}")
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
                return
            inspection.to_csv(out_df, out_path)
            self._log(f"[患者] 検収CSV出力: {out_path} (行数 {len(out_df)})")

            summary = self._ask_and_save_missing_and_matched(
                src=src, colmap=colmap, out_df=out_df, cfg=cfg, key_mode="patient", out_dir=out_dir
            )
            self._log(f"[患者] 一致のみ: {summary.get('matched_count', 0)} → {summary.get('matched_path')}")
            self._log(f"[患者] 未ヒット: {summary.get('missing_count', 0)} → {summary.get('missing_path')}")
            self._log(f"[患者] 対象外: {summary.get('excluded_count', 0)} → {summary.get('excluded_path')}")
            messagebox.showinfo(
                "完了",
                f"出力が完了しました。\n一致: {summary.get('matched_count', 0)} 件 / 未ヒット: {summary.get('missing_count', 0)} 件 / 対象外: {summary.get('excluded_count', 0)} 件"
            )
        except Exception as e:
            messagebox.showerror("エラー", f"検収処理中に失敗しました。\n{e}")

    def run_insurance(self):
        in_path = filedialog.askopenfilename(title="保険情報の入力CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        self._log(f"[保険] 入力CSV: {in_path}")
        if not in_path:
            return
        try:
            src = CsvLoader.read_csv_flex(in_path)
            out_dir = self._prepare_output_dir(in_path, "insurance")
            self._log(f"[保険] 出力先ディレクトリ: {out_dir}")
            required_cols = list(inspection.COLUMNS_INSURANCE) + ["保険終了日"]
            colmap = self.app._ask_inspection_colmap(src, required_cols=required_cols)
            if colmap is None:
                return
            self._log(f"[保険] マッピング完了: {colmap}")
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
                return
            inspection.to_csv(out_df, out_path)
            self._log(f"[保険] 検収CSV出力: {out_path} (行数 {len(out_df)})")

            summary = self._ask_and_save_missing_and_matched(
                src=src, colmap=colmap, out_df=out_df, cfg=cfg, key_mode="insurance", out_dir=out_dir
            )
            self._log(f"[保険] 一致のみ: {summary.get('matched_count', 0)} → {summary.get('matched_path')}")
            self._log(f"[保険] 未ヒット: {summary.get('missing_count', 0)} → {summary.get('missing_path')}")
            self._log(f"[保険] 対象外: {summary.get('excluded_count', 0)} → {summary.get('excluded_path')}")
            messagebox.showinfo(
                "完了",
                f"出力が完了しました。\n一致: {summary.get('matched_count', 0)} 件 / 未ヒット: {summary.get('missing_count', 0)} 件 / 対象外: {summary.get('excluded_count', 0)} 件"
            )
        except Exception as e:
            messagebox.showerror("エラー", f"保険情報の検収処理に失敗しました。\n{e}")

    def run_public(self):
        in_path = filedialog.askopenfilename(title="公費情報の入力CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        self._log(f"[公費] 入力CSV: {in_path}")
        if not in_path:
            return
        try:
            src = CsvLoader.read_csv_flex(in_path)
            out_dir = self._prepare_output_dir(in_path, "public")
            self._log(f"[公費] 出力先ディレクトリ: {out_dir}")
            required_cols = list(inspection.COLUMNS_PUBLIC) + ["公費終了日１"]
            colmap = self.app._ask_inspection_colmap(src, required_cols=required_cols)
            if colmap is None:
                return
            self._log(f"[公費] マッピング完了: {colmap}")
            # 共通の移行日（検収ページ右上の入力欄から取得／キャッシュ）
            mig = self._get_migration_date()
            self.public_migration_yyyymmdd = mig  # 後方互換
            self._log(f"[公費] 移行日: {mig}")
            cfg = inspection.InspectionConfig(patient_number_width=10)
            out_df = inspection.build_inspection_df(src, colmap, cfg, target_columns=inspection.COLUMNS_PUBLIC)

            default_name = f"公費_検収_{_dt.now().strftime('%Y%m%d')}.csv"
            out_path = filedialog.asksaveasfilename(
                title="公費情報 検収CSVを保存",
                defaultextension=".csv",
                initialfile=default_name,
                initialdir=str(out_dir),
                filetypes=[("CSV files", "*.csv")]
            )
            if not out_path:
                return
            inspection.to_csv(out_df, out_path)
            self._log(f"[公費] 検収CSV出力: {out_path} (行数 {len(out_df)})")

            summary = self._ask_and_save_missing_and_matched(
                src=src, colmap=colmap, out_df=out_df, cfg=cfg, key_mode="public", out_dir=out_dir
            )
            self._log(f"[公費] 一致のみ: {summary.get('matched_count', 0)} → {summary.get('matched_path')}")
            self._log(f"[公費] 未ヒット: {summary.get('missing_count', 0)} → {summary.get('missing_path')}")
            self._log(f"[公費] 対象外: {summary.get('excluded_count', 0)} → {summary.get('excluded_path')}")
            messagebox.showinfo(
                "完了",
                f"出力が完了しました。\n一致: {summary.get('matched_count', 0)} 件 / 未ヒット: {summary.get('missing_count', 0)} 件 / 対象外: {summary.get('excluded_count', 0)} 件"
            )
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
            inspection.to_csv(missing_df, out_path)
            messagebox.showinfo("完了", f"未ヒット {len(missing_df)} 件を出力しました。\n{out_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"未ヒットリストの保存に失敗しました。\n{e}")

