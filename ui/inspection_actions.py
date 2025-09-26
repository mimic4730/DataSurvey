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
        self.public_migration_yyyymmdd: str | None = None

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

                mask = (
                    mask_code_empty
                    | mask_code_dup
                    | mask_payer_empty
                    | mask_payer_invalid_len
                    | mask_cardno_empty
                    | mask_payer_39_cardno_not8
                    | mask_payer_prefix_ng
                    | mask_prefcode_invalid
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
                mig_yyyymmdd = migration_yyyymmdd or self.public_migration_yyyymmdd
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
            return excluded
        except Exception:
            # 失敗時は空のDF
            return src.head(0)

    # === 共通ユーティリティ ===
    def _ask_and_save_missing_and_matched(self, *, src: pd.DataFrame, colmap: dict,
                                          out_df: pd.DataFrame, cfg: inspection.InspectionConfig,
                                          key_mode: str = "patient") -> None:
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

        # 照合キー設定
        sub_key_name_cmp = None  # 突合対象(CMP)側の副キー列名
        sub_key_name_src = None  # 元CSV(SRC)側の副キー列名（マッピング）
        out_sub_col = None       # 検収出力(OUT)側で使う副キー列名（固定仕様名）

        if key_mode == "insurance":
            sub_key_name_src = colmap.get("保険者番号")
            out_sub_col = "保険者番号"
            # 比較側は固定名が基本だが、将来の互換性のためにフォールバックなし
            sub_key_name_cmp = "保険者番号"
            if sub_key_name_cmp not in cmp_df.columns or not sub_key_name_src or sub_key_name_src not in src.columns:
                _mb.showerror("エラー", "保険者番号が比較/元CSVのどちらかで見つかりません。マッピングとCSVを確認してください。")
                return

        elif key_mode == "public":
            # 元CSV側（ユーザーがマッピングした列）
            sub_key_name_src = colmap.get("公費負担者番号１") or colmap.get("負担者番号")
            out_sub_col = "公費負担者番号１"  # 出力仕様は固定

            # 比較CSV側は列名のゆらぎに幅広く対応
            public_aliases_cmp = [
                "公費負担者番号１",  # 全角1
                "公費負担者番号1",   # 半角1
                "第１公費負担者番号", # 全角 第１
                "第一公費負担者番号", # 漢数字「第一」
                "負担者番号",
            ]
            sub_key_name_cmp = None
            for cand in public_aliases_cmp:
                if cand in cmp_df.columns:
                    sub_key_name_cmp = cand
                    break

            # どれも見つからなければダイアログで選択
            if sub_key_name_cmp is None:
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
                _mb.showerror("エラー", "公費負担者番号の列が比較/元CSVのどちらかで見つかりません。マッピング・CSVを確認してください。")
                return

        # 幅の決定（患者番号）
        try:
            import unicodedata, re
            cmp_digits_pat = cmp_df["患者番号"].astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
            width_pat = int(cmp_digits_pat.str.len().max()) if cmp_digits_pat.notna().any() else cfg.patient_number_width
            if not width_pat or width_pat <= 0:
                width_pat = cfg.patient_number_width
        except Exception:
            width_pat = cfg.patient_number_width
        # out_df側の患者番号桁数も考慮
        try:
            if "患者番号" in out_df.columns:
                import unicodedata, re
                out_digits_pat = out_df["患者番号"].astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
                out_max_pat = int(out_digits_pat.str.len().max()) if out_digits_pat.notna().any() else 0
                width_pat = max(width_pat, out_max_pat or 0) or width_pat
        except Exception:
            pass

        # 幅の決定（副キー: 保険者番号 / 公費負担者番号１）
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

        # 未ヒット抽出（src -> cmp）
        missing_df = None
        src_code_col = colmap.get("患者番号")
        if src_code_col and src_code_col in src.columns:
            if key_mode == "patient":
                src_codes_norm = self._normalize_codes(src[src_code_col].astype(str), width_pat, mode="zfill")
                cmp_codes_norm = self._normalize_codes(cmp_df["患者番号"].astype(str), width_pat, mode="zfill")
                cmp_set = set(cmp_codes_norm.loc[cmp_codes_norm != ""])  # 空は除外
                mask_missing = (src_codes_norm != "") & (~src_codes_norm.isin(cmp_set))
                if mask_missing.any():
                    missing_df = src.loc[mask_missing].copy()
                    missing_df.insert(0, "__正規化患者番号__", src_codes_norm.loc[mask_missing])
                # フォールバック
                try:
                    total = len(src)
                    miss_cnt = int(mask_missing.sum()) if 'mask_missing' in locals() else 0
                    if miss_cnt in (0, total):
                        src_ls = self._normalize_codes(src[src_code_col].astype(str), width_pat, mode="lstrip")
                        cmp_ls = self._normalize_codes(cmp_df["患者番号"].astype(str), width_pat, mode="lstrip")
                        cmp_set_ls = set(cmp_ls.loc[cmp_ls != ""])  # 空は除外
                        mask2 = (src_ls != "") & (~src_ls.isin(cmp_set_ls))
                        if mask2.any() or (miss_cnt == total and not mask2.any()):
                            missing_df = src.loc[mask2].copy()
                            missing_df.insert(0, "__正規化患者番号__", src_ls.loc[mask2])
                except Exception:
                    pass
            else:
                # 複合キー（厳密）：患者番号 + 保険者番号 / 公費負担者番号１
                src_keys_z = set(self._make_composite_keys(src, src_code_col, sub_key_name_src, width_pat, width_sub, mode="zfill"))
                cmp_keys_z = set(self._make_composite_keys(cmp_df, "患者番号", sub_key_name_cmp, width_pat, width_sub, mode="zfill"))
                miss_keys = [k for k in src_keys_z if k[0] != "" and k[1] != "" and k not in cmp_keys_z]
                if miss_keys:
                    # 元行を抽出（正規化キーを先頭に付与）
                    src_pat_z = self._normalize_codes(src[src_code_col].astype(str), width_pat, mode="zfill")
                    src_sub_z = self._normalize_codes(src[sub_key_name_src].astype(str), width_sub, mode="zfill")
                    key_series = list(zip(src_pat_z, src_sub_z))
                    mask_missing = pd.Series([ks in miss_keys for ks in key_series], index=src.index)
                    if mask_missing.any():
                        missing_df = src.loc[mask_missing].copy()
                        missing_df.insert(0, "__正規化副キー__", src_sub_z.loc[mask_missing])
                        missing_df.insert(0, "__正規化患者番号__", src_pat_z.loc[mask_missing])
                # フォールバック（先頭0無視）
                if missing_df is None or missing_df.empty:
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
        else:
            _mb.showwarning("注意", "患者番号のマッピングが不明です。未ヒット抽出をスキップします。")

        if missing_df is not None and not missing_df.empty:
            if _mb.askyesno("未ヒット出力", f"未ヒットの患者が {len(missing_df)} 件見つかりました。CSVとして保存しますか？"):
                prefix = "保険" if key_mode == "insurance" else ("公費" if key_mode == "public" else "患者")
                default_name = f"{prefix}_未ヒット患者_{_dt.now().strftime('%Y%m%d')}.csv"
                miss_path = filedialog.asksaveasfilename(
                    title="未ヒット患者リストを保存",
                    defaultextension=".csv",
                    initialfile=default_name,
                    filetypes=[("CSV files", "*.csv")]
                )
                if miss_path:
                    inspection.to_csv(missing_df, miss_path)
                    _mb.showinfo("未ヒット出力", f"未ヒット {len(missing_df)} 件を保存しました。\n{miss_path}")
                else:
                    _mb.showinfo("未ヒット出力", "保存をキャンセルしました。")

        # === 対象外リストの保存（任意） ===
        try:
            mig = None
            if key_mode == "public":
                mig = self.public_migration_yyyymmdd
            excluded_df = self._extract_excluded(
                src=src, colmap=colmap, key_mode=key_mode, migration_yyyymmdd=mig
            )
            # --- ここから：未ヒットを必ず対象外へ吸収するための前処理（要件対応） ---
            # matched_mask_src は後続で計算・利用するため、ここで初期化
            matched_mask_src = pd.Series([False] * len(src), index=src.index)

            # 対象外(一次)のインデックス集合
            excluded_idx_initial = set(excluded_df.index) if excluded_df is not None else set()

            # まずは Eligible（= 対象外でない元CSV行）を定義
            eligible_mask_initial = ~src.index.to_series().isin(excluded_idx_initial)
            # --- ここまで：未ヒットを必ず対象外へ吸収するための前処理 ---

            if excluded_df is not None and not excluded_df.empty:
                # --- 一致（突合一致）に含まれる src 側レコードは対象外リストから除外 ---
                try:
                    matched_mask_src = pd.Series([False] * len(src), index=src.index)
                    src_code_col = colmap.get("患者番号")
                    if src_code_col and src_code_col in src.columns:
                        if key_mode == "patient":
                            # 厳密：ゼロ埋めで比較（空は除外）
                            src_codes_norm_m = self._normalize_codes(src[src_code_col].astype(str), width_pat, mode="zfill")
                            cmp_codes_norm_m = self._normalize_codes(cmp_df["患者番号"].astype(str), width_pat, mode="zfill")
                            cmp_set_m = set(cmp_codes_norm_m.loc[cmp_codes_norm_m != ""])
                            matched_mask_src = (src_codes_norm_m != "") & (src_codes_norm_m.isin(cmp_set_m))
                        elif key_mode in ("insurance", "public"):
                            # 複合キー（厳密）：患者番号 + 副キー（保険者番号 / 公費負担者番号１）
                            if key_mode == "insurance":
                                sub_src = colmap.get("保険者番号")
                                sub_cmp = "保険者番号"
                            else:
                                sub_src = colmap.get("公費負担者番号１") or colmap.get("負担者番号")
                                sub_cmp = sub_key_name_cmp  # 比較側で実在した列名
                            if sub_src and sub_src in src.columns and sub_cmp in cmp_df.columns:
                                src_keys_z = self._make_composite_keys(src, src_code_col, sub_src, width_pat, width_sub, mode="zfill")
                                cmp_keys_z = self._make_composite_keys(cmp_df, "患者番号", sub_cmp, width_pat, width_sub, mode="zfill")
                                # 空キー（患者番号 or 副キーが空）を除外
                                src_keys_z = [(p, s) for (p, s) in src_keys_z if p and s]
                                cmp_keys_z = [(p, s) for (p, s) in cmp_keys_z if p and s]
                                cmp_key_set = set(cmp_keys_z)
                                # src 全行に対する一致マスク（空キーは False 扱い）
                                def _is_matched(k):
                                    return bool(k[0] and k[1] and k in cmp_key_set)
                                matched_mask_src = pd.Series([_is_matched(ks) for ks in self._make_composite_keys(src, src_code_col, sub_src, width_pat, width_sub, mode="zfill")], index=src.index)
                    # 除外対象から一致分を取り除く
                    if matched_mask_src.any():
                        excluded_df = excluded_df.loc[~excluded_df.index.isin(src.index[matched_mask_src])]
                except Exception:
                    # フィルタに失敗した場合はそのまま（安全側）
                    pass

                # --- ここから：Eligible 未ヒットを「未分類（未ヒット・要ルール）」理由で対象外へ編入 ---
                try:
                    # 対象外(マッチ除外後)の再計算
                    excluded_idx_now = set(excluded_df.index) if excluded_df is not None else set()
                    eligible_mask_now = ~src.index.to_series().isin(excluded_idx_now)

                    # Eligible のうち一致しなかったレコード（= 未ヒットEligible）
                    unmatched_eligible_mask = eligible_mask_now & (~matched_mask_src)
                    if unmatched_eligible_mask.any():
                        unmatched_df = src.loc[unmatched_eligible_mask].copy()
                        # 可能であれば正規化キーも付与（診断に役立つ）
                        try:
                            src_code_col = colmap.get("患者番号")
                            if src_code_col and src_code_col in src.columns:
                                # 患者番号の正規化（zfill幅は上で推定済みの width_pat を使用）
                                pat_norm = self._normalize_codes(src[src_code_col].astype(str), width_pat, mode="zfill")
                                unmatched_df.insert(0, "__正規化患者番号__", pat_norm.loc[unmatched_eligible_mask])
                            if key_mode in ("insurance", "public"):
                                # 副キーの正規化（幅は width_sub）
                                if key_mode == "insurance":
                                    sub_src = colmap.get("保険者番号")
                                else:
                                    sub_src = colmap.get("公費負担者番号１") or colmap.get("負担者番号")
                                if sub_src and sub_src in src.columns:
                                    sub_norm = self._normalize_codes(src[sub_src].astype(str), width_sub, mode="zfill")
                                    unmatched_df.insert(1, "__正規化副キー__", sub_norm.loc[unmatched_eligible_mask])
                        except Exception:
                            pass
                        # 未分類理由を付与
                        unmatched_df.insert(0, "__対象外理由__", "未分類（未ヒット・要ルール）")
                        # 既存対象外と結合（行インデックスが重なっても縦結合は安全）
                        if excluded_df is None or excluded_df.empty:
                            excluded_df = unmatched_df
                        else:
                            excluded_df = pd.concat([excluded_df, unmatched_df], axis=0)
                except Exception:
                    # ここで失敗しても検収自体は続行
                    pass
                # --- ここまで：未ヒットEligibleの対象外編入 ---

                if excluded_df is not None and not excluded_df.empty:
                    if _mb.askyesno("対象外データの出力", f"移行対象外 {len(excluded_df)} 件が見つかりました。CSVとして保存しますか？"):
                        prefix_ex = "保険" if key_mode == "insurance" else ("公費" if key_mode == "public" else "患者")
                        default_name_ex = f"{prefix_ex}_対象外_{_dt.now().strftime('%Y%m%d')}.csv"
                        ex_path = filedialog.asksaveasfilename(
                            title="対象外データを保存",
                            defaultextension=".csv",
                            initialfile=default_name_ex,
                            filetypes=[("CSV files", "*.csv")]
                        )
                        if ex_path:
                            inspection.to_csv(excluded_df, ex_path)
                            _mb.showinfo("対象外出力", f"対象外 {len(excluded_df)} 件を保存しました。\n{ex_path}")
                else:
                    _mb.showinfo("対象外データ", "対象外は 0 件でした。")

            # --- ここから：件数検算の参考警告 ---
            try:
                total_src = len(src)
                # eligible 未ヒットは、上の編入で 0 になっている前提だが、再計算して参考表示
                excl_count = len(excluded_df) if excluded_df is not None else 0
                # matched は True の数
                matched_count = int(matched_mask_src.sum()) if isinstance(matched_mask_src, pd.Series) else 0
                # 未ヒット（表示用）は、eligible_now ∧ ~matched を数える
                # ただし上で対象外に編入済みのため、ここでは 0 になっているはず
                try:
                    excluded_idx_now = set(excluded_df.index) if excluded_df is not None else set()
                    eligible_mask_now = ~src.index.to_series().isin(excluded_idx_now)
                    not_matched_now = eligible_mask_now & (~matched_mask_src)
                    missing_count_view = int(not_matched_now.sum())
                except Exception:
                    missing_count_view = 0
                if matched_count + missing_count_view + excl_count != total_src:
                    # 参考警告（致命ではない）
                    _mb.showwarning(
                        "参考情報",
                        f"件数検算にズレの可能性があります。\n"
                        f"元:{total_src} / 一致:{matched_count} / 未ヒット(表示):{missing_count_view} / 対象外:{excl_count}"
                    )
            except Exception:
                pass
            # --- ここまで：件数検算の参考警告 ---
        except Exception as _ex_err:
            _mb.showwarning("対象外出力エラー", f"対象外データの出力に失敗しました。\n{_ex_err}")

        # 検収CSVを一致のみで別名保存
        try:
            if "患者番号" in out_df.columns:
                if key_mode == "patient":
                    out_codes_norm = self._normalize_codes(out_df["患者番号"].astype(str), width_pat, mode="zfill")
                    cmp_codes_norm = self._normalize_codes(cmp_df["患者番号"].astype(str), width_pat, mode="zfill")
                    cmp_set = set(cmp_codes_norm.loc[cmp_codes_norm != ""])  # 空は除外
                    matched_mask = (out_codes_norm != "") & (out_codes_norm.isin(cmp_set))
                    filtered_out_df = out_df.loc[matched_mask].copy()
                    if filtered_out_df.empty:
                        out_lstrip = self._normalize_codes(out_df["患者番号"].astype(str), width_pat, mode="lstrip")
                        cmp_lstrip = self._normalize_codes(cmp_df["患者番号"].astype(str), width_pat, mode="lstrip")
                        cmp_set2 = set(cmp_lstrip.loc[cmp_lstrip != ""])  # 空は除外
                        matched2 = (out_lstrip != "") & (out_lstrip.isin(cmp_set2))
                        if matched2.any():
                            filtered_out_df = out_df.loc[matched2].copy()
                else:
                    # 複合キー一致（厳密）
                    if not out_sub_col or out_sub_col not in out_df.columns:
                        _mb.showerror("エラー", f"出力検収CSVに『{out_sub_col or '副キー列'}』列がありません。")
                        return
                    out_keys_z = self._make_composite_keys(out_df, "患者番号", out_sub_col, width_pat, width_sub, mode="zfill")
                    cmp_keys_z = self._make_composite_keys(cmp_df, "患者番号", sub_key_name_cmp, width_pat, width_sub, mode="zfill")
                    # 空キー（患者番号 or 副キーが空）を除外
                    out_keys_z = [(p, s) for (p, s) in out_keys_z if p and s]
                    cmp_keys_z = [(p, s) for (p, s) in cmp_keys_z if p and s]
                    matched_keys = set(out_keys_z) & set(cmp_keys_z)
                    if matched_keys:
                        # マスク作成
                        out_pat_z = self._normalize_codes(out_df["患者番号"].astype(str), width_pat, mode="zfill")
                        out_sub_z = self._normalize_codes(out_df[out_sub_col].astype(str), width_sub, mode="zfill")
                        key_series = list(zip(out_pat_z, out_sub_z))
                        matched_mask = pd.Series([ks in matched_keys for ks in key_series], index=out_df.index)
                        filtered_out_df = out_df.loc[matched_mask].copy()
                    else:
                        # フォールバック（先頭0無視）
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
                # --- 一致のみCSVの重複除去 ---
                # 保険/公費は複合キー（患者番号 + 副キー）で重複を落とす
                if 'filtered_out_df' in locals() and filtered_out_df is not None and not filtered_out_df.empty:
                    if key_mode in ("insurance", "public"):
                        dedup_cols = ["患者番号", out_sub_col]
                        if all(c in filtered_out_df.columns for c in dedup_cols):
                            filtered_out_df = filtered_out_df.drop_duplicates(subset=dedup_cols, keep="first")
                    elif key_mode == "patient":
                        # 念のため患者でも患者番号での重複を除去
                        if "患者番号" in filtered_out_df.columns:
                            filtered_out_df = filtered_out_df.drop_duplicates(subset=["患者番号"], keep="first")
                # --------------------------------
                if _mb.askyesno(
                    "検収CSVの絞り込み",
                    f"検収CSVを『一致のみ』({len(filtered_out_df) if 'filtered_out_df' in locals() else 0}行)に絞って別名保存しますか？\n（元の検収CSVはそのまま残ります）",
                ):
                    prefix2 = "保険" if key_mode == "insurance" else ("公費" if key_mode == "public" else "患者")
                    default_name2 = f"{prefix2}_検収_一致のみ_{_dt.now().strftime('%Y%m%d')}.csv"
                    filtered_path = filedialog.asksaveasfilename(
                        title="検収CSV（一致のみ）を保存",
                        defaultextension=".csv",
                        initialfile=default_name2,
                        filetypes=[("CSV files", "*.csv")]
                    )
                    if filtered_path:
                        inspection.to_csv(filtered_out_df if 'filtered_out_df' in locals() else out_df.head(0), filtered_path)
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

            default_name = f"患者_検収_{_dt.now().strftime('%Y%m%d')}.csv"
            out_path = filedialog.asksaveasfilename(
                title="検収CSVを保存",
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV files", "*.csv")]
            )
            if not out_path:
                return
            inspection.to_csv(out_df, out_path)

            # 突合（任意）
            self._ask_and_save_missing_and_matched(src=src, colmap=colmap, out_df=out_df, cfg=cfg, key_mode="patient")

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

            default_name = f"保険_検収_{_dt.now().strftime('%Y%m%d')}.csv"
            out_path = filedialog.asksaveasfilename(
                title="保険情報 検収CSVを保存",
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV files", "*.csv")]
            )
            if not out_path:
                return
            inspection.to_csv(out_df, out_path)

            # 突合（任意）
            self._ask_and_save_missing_and_matched(src=src, colmap=colmap, out_df=out_df, cfg=cfg, key_mode="insurance")

            messagebox.showinfo("完了", f"保険情報の検収CSVを保存しました:\n{out_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"保険情報の検収処理に失敗しました。\n{e}")

    def run_public(self):
        in_path = filedialog.askopenfilename(title="公費情報の入力CSVを選択してください", filetypes=[("CSV files", "*.csv")])
        if not in_path:
            return
        try:
            src = CsvLoader.read_csv_flex(in_path)
            required_cols = list(inspection.COLUMNS_PUBLIC) + ["公費終了日１"]
            colmap = self.app._ask_inspection_colmap(src, required_cols=required_cols)
            if colmap is None:
                return
            from tkinter import simpledialog
            _today_str = _dt.now().strftime("%Y%m%d")
            mig_input = simpledialog.askstring(
                "データ移行日の入力",
                "データ移行日を入力してください（例: 2024/07/10, 2024-07-10, R6.7.10 など）\n"
                "※ 公費終了日がこの日より前のものは『期限切れ』として対象外になります。",
                initialvalue=_today_str,
                parent=self.app,
            )
            if mig_input:
                mig_yyyymmdd = inspection._parse_date_any_to_yyyymmdd(mig_input)
                if mig_yyyymmdd:
                    self.public_migration_yyyymmdd = mig_yyyymmdd
                else:
                    messagebox.showwarning("移行日", "移行日の解釈に失敗しました。今日の日付を使用します。")
                    self.public_migration_yyyymmdd = _today_str
            else:
                # キャンセルや空欄は今日を使用
                self.public_migration_yyyymmdd = _today_str
            cfg = inspection.InspectionConfig(patient_number_width=10)
            out_df = inspection.build_inspection_df(src, colmap, cfg, target_columns=inspection.COLUMNS_PUBLIC)

            default_name = f"公費_検収_{_dt.now().strftime('%Y%m%d')}.csv"
            out_path = filedialog.asksaveasfilename(
                title="公費情報 検収CSVを保存",
                defaultextension=".csv",
                initialfile=default_name,
                filetypes=[("CSV files", "*.csv")]
            )
            if not out_path:
                return
            inspection.to_csv(out_df, out_path)

            # 突合（任意）
            self._ask_and_save_missing_and_matched(src=src, colmap=colmap, out_df=out_df, cfg=cfg, key_mode="public")

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
            inspection.to_csv(missing_df, out_path)
            messagebox.showinfo("完了", f"未ヒット {len(missing_df)} 件を出力しました。\n{out_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"未ヒットリストの保存に失敗しました。\n{e}")