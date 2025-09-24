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

    def _make_composite_keys(self, df: pd.DataFrame, pat_col: str, sub_col: str,
                             width_pat: int, width_sub: int, mode: str = "zfill"):
        pat = self._normalize_codes(df[pat_col].astype(str), width_pat, mode=mode)
        sub = self._normalize_codes(df[sub_col].astype(str), width_sub, mode=mode)
        return list(zip(pat, sub))

    def _extract_excluded(self, src: pd.DataFrame, colmap: dict, key_mode: str) -> pd.DataFrame:
        """
        仕様に基づき「移行対象外」の行を抽出して返す。
        key_mode: "patient" | "insurance" | "public"
        戻り値: 対象外行のDataFrame（理由列 '__対象外理由__' を先頭に付与）
        """
        def _digits_or_empty(series: pd.Series) -> pd.Series:
            import re
            return series.astype(str).map(lambda x: re.sub(r"\D", "", x)).map(lambda x: x if x else "")

        try:
            df = src.copy()
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
                    code_digits = df[code_col].astype(str).str.replace(r"\D", "", regex=True).str.lstrip("0")
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
                    code_digits = df[code_col].astype(str).str.replace(r"\D", "", regex=True).str.lstrip("0")
                    mask_code_dup = code_digits.duplicated(keep=False) & (code_digits != "")
                # ---------------------------
                # 保険者番号の正規化: 7桁→頭0で8桁, 5桁→頭0で6桁。他は桁数維持（数字のみ残す）
                payer_norm = None
                if payer_col and payer_col in df.columns:
                    payer_digits = df[payer_col].astype(str).str.replace(r"\D", "", regex=True)
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
                    cardno_digits = df[cardno_col].astype(str).str.replace(r"\D", "", regex=True)
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
                # --- 新ルール: 患者番号の重複（先頭0を無視して比較） ---
                mask_code_dup = pd.Series([False] * len(df), index=df.index)
                if code_col and code_col in df.columns:
                    code_digits = df[code_col].astype(str).str.replace(r"\D", "", regex=True).str.lstrip("0")
                    mask_code_dup = code_digits.duplicated(keep=False) & (code_digits != "")
                # ---------------------------
                mask_payer1_empty = pd.Series([False] * len(df), index=df.index)
                mask_recip1_empty = pd.Series([False] * len(df), index=df.index)
                if payer1_col and payer1_col in df.columns:
                    mask_payer1_empty = _digits_or_empty(df[payer1_col]).map(lambda s: s == "")
                if recip1_col and recip1_col in df.columns:
                    mask_recip1_empty = _digits_or_empty(df[recip1_col]).map(lambda s: s == "")
                mask = mask_code_empty | mask_code_dup | mask_payer1_empty | mask_recip1_empty
                reasons.loc[mask_payer1_empty] = reasons.loc[mask_payer1_empty].astype(str).str.cat(pd.Series(["公費負担者番号空欄"] * int(mask_payer1_empty.sum()), index=reasons.loc[mask_payer1_empty].index), sep=" / ").str.strip(" /")
                reasons.loc[mask_recip1_empty] = reasons.loc[mask_recip1_empty].astype(str).str.cat(pd.Series(["公費受給者番号空欄"] * int(mask_recip1_empty.sum()), index=reasons.loc[mask_recip1_empty].index), sep=" / ").str.strip(" /")
                reasons.loc[mask_code_empty] = reasons.loc[mask_code_empty].astype(str).str.cat(
                    pd.Series(["患者番号空欄"] * int(mask_code_empty.sum()), index=reasons.loc[mask_code_empty].index),
                    sep=" / "
                ).str.strip(" /")
                reasons.loc[mask_code_dup] = reasons.loc[mask_code_dup].astype(str).str.cat(
                    pd.Series(["患者番号重複"] * int(mask_code_dup.sum()), index=reasons.loc[mask_code_dup].index),
                    sep=" / "
                ).str.strip(" /")

            else:
                # 不明モード
                return src.head(0)

            excluded = df.loc[mask].copy()
            if not excluded.empty:
                excluded.insert(0, "__対象外理由__", reasons.loc[excluded.index])
                # 保険者番号の正規化を2列目に追加
                if payer_norm is not None:
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
        sub_key_name_cmp = None
        sub_key_name_src = None
        if key_mode == "insurance":
            sub_key_name_cmp = "保険者番号"
            sub_key_name_src = colmap.get("保険者番号")
            if sub_key_name_cmp not in cmp_df.columns or not sub_key_name_src or sub_key_name_src not in src.columns:
                _mb.showerror("エラー", "保険者番号が比較/元CSVのどちらかで見つかりません。マッピングとCSVを確認してください。")
                return
        elif key_mode == "public":
            sub_key_name_cmp = "公費負担者番号１"
            sub_key_name_src = colmap.get("公費負担者番号１")
            if sub_key_name_cmp not in cmp_df.columns or not sub_key_name_src or sub_key_name_src not in src.columns:
                _mb.showerror("エラー", "公費負担者番号１が比較/元CSVのどちらかで見つかりません。マッピングとCSVを確認してください。")
                return

        # 幅の決定（患者番号）
        try:
            cmp_digits_pat = cmp_df["患者番号"].astype(str).str.replace(r"\D", "", regex=True)
            width_pat = int(cmp_digits_pat.str.len().max()) if cmp_digits_pat.notna().any() else cfg.patient_number_width
            if not width_pat or width_pat <= 0:
                width_pat = cfg.patient_number_width
        except Exception:
            width_pat = cfg.patient_number_width
        # out_df側の患者番号桁数も考慮
        try:
            if "患者番号" in out_df.columns:
                out_digits_pat = out_df["患者番号"].astype(str).str.replace(r"\D", "", regex=True)
                out_max_pat = int(out_digits_pat.str.len().max()) if out_digits_pat.notna().any() else 0
                width_pat = max(width_pat, out_max_pat or 0) or width_pat
        except Exception:
            pass

        # 幅の決定（副キー: 保険者番号 / 公費負担者番号１）
        width_sub = 0
        if key_mode in ("insurance", "public"):
            try:
                cmp_digits_sub = cmp_df[sub_key_name_cmp].astype(str).str.replace(r"\D", "", regex=True)
                width_sub = int(cmp_digits_sub.str.len().max()) if cmp_digits_sub.notna().any() else 0
            except Exception:
                width_sub = 0
            try:
                if sub_key_name_cmp in out_df.columns:
                    out_digits_sub = out_df[sub_key_name_cmp].astype(str).str.replace(r"\D", "", regex=True)
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
            excluded_df = self._extract_excluded(src=src, colmap=colmap, key_mode=key_mode)
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
                                sub_src = colmap.get("公費負担者番号１")
                                sub_cmp = "公費負担者番号１"
                            if sub_src and sub_src in src.columns and sub_cmp in cmp_df.columns:
                                src_keys_z = self._make_composite_keys(src, src_code_col, sub_src, width_pat, width_sub, mode="zfill")
                                cmp_keys_z = self._make_composite_keys(cmp_df, "患者番号", sub_cmp, width_pat, width_sub, mode="zfill")
                                cmp_key_set = set(cmp_keys_z)
                                matched_mask_src = pd.Series([ks in cmp_key_set for ks in src_keys_z], index=src.index)
                    # 除外対象から一致分を取り除く
                    if matched_mask_src.any():
                        excluded_df = excluded_df.loc[~excluded_df.index.isin(src.index[matched_mask_src])]
                except Exception:
                    # フィルタに失敗した場合はそのまま（安全側）
                    pass

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
                    if sub_key_name_cmp not in out_df.columns:
                        _mb.showerror("エラー", f"出力検収CSVに『{sub_key_name_cmp}』列がありません。")
                        return
                    out_keys_z = set(self._make_composite_keys(out_df, "患者番号", sub_key_name_cmp, width_pat, width_sub, mode="zfill"))
                    cmp_keys_z = set(self._make_composite_keys(cmp_df, "患者番号", sub_key_name_cmp, width_pat, width_sub, mode="zfill"))
                    matched_keys = out_keys_z & cmp_keys_z
                    if matched_keys:
                        # マスク作成
                        out_pat_z = self._normalize_codes(out_df["患者番号"].astype(str), width_pat, mode="zfill")
                        out_sub_z = self._normalize_codes(out_df[sub_key_name_cmp].astype(str), width_sub, mode="zfill")
                        key_series = list(zip(out_pat_z, out_sub_z))
                        matched_mask = pd.Series([ks in matched_keys for ks in key_series], index=out_df.index)
                        filtered_out_df = out_df.loc[matched_mask].copy()
                    else:
                        # フォールバック（先頭0無視）
                        out_keys_l = set(self._make_composite_keys(out_df, "患者番号", sub_key_name_cmp, width_pat, width_sub, mode="lstrip"))
                        cmp_keys_l = set(self._make_composite_keys(cmp_df, "患者番号", sub_key_name_cmp, width_pat, width_sub, mode="lstrip"))
                        matched_keys2 = out_keys_l & cmp_keys_l
                        if matched_keys2:
                            out_pat_l = self._normalize_codes(out_df["患者番号"].astype(str), width_pat, mode="lstrip")
                            out_sub_l = self._normalize_codes(out_df[sub_key_name_cmp].astype(str), width_sub, mode="lstrip")
                            key_series2 = list(zip(out_pat_l, out_sub_l))
                            matched_mask2 = pd.Series([ks in matched_keys2 for ks in key_series2], index=out_df.index)
                            filtered_out_df = out_df.loc[matched_mask2].copy()
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
            colmap = self.app._ask_inspection_colmap(src, required_cols=list(inspection.COLUMNS_PUBLIC))
            if colmap is None:
                return
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