"""
患者情報の『移行対象外』判定ルールの単一ソース。
- 検収CSV生成フローおよび内容検収（ui.patient_content_check.run_integrated）双方から参照されます。
"""
#core/rules/patient.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict
from core import inspection

@dataclass
class PatientRuleConfig:
    """患者情報の対象外判定設定。
    - birth_min_yyyymmdd: 生年月日の下限制約（これより前は対象外）
    - patient_number_width: 0埋めなどの幅情報（重複検知では左ゼロ無視を使用）
    """
    birth_min_yyyymmdd: str = "19000101"
    patient_number_width: int = 10

# ---- 正規化ユーティリティ ----

def _digits_only(s: pd.Series) -> pd.Series:
    import re, unicodedata
    return s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))

def _norm_text(s: pd.Series) -> pd.Series:
    import unicodedata
    return s.astype(str).map(lambda x: unicodedata.normalize("NFKC", x).strip())

def _birth_norm(s: pd.Series) -> pd.Series:
    return s.map(inspection._parse_date_any_to_yyyymmdd).fillna("").astype(str)

# カナが記号/数値のみか（空欄は別マスク）
import re as _re
_ALLOWED_CHARS = _re.compile(r"[ァ-ヶｦ-ﾟーｰ\u30A0-\u30FF\u3040-\u309F]")

def _is_kana_symbol_or_number_only(x: str) -> bool:
    if not x:
        return False
    # 許容文字（ひらがな/カタカナ/英数字）が1つも含まれなければ記号/空白のみとみなす
    return _ALLOWED_CHARS.search(x) is None

def evaluate_patient_exclusions(
    df: pd.DataFrame,
    colmap: Dict[str, str],
    cfg: PatientRuleConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    患者の移行対象外を抽出し、『残す行』も返す共通関数。
    戻り値: (remains_df, excluded_df)
      - remains_df … 突合/内容検収に回すレコード
      - excluded_df … 対象外レコード（先頭に '__対象外理由__' 列付き）

    期待する colmap キー:
      '患者番号', '患者氏名カナ', '患者氏名', '生年月日'
    足りない列は空欄扱いで処理します。
    """
    cfg = cfg or PatientRuleConfig()

    try:
        df = df.copy()
        if len(df) == 0:
            return df, df.head(0)

        def col(name: str) -> pd.Series:
            c = colmap.get(name)
            return df[c] if c and c in df.columns else pd.Series([""] * len(df), index=df.index, dtype="object")

        pat_raw  = col("患者番号")
        kana_raw = col("患者氏名カナ")
        name_raw = col("患者氏名")
        birth_raw= col("生年月日")

        # 正規化
        pat   = _digits_only(pat_raw)
        kana  = _norm_text(kana_raw)
        name  = _norm_text(name_raw)
        birth = _birth_norm(birth_raw)

        # 理由列
        reasons = pd.Series([""] * len(df), index=df.index, dtype="object")

        # マスク群
        mask_pat_empty        = (pat == "")
        mask_name_both_empty  = (kana.str.strip() == "") & (name.str.strip() == "")
        mask_kana_bad         = (kana.str.strip() != "") & kana.map(_is_kana_symbol_or_number_only)
        birth_min             = (cfg.birth_min_yyyymmdd or "").strip() or "19000101"
        mask_birth_too_old    = (birth != "") & (birth < birth_min)
        # 患者番号重複（左ゼロ無視）
        pat_lstrip            = pat.str.lstrip("0")
        mask_pat_dup          = pat_lstrip.duplicated(keep=False) & ~mask_pat_empty

        ex_mask = mask_pat_empty | mask_name_both_empty | mask_kana_bad | mask_birth_too_old | mask_pat_dup

        def add_reason(mask: pd.Series, text: str):
            if not isinstance(mask, pd.Series) or mask.index is not reasons.index:
                mask = pd.Series(mask, index=df.index)
            if mask.any():
                idx = mask[mask].index
                reasons.loc[idx] = reasons.loc[idx].astype(str).str.cat(
                    pd.Series([text] * len(idx), index=idx), sep=" / "
                ).str.strip(" /")

        add_reason(mask_pat_empty,       "患者番号空欄")
        add_reason(mask_name_both_empty, "氏名/カナの両方が空欄")
        add_reason(mask_kana_bad,        "カナ氏名が記号/数値のみ")
        add_reason(mask_birth_too_old,   f"生年月日が{birth_min}より前")
        add_reason(mask_pat_dup,         "患者番号重複")

        excluded = df.loc[ex_mask].copy()
        if not excluded.empty:
            excluded.insert(0, "__対象外理由__", reasons.loc[excluded.index])

        remains = df.loc[~ex_mask].copy()
        return remains, excluded

    except Exception:
        try:
            empty = df.head(0).copy()
        except Exception:
            empty = pd.DataFrame()
        return df.copy(), empty