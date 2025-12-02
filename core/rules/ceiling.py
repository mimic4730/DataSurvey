# core/rules/ceiling.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from core import inspection


@dataclass
class CeilingRuleConfig:
    """限度額認定証の対象外判定に使う設定."""
    migration_yyyymmdd: str  # 例: '20251114'


def _normalize_date_yyyymmdd(val) -> str:
    """
    いろいろな日付表記を YYYYMMDD の文字列に正規化。
    inspection._parse_date_any_to_yyyymmdd を薄くラップ。
    99999999 や空欄は \"\" に統一。
    """
    try:
        s = inspection._parse_date_any_to_yyyymmdd(val)
        if not s:
            return ""
        s = str(s)
        # 99999999 など「全部9」は無期限扱いとし、ここでは空にしておく
        if set(s) == {"9"}:
            return ""
        return s
    except Exception:
        return ""


def evaluate_ceiling_exclusions(
    src: pd.DataFrame,
    colmap: Dict[str, str],
    cfg: CeilingRuleConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    限度額認定証の「対象外」行を抽出する。

    ルール:
      1. マッピングした「限度額認定証適用区分」が空欄の行は対象外
      2. マッピングした「限度額認定証終了日」が移行日より前の日付の行は対象外
         - 終了日が空欄 or 無期限(99999999 等)は『終了日ルール』では対象外にしない
    """
    df = src.copy()

    col_apply = colmap.get("限度額認定証適用区分")
    col_end   = colmap.get("限度額認定証終了日")

    reasons = pd.Series([""] * len(df), index=df.index, dtype="object")

    # --- ルール1: 適用区分が空欄 ---
    if col_apply and col_apply in df.columns:
        s_apply = df[col_apply].astype(str).str.strip()
        mask_empty_apply = (s_apply == "") | s_apply.isna()
        reasons = reasons.mask(mask_empty_apply & (reasons == ""), "適用区分が空欄")
        reasons = reasons.mask(mask_empty_apply & (reasons != ""), reasons + " / 適用区分が空欄")
    else:
        # マッピングされていない場合は、何もしない（全部対象として扱う）
        mask_empty_apply = pd.Series([False] * len(df), index=df.index)

    # --- ルール2: 終了日が移行日より前 ---
    if col_end and col_end in df.columns and cfg.migration_yyyymmdd:
        s_end_norm = df[col_end].map(_normalize_date_yyyymmdd)
        # 空欄は比較しない（→このルールでは対象外にしない）
        mask_has_end = s_end_norm != ""
        mask_before_mig = (s_end_norm < cfg.migration_yyyymmdd) & mask_has_end

        reasons = reasons.mask(mask_before_mig & (reasons == ""), "終了日が移行日より前")
        reasons = reasons.mask(mask_before_mig & (reasons != ""), reasons + " / 終了日が移行日より前")
    else:
        mask_before_mig = pd.Series([False] * len(df), index=df.index)

    # --- 対象外フラグ ---
    mask_excluded = mask_empty_apply | mask_before_mig

    excluded = df.loc[mask_excluded].copy()
    if not excluded.empty:
        excluded.insert(0, "__対象外理由__", reasons.loc[mask_excluded])
    else:
        excluded = df.head(0).copy()
        excluded.insert(0, "__対象外理由__", [])

    remains = df.loc[~mask_excluded].copy()
    return remains, excluded