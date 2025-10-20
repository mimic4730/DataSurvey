# core/rules/public.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict
from core import inspection

@dataclass
class PublicRuleConfig:
    migration_yyyymmdd: str | None = None  # 期限切れ判定基準。Noneなら期限切れ判定は無効

def _digits_only(s: pd.Series) -> pd.Series:
    import re, unicodedata
    return s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))

def _date_norm(s: pd.Series) -> pd.Series:
    # 99999999 / 9999-99-99 / “9”埋め等は空欄扱いに寄せる（移行ツール運用に合わせる）
    def _norm(v):
        v2 = inspection._parse_date_any_to_yyyymmdd(v)
        if not v2:
            return ""
        v2 = str(v2)
        try:
            if set(v2) == {"9"}:
                return ""
        except Exception:
            pass
        return v2
    return s.map(_norm)

def _cap_future_to_migration(s: pd.Series, mig: str | None) -> pd.Series:
    if not mig:
        return s
    def _cap(v):
        if not v:
            return mig  # 空欄は移行日に補完（内容検収と同等の挙動）
        try:
            return mig if v > mig else v
        except Exception:
            return v
    return s.map(_cap)

def evaluate_public_exclusions(
    df: pd.DataFrame,
    colmap: Dict[str, str],
    cfg: PublicRuleConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    公費の移行対象外を抽出し、『残す行』も返す共通関数。
    戻り値: (remains_df, excluded_df)
      - remains_df … 比較/突合に回すレコード
      - excluded_df … 対象外レコード（先頭に '__対象外理由__' 列付き）
    期待する colmap キー（入力CSV上の列名を紐付け）:
      '患者番号', '公費負担者番号', '公費受給者番号', '公費開始日', '公費終了日'
      ※ スロット1/2を縦持ちにした後の「処理用フラット列」を渡してください
    """
    # 例外を上げずに常に戻す方針
    try:
        df = df.copy()
        if len(df) == 0:
            return df, df.head(0)

        # 列の取り出し（足りない場合は空列）
        def col(name: str) -> pd.Series:
            c = colmap.get(name)
            return df[c] if c and c in df.columns else pd.Series([""] * len(df), index=df.index, dtype="object")

        # 正規化
        pat = _digits_only(col("患者番号")).astype(str)
        payer = _digits_only(col("公費負担者番号")).astype(str)
        recip = _digits_only(col("公費受給者番号")).astype(str)

        start = _date_norm(col("公費開始日")).astype(str)
        end = _date_norm(col("公費終了日")).astype(str)

        # 未来開始日は移行日に丸める（cfg 準拠）
        mig = (cfg.migration_yyyymmdd or "").strip() or None
        start = _cap_future_to_migration(start, mig)

        # 理由列と各種マスク
        reasons = pd.Series([""] * len(df), index=df.index, dtype="object")

        mask_pat_empty   = (pat == "")
        mask_payer_empty = (payer == "")
        mask_recip_empty = (recip == "")

        # 先頭2桁 27/28 は対象外（空欄は除く）
        payer_prefix = payer.str[:2]
        mask_prefix_27_28 = payer_prefix.isin(["27", "28"]) & ~mask_payer_empty

        # 期限切れ（終了日 < 移行日）
        if mig:
            mask_expired = (end != "") & (end < mig)
        else:
            mask_expired = pd.Series([False] * len(df), index=df.index)

        # (患者番号+公費負担者番号) 重複（左ゼロ無視で検出）
        # zip(list(...)) はメモリ/型揺れで例外化するケースがあるため DataFrame.duplicated を利用
        combo_df = pd.DataFrame({
            "_p": pat.str.lstrip("0"),
            "_g": payer.str.lstrip("0"),
        }, index=df.index)
        mask_combo_dup = combo_df.duplicated(keep=False) & ~(mask_pat_empty | mask_payer_empty)

        # 総合対象外マスク
        ex_mask = mask_pat_empty | mask_payer_empty | mask_recip_empty | mask_prefix_27_28 | mask_expired | mask_combo_dup

        # 理由付け
        def add_reason(mask: pd.Series, text: str):
            if not isinstance(mask, pd.Series) or mask.index is not reasons.index:
                mask = pd.Series(mask, index=df.index)
            if mask.any():
                idx = mask[mask].index
                reasons.loc[idx] = reasons.loc[idx].astype(str).str.cat(
                    pd.Series([text] * len(idx), index=idx), sep=" / "
                ).str.strip(" /")

        add_reason(mask_pat_empty, "患者番号空欄")
        add_reason(mask_payer_empty, "公費負担者番号空欄")
        add_reason(mask_recip_empty, "公費受給者番号空欄")
        add_reason(mask_prefix_27_28, "公費負担者番号先頭2桁が27/28(対象外)")
        if mig:
            add_reason(mask_expired, f"公費終了日が移行日({mig})より前(期限切れ)")
        add_reason(mask_combo_dup, "患者番号+公費負担者番号重複")

        excluded = df.loc[ex_mask].copy()
        if not excluded.empty:
            excluded.insert(0, "__対象外理由__", reasons.loc[excluded.index])

        remains = df.loc[~ex_mask].copy()
        return remains, excluded
    except Exception as e:
        # 何が起きても黙って空返しは危険なので、最小限のデバッグ列を付けた空DFを返す
        try:
            empty = df.head(0).copy()
        except Exception:
            empty = pd.DataFrame()
        return df.copy(), empty