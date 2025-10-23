# core/rules/insurance.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

# 既存のユーティリティ（和暦/ゆるい日付→YYYYMMDD 変換）を利用
from core import inspection


@dataclass
class InsuranceRuleConfig:
    """
    保険の移行対象外ルール設定
    """
    migration_yyyymmdd: str | None = None         # 期限切れ・年齢判定の基準日（YYYYMMDD）。Noneで無効
    require_cardno_if_payer_present: bool = True  # 保険者番号があるとき保険証番号必須
    enforce_law39_cardno_len8: bool = True        # 法別39(8桁)のとき保険証番号は8桁のみ許可
    allowed_payer_prefix_8: tuple[str, ...] = (
        "01","02","03","04","06","07","39","31","32","33","34","63","72","73","74","75"
    )                                             # 8桁時の先頭2桁許容
    check_pref_code_when_8digits: bool = True     # 8桁時の3-4桁(都道府県コード)が01..47か


# ---- 内部ヘルパ ----
def _digits_only(s: pd.Series) -> pd.Series:
    import re, unicodedata
    return s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))

def _digits_or_empty(s: pd.Series) -> pd.Series:
    d = _digits_only(s)
    # すべて'0' の値（"0","000"...）は空扱い
    return d.map(lambda x: "" if (x == "" or set(x) == {"0"}) else x)

def _alnum_or_empty(s: pd.Series) -> pd.Series:
    """
    英数字のみを残し、空は空扱い。
    保険証番号の『存在』判定用（英字含む番号を許可、全0も保持）。
    """
    import re, unicodedata
    def _f(x: str) -> str:
        x = unicodedata.normalize("NFKC", str(x))
        x = re.sub(r"[^A-Za-z0-9]", "", x)
        if x == "":
            return ""
        return x
    return s.astype(str).map(_f)

def _norm_date_yyyymmdd_or_empty(s: pd.Series) -> pd.Series:
    def _norm(v):
        d = inspection._parse_date_any_to_yyyymmdd(v)
        if not d:
            return ""
        d = str(d)
        try:
            if set(d) == {"9"}:  # 99999999 系は空扱い（移行ツール運用に合わせる）
                return ""
        except Exception:
            pass
        return d
    return s.map(_norm)

def _pad_payer_len(s: pd.Series) -> pd.Series:
    """
    保険者番号の長さ補正：7→8桁、5→6桁。その他は数字のみ維持。
    """
    d = _digits_or_empty(s)
    def _pad(v: str) -> str:
        if v == "":
            return ""
        n = len(v)
        if n == 7:
            return v.zfill(8)
        if n == 5:
            return v.zfill(6)
        return v
    return d.map(_pad)

# ---- ルール本体 ----
def evaluate_insurance_exclusions(
    df: pd.DataFrame,
    colmap: Dict[str, str],
    cfg: InsuranceRuleConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    保険の『移行対象外』を抽出し、残り（比較対象に回す行）も返す共通関数。
    戻り値: (remains_df, excluded_df)
      - remains_df … 比較/突合に回すレコード
      - excluded_df … 対象外レコード（先頭に '__対象外理由__'、可能なら '__正規化保険者番号__' を付与）

    期待する colmap キー（入力CSV上の列名を紐付け）:
      '患者番号', '保険者番号', '保険証番号', '保険終了日', '生年月日'
    不足列があっても例外にせず、空列で評価します。
    """
    try:
        base = df.copy()
        if base.empty:
            return base, base.head(0)

        # 列取り出し（足りなければ空列）
        def col(name: str) -> pd.Series:
            c = colmap.get(name)
            if c and c in base.columns:
                return base[c]
            return pd.Series([""] * len(base), index=base.index, dtype="object")

        # 正規化
        pat = _digits_or_empty(col("患者番号")).astype(str)
        payer = _pad_payer_len(col("保険者番号")).astype(str)      # 長さ補正込み
        cardno = _alnum_or_empty(col("保険証番号")).astype(str)
        end = _norm_date_yyyymmdd_or_empty(col("保険終了日")).astype(str)
        birth = _norm_date_yyyymmdd_or_empty(col("生年月日")).astype(str)

        # 理由カラム
        reasons = pd.Series([""] * len(base), index=base.index, dtype="object")
        def add_reason(mask: pd.Series, text: str):
            if mask is None or not isinstance(mask, pd.Series):
                return
            if mask.any():
                idx = mask[mask].index
                # 追記型
                reasons.loc[idx] = reasons.loc[idx].astype(str).str.cat(
                    pd.Series([text]*len(idx), index=idx), sep=" / "
                ).str.strip(" /")

        # 1) 患者番号 空/重複（先頭ゼロ無視）
        mask_pat_empty = (pat == "")
        pat_l = pat.str.lstrip("0")
        mask_pat_dup = pat_l.duplicated(keep=False) & (pat_l != "")

        # 2) 保険者番号 空/桁不正（6 または 8 桁以外）
        mask_payer_empty = (payer == "")
        mask_payer_len_ng = (payer != "") & (~payer.str.len().isin([6, 8]))

        # 3) 保険証番号の必須・桁条件
        if cfg.require_cardno_if_payer_present:
            mask_cardno_empty = (~mask_payer_empty) & (cardno == "")
        else:
            mask_cardno_empty = pd.Series([False]*len(base), index=base.index)

        # 法別39（8桁・頭"39"）→ 保険証番号は 8 桁のみ
        if cfg.enforce_law39_cardno_len8:
            mask_law39_cardno_not8 = (
                (payer.str.len() == 8) &
                payer.str.startswith("39") &
                (cardno.str.len() != 8)
            )
        else:
            mask_law39_cardno_not8 = pd.Series([False]*len(base), index=base.index)

        # 4) 先頭2桁（8桁時）
        if len(cfg.allowed_payer_prefix_8) > 0:
            mask_payer_prefix_ng = (payer.str.len() == 8) & (~payer.str[:2].isin(cfg.allowed_payer_prefix_8))
        else:
            mask_payer_prefix_ng = pd.Series([False]*len(base), index=base.index)

        # 5) 都道府県コード（3-4桁）01..47
        if cfg.check_pref_code_when_8digits:
            allowed_pref = {f"{i:02d}" for i in range(1, 48)}
            mask_prefcode_invalid = (payer.str.len() == 8) & (~payer.str[2:4].isin(allowed_pref))
        else:
            mask_prefcode_invalid = pd.Series([False]*len(base), index=base.index)

        # 6) 期限切れ（終了日 < 移行日）
        mig = (cfg.migration_yyyymmdd or "").strip() or None
        if mig:
            mask_expired = (end != "") & (end < mig)
        else:
            mask_expired = pd.Series([False]*len(base), index=base.index)

        # 7) 75歳以上 & 法別≠39（移行日基準）
        def _is_75_or_over(b: str, mig_yyyymmdd: str | None) -> bool:
            if not mig_yyyymmdd or not b or len(b) != 8:
                return False
            try:
                from datetime import date
                by, bm, bd = int(b[:4]), int(b[4:6]), int(b[6:8])
                my, mm, md = int(mig_yyyymmdd[:4]), int(mig_yyyymmdd[4:6]), int(mig_yyyymmdd[6:8])
                bd_ = date(by, bm, bd); md_ = date(my, mm, md)
                age = md_.year - bd_.year - ((md_.month, md_.day) < (bd_.month, bd_.day))
                return age >= 75
            except Exception:
                return False

        if mig:
            is75 = birth.map(lambda b: _is_75_or_over(b, mig))
        else:
            is75 = pd.Series([False]*len(base), index=base.index)

        law2 = payer.str[:2].where(payer != "", "")
        mask_age75_law_not39 = is75 & (law2 != "") & (law2 != "39")

        # 総合対象外
        ex_mask = (
            mask_pat_empty
            | mask_pat_dup
            | mask_payer_empty
            | mask_payer_len_ng
            | mask_cardno_empty
            | mask_law39_cardno_not8
            | mask_payer_prefix_ng
            | mask_prefcode_invalid
            | mask_expired
            | mask_age75_law_not39
        )

        # 理由付与
        add_reason(mask_pat_empty, "患者番号空欄")
        add_reason(mask_pat_dup, "患者番号重複")
        add_reason(mask_payer_empty, "保険者番号空欄")
        add_reason(mask_payer_len_ng, "保険者番号桁数不正(6または8桁以外)")
        if cfg.require_cardno_if_payer_present:
            add_reason(mask_cardno_empty, "保険証番号空欄")
        if cfg.enforce_law39_cardno_len8:
            add_reason(mask_law39_cardno_not8, "保険者番号(39xxxxxx)は保険証番号8桁のみ対象")
        add_reason(mask_payer_prefix_ng, "保険者番号先頭2桁が対象外")
        if cfg.check_pref_code_when_8digits:
            add_reason(mask_prefcode_invalid, "保険者番号都道府県コード不正(3-4桁)")
        if mig:
            add_reason(mask_expired, f"保険終了日が移行日({mig})より前(期限切れ)")
        add_reason(mask_age75_law_not39, "75歳以上・法別≠39")

        # 出力
        excluded = base.loc[ex_mask].copy()
        if not excluded.empty:
            excluded.insert(0, "__対象外理由__", reasons.loc[excluded.index])
            # 参考用に正規化済みの保険者番号を添付
            excluded.insert(1, "__正規化保険者番号__", payer.loc[excluded.index])

        remains = base.loc[~ex_mask].copy()
        return remains, excluded

    except Exception:
        # 例外時は『全件残す/対象外は空』の安全側で返す
        try:
            return df.copy(), df.head(0).copy()
        except Exception:
            return pd.DataFrame(), pd.DataFrame()