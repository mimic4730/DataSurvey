# core/rules/public.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict
from core import inspection

"""
公費情報の『移行対象外』と『内容検収（比較）』を単一ソースで提供するモジュール。
- 検収CSV生成フローから直接呼び出せる run_public_content_integrated(...) を提供します。
"""
from pathlib import Path

@dataclass
class PublicRuleConfig:
    migration_yyyymmdd: str | None = None  # 期限切れ判定基準。Noneなら期限切れ判定は無効


@dataclass
class PublicComparePolicy:
    """
    公費の内容検収における比較ポリシー。
    """
    compare_key_uses_end_if_src_present: bool = True  # 元の終了日が空ならキーに含めない
    fill_start_with_migration_month_first: bool = True  # 開始日が空なら移行月初(YYYYMM01)で比較
    cap_future_start_to_migration: bool = True  # 未来開始日を移行日にキャップ

def _normalize_codes(s: pd.Series, width: int, mode: str = "zfill") -> pd.Series:
    import re, unicodedata
    digits = s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x)))
    if mode == "zfill":
        return digits.map(lambda x: x.zfill(width) if x else "")
    elif mode == "lstrip":
        return digits.map(lambda x: x.lstrip("0"))
    return digits

def _digits_len_max(s: pd.Series) -> int:
    import re, unicodedata
    if s is None or len(s) == 0:
        return 0
    return int(s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x))).str.len().max() or 0)

def _unpivot_public_slots(src: pd.DataFrame, mapping: Dict[str, str], slot: int) -> pd.DataFrame:
    """
    1/2スロット（負担者/受給者/開始/終了）を縦持ち化する。mapping は列名の対応を受け取る。
    必須キー: '患者番号', '公費負担者番号{slot}', '公費受給者番号{slot}', '公費開始日{slot}', '公費終了日{slot}'
    """
    col_pat = mapping.get("患者番号")
    col_g   = mapping.get(f"公費負担者番号{slot}")
    col_r   = mapping.get(f"公費受給者番号{slot}")
    col_s   = mapping.get(f"公費開始日{slot}")
    col_e   = mapping.get(f"公費終了日{slot}")
    n = len(src)
    def _col(c):
        return src[c] if c and c in src.columns else pd.Series([""] * n, index=src.index, dtype="object")
    out = pd.DataFrame({
        "患者番号": _col(col_pat),
        "公費負担者番号": _col(col_g),
        "公費受給者番号": _col(col_r),
        "公費開始日": _col(col_s),
        "公費終了日": _col(col_e),
    }, index=src.index).copy()
    out["__slot__"] = slot
    return out

def _read_csv_flex(path: str) -> pd.DataFrame:
    import pandas as _pd
    try:
        from ui.csv_loader import CsvLoader  # あれば利用
        return CsvLoader.read_csv_flex(path)
    except Exception:
        pass

    # まず普通に読む
    try:
        return _pd.read_csv(path, dtype=str, encoding="utf-8", engine="python")
    except Exception:
        pass

    # cp932 は open() で decode エラーを握りつぶしてから read_csv に渡す（互換性高い）
    try:
        with open(path, "r", encoding="cp932", errors="ignore", newline="") as f:
            return _pd.read_csv(f, dtype=str, engine="python")
    except Exception:
        # 最後の保険（utf-8でも同様に）
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            return _pd.read_csv(f, dtype=str, engine="python")

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


# === 追加: 公費内容検収比較・統合実行 ===
def _alias_slots_map(colmap_src: Dict[str, str]) -> Dict[str, str]:
    """
    '１/２' と '1/2' のゆれを解消して、足りないキーを補う。
    """
    m = dict(colmap_src or {})
    repl = {
        "公費負担者番号１": "公費負担者番号1",
        "公費受給者番号１": "公費受給者番号1",
        "公費開始日１":   "公費開始日1",
        "公費終了日１":   "公費終了日1",
        "公費負担者番号２": "公費負担者番号2",
        "公費受給者番号２": "公費受給者番号2",
        "公費開始日２":   "公費開始日2",
        "公費終了日２":   "公費終了日2",
    }
    for k_j, k_h in repl.items():
        if k_j in m and k_h not in m:
            m[k_h] = m[k_j]
    return m

def _prepare_longform_from_wide(
    df: pd.DataFrame,
    colmap_src: Dict[str, str],
    width: int,
    mig: str | None,
    policy: PublicComparePolicy
) -> pd.DataFrame:
    m = _alias_slots_map(colmap_src)
    src_1 = _unpivot_public_slots(df, {
        "患者番号": m.get("患者番号"),
        "公費負担者番号1": m.get("公費負担者番号1"),
        "公費受給者番号1": m.get("公費受給者番号1"),
        "公費開始日1": m.get("公費開始日1"),
        "公費終了日1": m.get("公費終了日1"),
    }, slot=1)
    src_2 = _unpivot_public_slots(df, {
        "患者番号": m.get("患者番号"),
        "公費負担者番号2": m.get("公費負担者番号2"),
        "公費受給者番号2": m.get("公費受給者番号2"),
        "公費開始日2": m.get("公費開始日2"),
        "公費終了日2": m.get("公費終了日2"),
    }, slot=2)
    out = pd.concat([src_1, src_2], axis=0, ignore_index=True)

    # 正規化
    out["患者番号"]     = _normalize_codes(out["患者番号"], width, mode="zfill")
    out["公費負担者番号"] = _digits_only(out["公費負担者番号"])
    out["公費受給者番号"] = _digits_only(out["公費受給者番号"])
    out["公費開始日"]   = _date_norm(out["公費開始日"])
    out["公費終了日"]   = _date_norm(out["公費終了日"])

    # 開始日の補完・未来キャップ
    if mig:
        if policy.fill_start_with_migration_month_first and (out["公費開始日"].notna().any() or True):
            out["公費開始日"] = out["公費開始日"].map(lambda v: (mig[:6] + "01") if (str(v) == "") else v)
        if policy.cap_future_start_to_migration:
            out["公費開始日"] = _cap_future_to_migration(out["公費開始日"], mig)

    return out

def _build_cmp_longform(
    cmp_df: pd.DataFrame,
    width: int
) -> pd.DataFrame:
    """
    突合側の列名ゆれを吸収し、1/2スロットを縦持ち化して正規化。
    """
    # 列名ゆれ
    aliases = {
        "患者番号": ["患者番号"],
        "公費負担者番号1": ["公費負担者番号1","公費負担者番号１","第１公費負担者番号","第一公費負担者番号","負担者番号1","負担者番号１","負担者番号"],
        "公費受給者番号1": ["公費受給者番号1","公費受給者番号１","受給者番号1","受給者番号１","受給者番号"],
        "公費開始日1":     ["公費開始日1","公費開始日１","開始日1","開始日１","開始日"],
        "公費終了日1":     ["公費終了日1","公費終了日１","終了日1","終了日１","終了日"],
        "公費負担者番号2": ["公費負担者番号2","公費負担者番号２","負担者番号2","負担者番号２"],
        "公費受給者番号2": ["公費受給者番号2","公費受給者番号２","受給者番号2","受給者番号２"],
        "公費開始日2":     ["公費開始日2","公費開始日２","開始日2","開始日２"],
        "公費終了日2":     ["公費終了日2","公費終了日２","終了日2","終了日２"],
    }
    def pick(name: str) -> pd.Series:
        for k in aliases.get(name, []):
            if k in cmp_df.columns:
                return cmp_df[k]
        return pd.Series([""] * len(cmp_df))
    cmp_1 = pd.DataFrame({
        "患者番号": pick("患者番号"),
        "公費負担者番号": pick("公費負担者番号1"),
        "公費受給者番号": pick("公費受給者番号1"),
        "公費開始日": pick("公費開始日1"),
        "公費終了日": pick("公費終了日1"),
    })
    cmp_2 = pd.DataFrame({
        "患者番号": pick("患者番号"),
        "公費負担者番号": pick("公費負担者番号2"),
        "公費受給者番号": pick("公費受給者番号2"),
        "公費開始日": pick("公費開始日2"),
        "公費終了日": pick("公費終了日2"),
    })
    out = pd.concat([cmp_1, cmp_2], axis=0, ignore_index=True)

    # 正規化
    out["患者番号"]     = _normalize_codes(out["患者番号"], width, mode="zfill")
    out["公費負担者番号"] = _digits_only(out["公費負担者番号"])
    out["公費受給者番号"] = _digits_only(out["公費受給者番号"])
    out["公費開始日"]   = _date_norm(out["公費開始日"])
    out["公費終了日"]   = _date_norm(out["公費終了日"])
    # 空患者/負担者は除外
    out = out[(out["患者番号"] != "") & (out["公費負担者番号"] != "")].copy()
    # 完全重複の除去
    subset_cols = ["患者番号","公費負担者番号","公費受給者番号","公費開始日","公費終了日"]
    out = out.drop_duplicates(subset=subset_cols, keep="first")
    return out

def compare_public_content(
    src_long: pd.DataFrame,
    cmp_long: pd.DataFrame,
    policy: PublicComparePolicy
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    return: (matched_df, mismatch_df, missing_df)
      matched: キー一致 & （src_end=="" or src_end==cmp_end）
      mismatch: キー一致 & src_end!="" & src_end!=cmp_end（項目名='公費終了日'）
      missing: srcにあるキーがcmpに無い（対象外キーは呼び出し側で除外してから渡す想定）
    """
    # 未ヒット（終了日は src が空ならキーに含めないポリシー）
    def _key_rows(df: pd.DataFrame, use_src_rule: bool) -> list[tuple]:
        keys = []
        ends = df.get("公費終了日", pd.Series([""] * len(df)))
        for p, g, r, s, e in zip(df["患者番号"], df["公費負担者番号"], df["公費受給者番号"], df["公費開始日"], ends):
            e_norm = (e if str(e) != "" else None)
            if use_src_rule and policy.compare_key_uses_end_if_src_present:
                k = (p, g, r, s, e_norm if e_norm else None)
            else:
                k = (p, g, r, s, e_norm)
            keys.append(k)
        return keys

    src_keys = _key_rows(src_long, use_src_rule=True)
    cmp_keys = set(_key_rows(cmp_long, use_src_rule=False))
    missing_mask = pd.Series([k not in cmp_keys for k in src_keys], index=src_long.index)
    missing_df = src_long.loc[missing_mask].copy()

    # 一致/不一致（終了日は src が空なら不問）
    merged = src_long.merge(
        cmp_long,
        on=["患者番号", "公費負担者番号", "公費受給者番号", "公費開始日"],
        how="inner",
        suffixes=("_src", "_cmp")
    )
    src_end = merged["公費終了日_src"].fillna("")
    cmp_end = merged["公費終了日_cmp"].fillna("")
    eq_end = (src_end == "") | (src_end == cmp_end)

    matched_df = merged.loc[eq_end, [
        "患者番号","公費負担者番号","公費受給者番号","公費開始日","公費終了日_src"
    ]].copy()
    matched_df.rename(columns={"公費終了日_src": "公費終了日"}, inplace=True)

    mismatch_df = merged.loc[~eq_end, [
        "患者番号","公費負担者番号","公費受給者番号","公費開始日","公費終了日_src","公費終了日_cmp"
    ]].copy()
    if not mismatch_df.empty:
        mismatch_df.insert(2, "項目名", "公費終了日")
        mismatch_df.rename(columns={"公費終了日_src": "正規化_元", "公費終了日_cmp": "正規化_突合"}, inplace=True)
    else:
        mismatch_df = pd.DataFrame(columns=["患者番号","公費負担者番号","項目名","正規化_元","公費受給者番号","公費開始日","正規化_突合"])

    return matched_df, mismatch_df, missing_df

def run_public_content_integrated(
    *,
    src_df: pd.DataFrame,
    colmap_src: Dict[str, str],
    cmp_path: str,
    out_dir: Path,
    logger=None,
    migration_yyyymmdd: str | None = None,
    policy: PublicComparePolicy | None = None
) -> Dict[str, str | int]:
    """
    公費の内容検収を rules 層で完結させる統合実行。
    - 出力は out_dir 直下に『公費_内容_*_YYYYMMDD.csv』を作成。
    """
    def _log(msg: str):
        if callable(logger):
            try:
                logger(msg)
            except Exception:
                pass

    tag = inspection._dt_now_str() if hasattr(inspection, "_dt_now_str") else pd.Timestamp.now().strftime("%Y%m%d")
    out_dir = Path(out_dir)

    # 1) 突合CSVロード
    cmp_df = _read_csv_flex(cmp_path)

    # 2) 患者番号の幅推定
    src_pat = src_df[colmap_src.get("患者番号")] if colmap_src.get("患者番号") in src_df.columns else pd.Series([], dtype="object")
    cmp_pat = cmp_df["患者番号"] if "患者番号" in cmp_df.columns else pd.Series([], dtype="object")
    width = max(_digits_len_max(src_pat), _digits_len_max(cmp_pat), 1)
    _log(f"[公費-内容] 患者番号幅: {width}")

    # 3) 広→縦（元/突合）＋ 正規化
    mig = inspection._parse_date_any_to_yyyymmdd(migration_yyyymmdd) if migration_yyyymmdd else None
    pol = policy or PublicComparePolicy()
    src_long = _prepare_longform_from_wide(src_df, colmap_src, width, mig, pol)
    cmp_long = _build_cmp_longform(cmp_df, width)

    # 4) 対象外抽出（比較母集団から除外）
    proc_colmap = {"患者番号":"患者番号","公費負担者番号":"公費負担者番号","公費受給者番号":"公費受給者番号","公費開始日":"公費開始日","公費終了日":"公費終了日"}
    remains_df, excluded_df = evaluate_public_exclusions(src_long, proc_colmap, PublicRuleConfig(migration_yyyymmdd=mig))
    if not isinstance(remains_df, pd.DataFrame):
        remains_df = src_long
    src_long = remains_df
    excluded_len = 0
    out_excluded = out_dir / f"公費_内容_対象外_{tag}.csv"
    if isinstance(excluded_df, pd.DataFrame) and not excluded_df.empty:
        excluded_len = len(excluded_df)
        inspection.to_csv(excluded_df, str(out_excluded))
        _log(f"[公費-内容] 対象外: {excluded_len} 件 → {out_excluded}")

    # 5) 比較
    matched_df, mismatch_df, missing_df = compare_public_content(src_long, cmp_long, pol)

    # 対象外キーの未ヒット除外
    if excluded_len and not missing_df.empty:
        ex_keys = set(zip(excluded_df["患者番号"], excluded_df["公費負担者番号"]))
        _miss_keys = list(zip(missing_df["患者番号"], missing_df["公費負担者番号"]))
        keep_mask = pd.Series([k not in ex_keys for k in _miss_keys], index=missing_df.index)
        missing_df = missing_df.loc[keep_mask].copy()

    # 6) 出力
    out_matched  = out_dir / f"公費_内容_一致_{tag}.csv"
    out_mismatch = out_dir / f"公費_内容_不一致_{tag}.csv"
    out_missing  = out_dir / f"公費_内容_未ヒット_{tag}.csv"
    inspection.to_csv(matched_df,  str(out_matched))
    inspection.to_csv(mismatch_df, str(out_mismatch))
    inspection.to_csv(missing_df,  str(out_missing))

    _log(f"[公費-内容] 一致: {len(matched_df)} / 不一致明細: {len(mismatch_df)} / 未ヒット: {len(missing_df)} / 対象外: {excluded_len}")

    return {
        "matched_path": str(out_matched),
        "mismatch_path": str(out_mismatch),
        "missing_path": str(out_missing),
        "excluded_path": str(out_excluded),
        "matched_count": int(len(matched_df)),
        "mismatch_count": int(len(mismatch_df)),
        "missing_count": int(len(missing_df)),
        "excluded_count": int(excluded_len),
    }