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
    check_pref_code_when_8digits: bool = True     # 8桁時の3-4桁(都道府県コード)が01..47か、または6桁の1−2桁
    duplicate_date_column: str | None = "最終確認日"  # 同一患者複数レコードの重複解消に使う日付列（colmap上の論理名）。Noneなら保険終了日を使用
    dummy_payer_prefixes: tuple[str, ...] = ()    # UI から渡される保険者番号ダミー法別(頭2桁)一覧。指定がなければ従来の列ベース設定を使用

@dataclass
class InsuranceKeyConfig:
    """
    検収や突合で使用する保険キー設定:
      - patient_width: 患者番号の0埋め桁数
      - payer_width  : 保険者番号の0埋め桁数
      - migration_yyyymmdd: 開始日などの補完に使う「データ移行日」（YYYYMMDD）
    """
    patient_width: int = 10
    payer_width: int = 8
    migration_yyyymmdd: str | None = None

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

# ---- 日付正規化: 開始日・終了日・確認日（移行ルール対応） ----
def normalize_insurance_dates_for_migration(
    start: pd.Series,
    end: pd.Series,
    confirm: pd.Series,
    migration_yyyymmdd: str | None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    元データ→移行データ変換時の『開始日・終了日・確認日』の正規化ルールを適用するヘルパ。

    仕様:
      - 開始日:
          値が 0 / 99999999 / 空白 の場合は「データ移行月の月初」に置換して移行する。
          それ以外はゆるい日付パース(inspection._parse_date_any_to_yyyymmdd)結果(YYYYMMDD)を使用。
      - 終了日:
          値が 0 / 99999999 / 空白 の場合は「空欄」で移行する。
          それ以外はゆるい日付パース結果(YYYYMMDD)を使用（99999999は空欄扱い）。
      - 確認日:
          値が 0 / 99999999 / 空白 の場合は「データ移行日(YYYYMMDD)」で移行する。
          それ以外はゆるい日付パース結果(YYYYMMDD)を使用。

    migration_yyyymmdd が None または空の場合は、開始日/終了日/確認日とも
    単純な _norm_date_yyyymmdd_or_empty による正規化のみにフォールバックする。
    """
    mig = (migration_yyyymmdd or "").strip() or None

    # データ移行日が不明な場合は、従来通りのゆるい正規化のみ行う
    if not mig:
        return (
            _norm_date_yyyymmdd_or_empty(start),
            _norm_date_yyyymmdd_or_empty(end),
            _norm_date_yyyymmdd_or_empty(confirm),
        )

    # データ移行月の月初 (YYYYMM01)
    mig_month_first = f"{mig[:6]}01"

    def _normalize_start(v: str) -> str:
        v = "" if v is None else str(v)
        digits = "".join(ch for ch in v if ch.isdigit())

        # 0 / 99999999 / 空白 → データ移行月の月初
        if digits == "" or set(digits) == {"0"} or set(digits) == {"9"}:
            return mig_month_first

        d = inspection._parse_date_any_to_yyyymmdd(digits or v)
        if not d:
            return mig_month_first
        s = str(d)
        if set(s) == {"9"} or set(s) == {"0"}:
            return mig_month_first
        return s

    def _normalize_end(v: str) -> str:
        v = "" if v is None else str(v)
        digits = "".join(ch for ch in v if ch.isdigit())

        # 0 / 99999999 / 空白 → 空欄
        if digits == "" or set(digits) == {"0"} or set(digits) == {"9"}:
            return ""

        d = inspection._parse_date_any_to_yyyymmdd(digits or v)
        if not d:
            return ""
        s = str(d)
        if set(s) == {"9"} or set(s) == {"0"}:
            return ""
        return s

    def _normalize_confirm(v: str) -> str:
        v = "" if v is None else str(v)
        digits = "".join(ch for ch in v if ch.isdigit())

        # 0 / 99999999 / 空白 → データ移行日
        if digits == "" or set(digits) == {"0"} or set(digits) == {"9"}:
            return mig

        d = inspection._parse_date_any_to_yyyymmdd(digits or v)
        if not d:
            return mig
        s = str(d)
        if set(s) == {"9"} or set(s) == {"0"}:
            return mig
        return s

    start_norm = start.map(_normalize_start)
    end_norm = end.map(_normalize_end)
    confirm_norm = confirm.map(_normalize_confirm)

    return start_norm.astype("object"), end_norm.astype("object"), confirm_norm.astype("object")

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

def normalize_insurance_keys(
    df: pd.DataFrame,
    colmap: Dict[str, str],
    cfg: InsuranceKeyConfig,
) -> pd.DataFrame:
    """
    保険検収・突合用のキー列を生成するユーティリティ。
      __key_patient__ : 患者番号（数字のみ・0埋め）
      __key_payer__   : 保険者番号（数字のみ・0埋め）
      __key_sym__     : 保険証番号（数字のみ）
      __key_start__   : 保険開始日（保険ルールに準じて正規化後の yyyymmdd / 空）
      __key_end__     : 保険終了日（同上）

    既存の inspection / inspection_actions から参照される想定。
    """
    import re as _re

    def _digits(v: object) -> str:
        if v is None:
            return ""
        return _re.sub(r"\D", "", str(v))

    def _zfill_or_empty(s: pd.Series, width: int) -> pd.Series:
        d = s.astype("object").map(_digits)
        return d.map(lambda v: v.zfill(width) if v else "")

    # 列名の解決（colmap は『論理名 → 実列名』のマップを想定）
    pat_col = colmap.get("患者番号")
    payer_col = colmap.get("保険者番号")
    sym_col = colmap.get("保険証番号")
    start_col = colmap.get("保険開始日")
    end_col = colmap.get("保険終了日")

    # 患者番号キー
    if pat_col and pat_col in df.columns:
        df["__key_patient__"] = _zfill_or_empty(df[pat_col], cfg.patient_width)
    else:
        df["__key_patient__"] = ""

    # 保険者番号キー
    if payer_col and payer_col in df.columns:
        df["__key_payer__"] = _zfill_or_empty(df[payer_col], cfg.payer_width)
    else:
        df["__key_payer__"] = ""

    # 保険証番号キー（記号＋番号統合は上位で済ませる想定。ここでは数字のみ）
    if sym_col and sym_col in df.columns:
        df["__key_sym__"] = df[sym_col].astype("object").map(_digits)
    else:
        df["__key_sym__"] = ""

    # 日付キー：開始・終了を保険ルールに沿って正規化（migration_yyyymmdd を考慮）
    empty_dates = pd.Series([""] * len(df), index=df.index, dtype="object")
    start_series = df[start_col] if start_col and start_col in df.columns else empty_dates
    end_series = df[end_col] if end_col and end_col in df.columns else empty_dates

    # 第3引数(confirm) はキー生成では使わないため空Seriesを渡す
    start_norm, end_norm, _ = normalize_insurance_dates_for_migration(
        start_series,
        end_series,
        empty_dates,
        cfg.migration_yyyymmdd,
    )
    df["__key_start__"] = start_norm.astype("object")
    df["__key_end__"] = end_norm.astype("object")

    return df

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
    ※ 保険者番号ダミー法別(頭2桁)は通常 UI から `InsuranceRuleConfig.dummy_payer_prefixes`
      経由で渡されます。旧仕様との互換のため、colmap に "保険者番号ダミーコード" を
      マッピングしている場合は、その列値からも自動的に読むフォールバックを残しています。
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

        # 保険者番号ダミーコード（頭2桁のダミー法別）を考慮して 8桁→6桁に落とす
        #   - 通常は UI から InsuranceRuleConfig.dummy_payer_prefixes 経由で '80','81','99' 等が渡される
        #   - 8桁保険者番号の先頭2桁がこのいずれかに一致した場合、頭2桁を削除して 6桁コードとして扱う
        #   - 旧仕様との互換のため、cfg.dummy_payer_prefixes が空の場合は
        #     colmap 上の「保険者番号ダミーコード」列からも同様の情報を読み取る
        try:
            import re as _re_dummy

            dummy_prefixes: set[str] = set()

            # 1) 新仕様: cfg.dummy_payer_prefixes（UI からの直接指定）を優先
            if getattr(cfg, "dummy_payer_prefixes", None):
                for part in cfg.dummy_payer_prefixes:
                    part_str = "".join(ch for ch in str(part) if ch.isdigit())
                    if len(part_str) >= 2:
                        dummy_prefixes.add(part_str[:2])

            if dummy_prefixes:
                def _strip_dummy_prefix(v: str) -> str:
                    if not v:
                        return v
                    try:
                        s = str(v)
                    except Exception:
                        s = ""
                    if len(s) == 8 and s[:2] in dummy_prefixes:
                        # 頭2桁ダミーを落として 6桁実コードとして扱う
                        return s[2:]
                    return s

                payer = payer.map(_strip_dummy_prefix).astype(str)
        except Exception:
            # ダミー処理に失敗しても他ルールには影響させない
            pass

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

        # 1) 患者番号 空（先頭ゼロ無視は重複解消ロジックでのみ使用）
        mask_pat_empty = (pat == "")
        pat_l = pat.str.lstrip("0")

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

        # 5) 都道府県コードチェック
        #    - 8桁: 3-4桁が都道府県コード(01..47)
        #    - 6桁: 1-2桁が都道府県コード(01..47)  ※ダミー法別削除後の6桁コードを想定
        if cfg.check_pref_code_when_8digits:
            allowed_pref = {f"{i:02d}" for i in range(1, 48)}
            mask_prefcode_invalid_8 = (payer.str.len() == 8) & (~payer.str[2:4].isin(allowed_pref))
            mask_prefcode_invalid_6 = (payer.str.len() == 6) & (~payer.str[:2].isin(allowed_pref))
            mask_prefcode_invalid = mask_prefcode_invalid_8 | mask_prefcode_invalid_6
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

        # 総合対象外（患者番号重複はここでは判定しない）
        ex_mask = (
            mask_pat_empty
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
        add_reason(mask_payer_empty, "保険者番号空欄")
        add_reason(mask_payer_len_ng, "保険者番号桁数不正(6または8桁以外)")
        if cfg.require_cardno_if_payer_present:
            add_reason(mask_cardno_empty, "保険証番号空欄")
        if cfg.enforce_law39_cardno_len8:
            add_reason(mask_law39_cardno_not8, "保険者番号(39xxxxxx)は保険証番号8桁のみ対象")
        add_reason(mask_payer_prefix_ng, "保険者番号先頭2桁が対象外")
        if cfg.check_pref_code_when_8digits:
            add_reason(mask_prefcode_invalid, "保険者番号都道府県コード不正")
        if mig:
            add_reason(mask_expired, f"保険終了日が移行日({mig})より前(期限切れ)")
        add_reason(mask_age75_law_not39, "75歳以上・法別≠39")

        # --- 同一患者＋有効な保険が複数ある場合の重複解消（日付順・昇順相当） ---
        try:
            # 対象: すでに対象外でなく、患者番号(先頭ゼロ除去)が空でない行
            cand_mask = (~ex_mask) & (pat_l != "")
            if cand_mask.any():
                # 重複解消に使う日付列（colmap上の論理名）。設定がなければ保険終了日を用いる。
                dup_logical_name = getattr(cfg, "duplicate_date_column", None)

                # 生のシリーズを取得（「最終確認日」など）
                if dup_logical_name:
                    raw_dup = col(dup_logical_name)
                else:
                    # 設定がなければ保険終了日で代用
                    raw_dup = col("保険終了日")

                # データ移行日（YYYYMMDD）。None の場合はランキング上は「00000000」を使う
                mig_dup = mig  # 上で計算した migration_yyyymmdd をそのまま利用

                # 0 / 99999999 / 空欄 は「データ移行日」とみなすための正規化
                def _dup_rank_value(v: str) -> str:
                    v = "" if v is None else str(v)
                    # 数字だけ取り出し
                    dv = "".join([ch for ch in v if ch.isdigit()])
                    # migration_yyyymmdd が分かっている場合:
                    #   - 完全に空
                    #   - 全部 0
                    #   - 全部 9
                    # は「移行日」として扱う
                    if mig_dup:
                        if dv == "" or set(dv) == {"0"} or set(dv) == {"9"}:
                            return mig_dup
                    # 上記以外は通常のゆるい日付パーサで YYYYMMDD 化
                    try:
                        d = inspection._parse_date_any_to_yyyymmdd(dv or v)
                        if not d:
                            return mig_dup or "00000000"
                        return str(d)
                    except Exception:
                        return mig_dup or "00000000"

                dup_date = raw_dup.map(_dup_rank_value).astype(str)

                helper = pd.DataFrame(
                    {
                        "pat_key": pat_l,
                        "date": dup_date,
                        "_row": base.index,
                    },
                    index=base.index,
                )
                helper = helper.loc[cand_mask]

                # 患者キー＋日付昇順＋行番号昇順でソートし、「最後の行」だけ残す（＝最も新しい最終確認日を採用）
                helper_sorted = helper.sort_values(
                    by=["pat_key", "date", "_row"],
                    ascending=[True, True, True],
                )
                keep_idx = helper_sorted.groupby("pat_key").tail(1).index

                # keep_idx 以外を対象外へ
                dup_reject_mask = cand_mask & (~base.index.isin(keep_idx))
                if dup_reject_mask.any():
                    ex_mask |= dup_reject_mask
                    add_reason(dup_reject_mask, "同一患者複数有効保険のうち非優先")

        except Exception:
            # 重複解消でエラーが出ても、他のルールには影響させない
            pass

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