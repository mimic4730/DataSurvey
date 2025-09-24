# core/inspection.py
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Iterable, Optional

import pandas as pd

# 出力カラム（検収CSVの仕様）
INSPECTION_COLUMNS = [
    "患者番号", "患者氏名カナ", "患者氏名", "性別", "生年月日", "年齢",
    "郵便番号", "住所１", "電話番号",
    "保険者番号", "患者負担割", "保険開始日", "保険証記号", "保険証番号",
    "公費負担者番号１", "公費受給者番号１", "公費開始日１",
    "公費負担者番号２", "公費受給者番号２", "公費開始日２",
]

# プリセット: 出力サブセット
COLUMNS_PATIENT = [
    "患者番号", "患者氏名カナ", "患者氏名", "性別", "生年月日", "年齢",
    "郵便番号", "住所１", "電話番号",
]
COLUMNS_INSURANCE = [
    "患者番号", "保険者番号", "患者負担割", "保険証記号", "保険証番号", "保険開始日",
]
COLUMNS_PUBLIC = [
    "患者番号", "公費負担者番号１", "公費受給者番号１", "公費開始日１",
    "公費負担者番号２", "公費受給者番号２", "公費開始日２",
]

# 既存importに続けて
ERA_MAP = {
    "M": 1868, "明治": 1868,
    "T": 1912, "大正": 1912,
    "S": 1926, "昭和": 1926,
    "H": 1989, "平成": 1989,
    "R": 2019, "令和": 2019,
}

# ========= 正規化ユーティリティ =========
def _to_halfwidth_digits(s: str) -> str:
    # 数字のみ半角へ（他はそのまま）
    # NFKCすると全角英数→半角になるため、先に保持したいものがあれば調整
    s = "".join(
        chr(ord(c) - 0xFEE0) if "０" <= c <= "９" else c
        for c in s
    )
    return s

def _to_fullwidth_ascii(s: str) -> str:
    # ASCII英字とスペースのみ全角化（他はそのまま維持）
    def conv(c: str) -> str:
        if c == " ":
            return "　"
        oc = ord(c)
        if 0x21 <= oc <= 0x7E:  # 可視ASCII
            return chr(oc + 0xFEE0)
        return c
    return "".join(conv(c) for c in s)

def _kana_keep_chōon_delete_others(s: str) -> str:
    # 伸ばし棒（全角/半角）以外の記号類を削除
    # 許容: カタカナ・ひらがな・全角スペース・半角スペース・伸ばし棒（ー/ｰ）
    # ※カナは全角に揃える前提で呼び出す
    return re.sub(r"[^\u30A0-\u30FF\u3040-\u309F 　ーｰ]", "", s)

def _normalize_kana_name(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw)
    # 1) 全角スペース→半角スペース
    s = s.replace("　", " ")
    # 2) 半角カナ→全角カナ、その他互換は基本NFKCで正規化
    s = unicodedata.normalize("NFKC", s)
    # 3) 伸ばし棒以外の記号を削除
    s = _kana_keep_chōon_delete_others(s)
    # 4) スペースの畳み込み
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_person_name(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw)
    # 半角カナ・スペース・英字は全角へ
    # - 先に半角カナ→全角（NFKC）
    s = unicodedata.normalize("NFKC", s)
    # - スペース/英字を全角化
    s = _to_fullwidth_ascii(s)
    # 余計なスペースは1個に
    s = re.sub(r"[ 　]+", "　", s).strip(" 　")
    return s

def _normalize_gender(raw: str) -> str:
    """
    男→1、女→2。その他は空欄。
    許容例: 男/男子/M/m/1 → 1、 女/女子/F/f/2 → 2
    """
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    if s in {"男", "男子", "m", "male", "1"}:
        return "1"
    if s in {"女", "女子", "f", "female", "2"}:
        return "2"
    return ""

def _parse_date_any_to_yyyymmdd(raw: str) -> str:
    """和暦(令和/平成/昭和/大正/明治, R/H/S/T/M)や西暦、区切り(年/月/日, -, /, .)対応で yyyymmdd に。失敗は空欄。"""
    if raw is None:
        return ""
    s = str(raw).strip()
    if s == "":
        return ""

    import re, unicodedata
    s = unicodedata.normalize("NFKC", s)

    # 8桁数字ならそのまま検証
    digits = re.sub(r"\D", "", s)
    if len(digits) == 8:
        try:
            datetime.strptime(digits, "%Y%m%d")
            return digits
        except Exception:
            pass

    # 和暦: R5.4.1 / 令和5年4月1日 / H01-04-01 など
    m = re.match(r"^(?P<era>R|H|S|T|M|令和|平成|昭和|大正|明治)\s*(?P<y>\d{1,2}|\d{1,4})[^0-9A-Za-z]*(?P<m>\d{1,2})[^0-9A-Za-z]*(?P<d>\d{1,2})$", s)
    if m:
        base = ERA_MAP.get(m.group("era"))
        if base:
            try:
                y = int(m.group("y")); mth = int(m.group("m")); day = int(m.group("d"))
                gy = base + (y - 1)  # 元年を+0換算
                dt = datetime(gy, mth, day)
                return dt.strftime("%Y%m%d")
            except Exception:
                return ""

    # 西暦 yyyy-any-mm-any-dd
    m = re.match(r"^(?P<y>\d{4})\D*(?P<m>\d{1,2})\D*(?P<d>\d{1,2})$", s)
    if not m and digits != s:
        m = re.match(r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})$", digits)
    if m:
        try:
            dt = datetime(int(m.group("y")), int(m.group("m")), int(m.group("d")))
            return dt.strftime("%Y%m%d")
        except Exception:
            return ""

    # 2桁年の扱いは施設規約が必要なため未対応（必要なら規則を追加）
    return ""

def _normalize_birth_yyyymmdd(raw: str) -> str:
    return _parse_date_any_to_yyyymmdd(raw)

def _calc_age(birth_yyyymmdd: str, today: Optional[date] = None) -> str:
    if not birth_yyyymmdd or len(birth_yyyymmdd) != 8:
        return ""
    today = today or date.today()
    y = int(birth_yyyymmdd[0:4])
    m = int(birth_yyyymmdd[4:6])
    d = int(birth_yyyymmdd[6:8])
    try:
        bd = date(y, m, d)
    except Exception:
        return ""
    age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    return str(age)

def _normalize_postal(raw: str) -> str:
    """ 郵便番号: 000-0000。7桁未満は空欄。 """
    if raw is None:
        return ""
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) != 7:
        return ""
    return f"{digits[:3]}-{digits[3:]}"

def _normalize_address(raw: str) -> str:
    """ 住所：数字を半角化。伸ばし棒（ー/ｰ）はどちらも許容（そのまま）。 """
    if raw is None:
        return ""
    s = str(raw)
    # 数字を半角に（他はそのまま）
    s = _to_halfwidth_digits(s)
    # 余計なスペースの正規化（必要に応じて調整）
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

def _zero_pad(s: str, width: int) -> str:
    digits = re.sub(r"\D", "", str(s))
    if not digits:
        return ""
    return digits.zfill(width)

# ========= 設定 =========
@dataclass
class InspectionConfig:
    patient_number_width: int = 10  # 患者番号の 0 埋め桁数（任意桁数とあったためデフォルトは10）
    today: Optional[date] = None    # 年齢計算のための基準日（デフォルトは実行日）

# ========= メイン：患者情報の検収 =========
def build_inspection_df(
    src_df: pd.DataFrame,
    colmap: Dict[str, str],
    config: Optional[InspectionConfig] = None,
    target_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    患者情報の検収用データフレームを生成する。

    Parameters
    ----------
    src_df : 入力DataFrame（元ファイル）
    colmap : 入力→検収カラム名の対応辞書
        例:
        {
            "患者番号": "patient_code",
            "患者氏名カナ": "kana",
            "患者氏名": "name",
            "性別": "sex",
            "生年月日": "birth",
            "郵便番号": "zip",
            "住所１": "address",
            "電話番号": "tel",
            ... 必要に応じて
        }
        ※存在しない場合は空欄で埋めます
    config : 検収設定（0埋め桁数など）

    Returns
    -------
    pd.DataFrame : INSPECTION_COLUMNS 順で整形済みのDataFrame
    """
    cfg = config or InspectionConfig()
    today = cfg.today or date.today()

    # 入力列を安全に取得するヘルパ
    def get(col_name: str) -> pd.Series:
        src_col = colmap.get(col_name)
        if src_col in src_df.columns:
            return src_df[src_col].astype(str)
        else:
            return pd.Series([""] * len(src_df), index=src_df.index, dtype="object")

    # 各列の正規化
    out = pd.DataFrame(index=src_df.index)

    # 患者番号（0埋め）
    out["患者番号"] = get("患者番号").map(lambda x: _zero_pad(x, cfg.patient_number_width))

    # カナ：全角スペース→半角 / 半角カナ→全角 / 記号削除（ーのみ許容）
    out["患者氏名カナ"] = get("患者氏名カナ").map(_normalize_kana_name)

    # 氏名：半角カナ・スペース・英字は全て全角に
    out["患者氏名"] = get("患者氏名").map(_normalize_person_name)

    # 性別：男→1 女→2
    out["性別"] = get("性別").map(_normalize_gender)

    # 生年月日：yyyymmdd
    out["生年月日"] = get("生年月日").map(_normalize_birth_yyyymmdd)

    # 年齢：実行日で算出
    out["年齢"] = out["生年月日"].map(lambda s: _calc_age(s, today=today))

    # 郵便番号：000-0000（7桁未満は空）
    out["郵便番号"] = get("郵便番号").map(_normalize_postal)

    # 住所：数字半角化、伸ばし棒はそのまま許容
    out["住所１"] = get("住所１").map(_normalize_address)

    # 電話番号：今回は仕様未指定のため、そのまま（必要なら整形ルール追加可）
    out["電話番号"] = get("電話番号")

    # 以降、保険・公費系
    out["保険者番号"] = get("保険者番号")
    out["患者負担割"] = get("患者負担割")
    out["保険証記号"] = get("保険証記号")
    out["保険証番号"] = get("保険証番号")
    # 開始日を正規化
    out["保険開始日"] = get("保険開始日").map(_parse_date_any_to_yyyymmdd)

    out["公費負担者番号１"] = get("公費負担者番号１")
    out["公費受給者番号１"] = get("公費受給者番号１")
    out["公費開始日１"] = get("公費開始日１").map(_parse_date_any_to_yyyymmdd)

    out["公費負担者番号２"] = get("公費負担者番号２")
    out["公費受給者番号２"] = get("公費受給者番号２")
    out["公費開始日２"] = get("公費開始日２").map(_parse_date_any_to_yyyymmdd)

    # カラム順を保証（カテゴリ指定があればそれを優先）
    cols_out = list(target_columns) if target_columns is not None else INSPECTION_COLUMNS
    out = out.reindex(columns=cols_out)

    return out

# ========= 便利関数 =========
def to_csv(
    df: pd.DataFrame,
    path: str,
    encoding: str | None = None,
    index: bool = False,
    encoding_errors: str = "replace",
) -> None:
    """
    CSV出力（プラットフォームに応じて既定エンコーディングを切替）
    - Windows: cp932（Excel互換）
    - 非Windows: utf-8-sig（Excel/Mac・他ツール互換）
    明示指定があれば `encoding` を優先。
    `errors` は Python 側で制御して安全に書き込みます。
    """
    import os, sys
    if encoding is None:
        # Windows なら cp932、その他は UTF-8 BOM 付き
        encoding = "cp932" if os.name == "nt" or sys.platform.startswith("win") else "utf-8-sig"

    with open(path, "w", encoding=encoding, errors=encoding_errors, newline="") as f:
        df.to_csv(f, index=index)