# core/naming.py
import re, unicodedata
from typing import Iterable, Optional

# 「患者コード」系の候補（正規化後に使うパターン）
_PATIENT_CODE_KEYS = [
    "患者コード", "患者番号", "患者id", "患者ｺｰﾄﾞ",
    "patientcode", "patient_id", "patientid", "id", "コード"
]

def normalize_text(s: str) -> str:
    # 全角→半角含むNFKC、空白/アンダースコア/ハイフン等を除去、大小無視
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = re.sub(r"[\s_\-‐-–—―･・．。.,/\\()\[\]{}:：;；|｜]+", "", s)
    return s.casefold()

def find_patient_code_column(columns: Iterable[str]) -> Optional[str]:
    """
    与えられた列名群から患者コード列を推定して原名を返す。
    - 正規化して候補と比較
    - 完全一致＞部分一致の優先
    """
    cols = list(columns)
    norm_map = {c: normalize_text(c) for c in cols}

    # 完全一致優先
    targets = [normalize_text(k) for k in _PATIENT_CODE_KEYS]
    for orig, norm in norm_map.items():
        if norm in targets:
            return orig

    # 部分一致フォールバック（例: "患者IDコード" など）
    for orig, norm in norm_map.items():
        if any(t in norm for t in targets):
            return orig

    return None