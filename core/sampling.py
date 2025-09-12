# core/sampling.py
import pandas as pd
import unicodedata, re
from .file_keys import COL_KUBUN, COL_LOCAL_PUB_UUID, SAMPLE_SIZE

def normalize_patient_code_series(s: pd.Series) -> pd.Series:
    return s.dropna().map(lambda x: str(x).strip())

def sample_unique_by_col(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    """
    col（例: 区分, 地方公費制度UUID）の値が重複しないように最大n行サンプル。
    足りなければ、残りはランダムに補完（重複許容）。
    """
    if col not in df.columns:
        return df.sample(n=min(n, len(df)), random_state=None)

    work = df[df[col].astype(str).map(lambda x: x.strip() != "")]
    if work.empty:
        return df.sample(n=min(n, len(df)), random_state=None)

    top_by_value = work.drop_duplicates(subset=[col], keep="first")
    base = top_by_value.sample(n=min(n, len(top_by_value)), random_state=None)

    need = n - len(base)
    if need > 0:
        rest = df.drop(index=base.index)
        if not rest.empty:
            add = rest.sample(n=min(need, len(rest)), random_state=None)
            base = pd.concat([base, add], ignore_index=False)

    if len(base) > n:
        base = base.sample(n=n, random_state=None)

    return base

def sample_by_patient_code(df: pd.DataFrame, code_col: str, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    患者コードをユニーク化し、その先頭1行から最大n行サンプル。
    """
    df2 = df.copy()
    df2[code_col] = df2[code_col].map(lambda x: "" if x is None else str(x).strip())
    df2 = df2[df2[code_col] != ""]
    df_unique = df2.drop_duplicates(subset=[code_col], keep="first")
    if df_unique.empty:
        return df.head(0)
    return df_unique.sample(n=min(n, len(df_unique)), random_state=None)

def normalize_patient_code_series(s: pd.Series) -> pd.Series:
    def _norm(x):
        x = unicodedata.normalize("NFKC", str(x)).strip()
        # Unicodeの数字以外を最小限に（必要に応じて調整）
        x = re.sub(r"[^\w\-]", "", x)  # 記号類の大半削除（院の方針に合わせ調整可）
        return x
    return s.dropna().map(_norm)