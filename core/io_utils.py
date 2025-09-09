# core/io_utils.py
from pathlib import Path
import pandas as pd
from charset_normalizer import from_path as cn_from_path
from .file_keys import ENCODING_CANDIDATES

class CsvLoader:
    @staticmethod
    def detect_encoding(path: str) -> str | None:
        try:
            result = cn_from_path(path)
            if not result:
                return None
            best = result.best()
            if best and best.encoding:
                return best.encoding
        except Exception:
            return None
        return None

    @staticmethod
    def read_csv_flex(path: str) -> pd.DataFrame:
        """
        文字コード自動推定→候補総当たり + 区切り自動判定
        """
        enc = CsvLoader.detect_encoding(path)
        tried = []

        def _try(encod: str):
            return pd.read_csv(path, dtype=str, encoding=encod, sep=None, engine="python")

        if enc:
            try:
                return _try(enc)
            except Exception as e:
                tried.append((enc, str(e)))

        for enc2 in ENCODING_CANDIDATES:
            if enc and enc2.lower() == enc.lower():
                continue
            try:
                return _try(enc2)
            except Exception as e:
                tried.append((enc2, str(e)))

        msg = "CSVの読み込みに失敗しました。\n試したエンコーディング:\n"
        msg += "\n".join([f"- {e} : {err[:120]}..." for e, err in tried])
        raise RuntimeError(msg)