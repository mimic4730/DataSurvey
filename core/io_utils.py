# core/io_utils.py
from pathlib import Path
import pandas as pd
from charset_normalizer import from_path as cn_from_path
from .file_keys import ENCODING_CANDIDATES
from .naming import find_patient_code_column

class CsvLoader:
    @staticmethod
    def _score_header(bytes_data: bytes, encoding: str) -> int:
        """
        指定encodingで最初の1行/2行をデコードし、患者コード候補列が検出できればスコア加点。
        """
        try:
            text = bytes_data.decode(encoding, errors="strict")
        except Exception:
            return -1
        head = text.splitlines()[:2]
        if not head:
            return 0
        # 区切り推定（, or \t）
        import csv, io
        sample = "\n".join(head)
        for sep in [",", "\t", ";", "|"]:
            reader = csv.reader(io.StringIO(sample), delimiter=sep)
            try:
                row = next(reader, None)
            except Exception:
                continue
            if row:
                # 列候補で患者コード列が見つかるか
                if find_patient_code_column(row):
                    return 2  # 高スコア
        return 0

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
        文字コード自動推定→(ヘッダ一致スコアで補正)→候補総当たり + 区切り自動判定
        """
        p = Path(path)
        data = p.read_bytes()

        # 1) charset-normalizer の推定
        enc_detected = CsvLoader.detect_encoding(path)

        # 2) ヘッダ・スコアで最適encodingを選択
        enc_list = []
        if enc_detected:
            enc_list.append(enc_detected)
        # 既知候補を追加（重複除去）
        for e in ENCODING_CANDIDATES:
            if e.lower() not in [x.lower() for x in enc_list]:
                enc_list.append(e)

        scored = sorted(
            [(e, CsvLoader._score_header(data, e)) for e in enc_list],
            key=lambda x: x[1],
            reverse=True
        )
        tried = []

        def _try(encod: str):
            return pd.read_csv(path, dtype=str, encoding=encod, sep=None, engine="python")

        # スコア高い順（患者コード列が読める可能性の高い順）に試す
        for enc, _score in scored:
            try:
                return _try(enc)
            except Exception as e:
                tried.append((enc, str(e)))

        # すべて失敗
        msg = "CSVの読み込みに失敗しました。\n試したエンコーディング:\n"
        msg += "\n".join([f"- {e} : {err[:120]}..." for e, err in tried])
        raise RuntimeError(msg)