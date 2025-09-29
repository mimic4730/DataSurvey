from __future__ import annotations
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd
from datetime import datetime as _dt
from core.io_utils import CsvLoader
from core import inspection


class PatientContentChecker:
    """患者情報の【内容】検収（項目値の一致判定）を行うクラス。
    - 元CSV と 突合CSV（検収用/他システム出力）を読み、
      キー=患者番号 で突合し、項目ごとの一致/不一致を判定して出力します。
    - 元CSV 側の電話番号は、空欄なら携帯電話番号で補完して比較します。
    - 変換・正規化ルールは要件に従って実装。
    """

    def __init__(self, logger=None, preset_colmap: dict | None = None):
        self._logger = logger
        self._preset = preset_colmap or {}

    # === ログユーティリティ ===
    def log(self, msg: str):
        if self._logger:
            try:
                self._logger(msg)
            except Exception:
                pass

    # === 出力ディレクトリ作成 ===
    def _prepare_output_dir(self, in_path: str | Path, kind: str) -> Path:
        base = Path(in_path).resolve().parent
        tag = _dt.now().strftime("%Y%m%d")
        out_dir = base / "検収結果" / f"{kind}_内容_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # === 正規化ヘルパ ===
    def _normalize_codes(self, s: pd.Series, width: int, mode: str = "zfill") -> pd.Series:
        """患者番号の正規化
        mode:
          - "zfill": 数字以外除去→ゼロ埋め
          - "lstrip": 数字以外除去→先頭0除去
          - "rawdigits": 数字以外除去のみ
        """
        import re
        digits = s.astype(str).map(lambda x: re.sub(r"\D", "", x))
        if mode == "zfill":
            return digits.map(lambda x: x.zfill(width) if x else "")
        elif mode == "lstrip":
            return digits.map(lambda x: x.lstrip("0"))
        return digits

    @staticmethod
    def _norm_kana_for_compare(s: str) -> str:
        import unicodedata, re
        if s is None:
            return ""
        t = unicodedata.normalize("NFKC", str(s))
        # 中黒を半角スペースへ
        t = t.replace("・", " ").replace("･", " ")
        # スペースは半角に統一
        t = t.replace("　", " ")
        # 連続スペースを1つに
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _norm_name_for_compare(s: str) -> str:
        import unicodedata, re
        if s is None:
            return ""
        t = unicodedata.normalize("NFKC", str(s))
        t = t.replace("　", " ")
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _norm_gender_for_compare(s: str) -> str:
        if s is None:
            return ""
        t = str(s).strip().lower()
        if t in {"男", "男子", "m", "male", "1"}:
            return "1"
        if t in {"女", "女子", "f", "female", "2"}:
            return "2"
        return ""

    @staticmethod
    def _norm_birth_for_compare(s: str) -> str:
        return inspection._parse_date_any_to_yyyymmdd(s)

    @staticmethod
    def _norm_age_from_birth(yyyymmdd: str) -> str:
        from datetime import date
        if not yyyymmdd or len(yyyymmdd) != 8:
            return ""
        try:
            y, m, d = int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])
            bd = date(y, m, d)
            today = date.today()
            age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
            return str(age)
        except Exception:
            return ""

    @staticmethod
    def _norm_zip_for_compare(s: str) -> str:
        import re, unicodedata
        if s is None:
            return ""
        digits = re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", str(s)))
        return f"{digits[:3]}-{digits[3:]}" if len(digits) == 7 else ""

    @staticmethod
    def _norm_addr_for_compare(s: str) -> str:
        # 数字は半角化・記号はそのまま（ただしハイフン類は統一）・スペース正規化
        import re, unicodedata
        if s is None:
            return ""
        t = str(s)
        # Unicode正規化（記号の統一に備える）
        t = unicodedata.normalize("NFKC", t)
        # 数字を半角へ（他はそのまま）
        t = "".join(chr(ord(c) - 0xFEE0) if "０" <= c <= "９" else c for c in t)
        # アポストロフィ類（全角/タイプグラフィクォート/プライム等）を ASCII アポストロフィに統一
        # 例：M’s → M's, J’s- → J's-
        t = re.sub(r"[’‘＇´`ˈʹʽʾʿ]", "'", t)
        # ハイフン類（全角/半角/長音符/ダッシュ等）を ASCII ハイフンに統一
        t = re.sub(r"[‐‑‒–—―ーｰ－−]", "-", t)
        # 連続ハイフンは1つに
        t = re.sub(r"-{2,}", "-", t)
        # スペースは半角に揃え、連続は1つに
        t = t.replace("　", " ")
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def _norm_tel_for_compare(s: str) -> str:
        import unicodedata, re
        if s is None:
            return ""
        digits = re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", str(s)))
        return digits  # ハイフン差異を吸収するため数字のみ

    @staticmethod
    def _pick_phone_with_mobile_fallback(df: pd.DataFrame, tel_col: str | None, mobile_col: str | None) -> pd.Series:
        # 元CSV限定：電話が空なら携帯で埋める
        tel = pd.Series([""
                         ] * len(df), index=df.index, dtype="object")
        if tel_col and tel_col in df.columns:
            tel = df[tel_col].astype(str)
        if mobile_col and mobile_col in df.columns:
            mobile = df[mobile_col].astype(str)
            use_mobile = tel.map(lambda x: str(x).strip() == "")
            tel = tel.mask(use_mobile, other=mobile)
        return tel

    # === メイン処理 ===
    def run(self, app):
        # 1) 元CSV
        in_path = filedialog.askopenfilename(title="患者情報（元CSV）を選択", filetypes=[("CSV files", "*.csv")])
        self.log(f"[患者-内容] 元CSV: {in_path}")
        if not in_path:
            return
        src = CsvLoader.read_csv_flex(in_path)
        out_dir = self._prepare_output_dir(in_path, "患者")

        # 2) マッピング（携帯電話(任意)も追加）
        required = list(inspection.COLUMNS_PATIENT)
        if "携帯電話番号" not in required:
            required.append("携帯電話番号")
        colmap = app._ask_inspection_colmap(src, required_cols=required, preset=self._preset)
        if colmap is None:
            return
        self.log(f"[患者-内容] 元マッピング: {colmap}")

        # 3) 突合CSV（検収用=固定カラム想定）
        cmp_path = filedialog.askopenfilename(title="突合用（検収CSV/他システム出力）を選択", filetypes=[("CSV files", "*.csv")])
        self.log(f"[患者-内容] 突合CSV: {cmp_path}")
        if not cmp_path:
            return
        cmp_df = CsvLoader.read_csv_flex(cmp_path)

        # 4) 患者番号の桁幅（ログ用）
        import unicodedata, re

        def _digits_len_max(s: pd.Series) -> int:
            return int(s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x))).str.len().max() or 0)

        src_pat = src[colmap.get("患者番号")] if colmap.get("患者番号") in src.columns else pd.Series([], dtype="object")
        cmp_pat = cmp_df["患者番号"] if "患者番号" in cmp_df.columns else pd.Series([], dtype="object")
        src_w = _digits_len_max(src_pat) if len(src_pat) else 0
        cmp_w = _digits_len_max(cmp_pat) if len(cmp_pat) else 0
        width = max(src_w, cmp_w, 1)
        self.log(f"[患者-内容] 患者番号桁数: 元={src_w} / 突合={cmp_w} / 使用幅={width}")

        # 5) 比較用 正規化データ作成（元CSV側）
        src_k = src[colmap.get("患者番号")] if colmap.get("患者番号") in src.columns else pd.Series([""] * len(src))
        src_key = self._normalize_codes(src_k.astype(str), width, mode="zfill")
        src_kana = src[colmap.get("患者氏名カナ")] if colmap.get("患者氏名カナ") in src.columns else pd.Series([""] * len(src))
        src_kana = src_kana.map(self._norm_kana_for_compare)
        src_name = src[colmap.get("患者氏名")] if colmap.get("患者氏名") in src.columns else pd.Series([""] * len(src))
        src_name = src_name.map(self._norm_name_for_compare)
        # 氏名：空欄ならカナ氏名で補完（比較用）
        src_name_filled = src_name.copy()
        try:
            src_name_filled = src_name_filled.mask(src_name_filled.str.strip() == "", other=src_kana)
        except Exception:
            # 安全策（strアクセサ例外時）
            src_name_filled = src_name_filled.map(lambda v, k=src_kana: v if str(v).strip() != "" else "").astype(str)
            src_name_filled = src_name_filled.where(src_name_filled != "", other=src_kana)
        src_sex = src[colmap.get("性別")] if colmap.get("性別") in src.columns else pd.Series([""] * len(src))
        src_sex = src_sex.map(self._norm_gender_for_compare)
        src_birth = src[colmap.get("生年月日")] if colmap.get("生年月日") in src.columns else pd.Series([""] * len(src))
        src_birth = src_birth.map(self._norm_birth_for_compare)
        src_age = src_birth.map(self._norm_age_from_birth)
        src_zip = src[colmap.get("郵便番号")] if colmap.get("郵便番号") in src.columns else pd.Series([""] * len(src))
        src_zip = src_zip.map(self._norm_zip_for_compare)
        src_addr = src[colmap.get("住所１")] if colmap.get("住所１") in src.columns else pd.Series([""] * len(src))
        src_addr = src_addr.map(self._norm_addr_for_compare)
        src_tel_series = self._pick_phone_with_mobile_fallback(src, colmap.get("電話番号"), colmap.get("携帯電話番号"))
        src_tel = src_tel_series.map(self._norm_tel_for_compare)

        src_norm = pd.DataFrame({
            "患者番号": src_key,
            "患者氏名カナ": src_kana,
            "患者氏名": src_name_filled,
            "性別": src_sex,
            "生年月日": src_birth,
            "年齢": src_age,
            "郵便番号": src_zip,
            "住所１": src_addr,
            "電話番号": src_tel,
        })

        # 6) 突合側（検収CSV想定: 既に固定カラム名）
        if "患者番号" not in cmp_df.columns:
            messagebox.showerror("エラー", "突合CSVに『患者番号』列がありません。")
            return

        cmp_key = self._normalize_codes(cmp_df["患者番号"].astype(str), width, mode="zfill")

        # 突合CSVの列名ゆれ対応（例: 住所1 vs 住所１, 氏名 vs 患者氏名 など）
        CMP_ALIASES = {
            "患者氏名カナ": ["患者氏名カナ", "カナ氏名", "ﾌﾘｶﾞﾅ", "フリガナ"],
            "患者氏名": ["患者氏名", "氏名", "名前"],
            "性別": ["性別", "性", "男女区分"],
            "生年月日": ["生年月日", "誕生日", "出生年月日"],
            "郵便番号": ["郵便番号", "郵便No", "郵便番号１", "郵便番号1"],
            "住所１": ["住所１", "住所1", "住所", "住所_1"],
            "電話番号": ["電話番号", "電話", "TEL", "Tel", "電話番号１", "電話番号1"],
        }

        def _get_cmp(col: str) -> pd.Series:
            # 既定: 空欄列を返す
            empty = pd.Series([""] * len(cmp_df))
            # そのままヒット
            if col in cmp_df.columns:
                return cmp_df[col]
            # エイリアス探索
            for cand in CMP_ALIASES.get(col, []):
                if cand in cmp_df.columns:
                    return cmp_df[cand]
            # 一部の表示ゆれ（全角/半角数字）を総当たりで試行
            if col == "住所１" and "住所1" in cmp_df.columns:
                return cmp_df["住所1"]
            if col == "郵便番号" and "郵便番号1" in cmp_df.columns:
                return cmp_df["郵便番号1"]
            if col == "電話番号" and "電話番号1" in cmp_df.columns:
                return cmp_df["電話番号1"]
            return empty

        cmp_kana_norm = _get_cmp("患者氏名カナ").map(self._norm_kana_for_compare)
        cmp_name_norm = _get_cmp("患者氏名").map(self._norm_name_for_compare)
        # 氏名：空欄ならカナ氏名で補完（比較用）
        cmp_name_filled = cmp_name_norm.copy()
        try:
            cmp_name_filled = cmp_name_filled.mask(cmp_name_filled.str.strip() == "", other=cmp_kana_norm)
        except Exception:
            cmp_name_filled = cmp_name_filled.map(lambda v, k=cmp_kana_norm: v if str(v).strip() != "" else "").astype(str)
            cmp_name_filled = cmp_name_filled.where(cmp_name_filled != "", other=cmp_kana_norm)

        cmp_norm = pd.DataFrame({
            "患者番号": cmp_key,
            "患者氏名カナ": cmp_kana_norm,
            "患者氏名": cmp_name_filled,
            "性別": _get_cmp("性別").map(self._norm_gender_for_compare),
            "生年月日": _get_cmp("生年月日").map(self._norm_birth_for_compare),
            "郵便番号": _get_cmp("郵便番号").map(self._norm_zip_for_compare),
            "住所１": _get_cmp("住所１").map(self._norm_addr_for_compare),
            "電話番号": _get_cmp("電話番号").map(self._norm_tel_for_compare),
        })
        cmp_norm["年齢"] = cmp_norm["生年月日"].map(self._norm_age_from_birth)

        # --- 完全重複のみ除去（キー＋比較列がすべて同一の行だけ落とす） ---
        dup_subset = ["患者番号","患者氏名カナ","患者氏名","性別","生年月日","年齢","郵便番号","住所１","電話番号"]
        _src_before = len(src_norm)
        _cmp_before = len(cmp_norm)
        src_norm = src_norm.drop_duplicates(subset=dup_subset, keep="first")
        cmp_norm = cmp_norm.drop_duplicates(subset=dup_subset, keep="first")
        try:
            self.log(f"[患者-内容] 完全重複削除: 元 { _src_before }→{ len(src_norm) } / 突合 { _cmp_before }→{ len(cmp_norm) }")
        except Exception:
            pass

        # 7) キー突合 & 未ヒット
        src_norm = src_norm[src_norm["患者番号"] != ""].copy()
        cmp_norm = cmp_norm[cmp_norm["患者番号"] != ""].copy()

        cmp_key_set = set(cmp_norm["患者番号"].tolist())
        missing_mask = ~src_norm["患者番号"].isin(cmp_key_set)
        missing_df = src.loc[src_norm.index.intersection(src_norm[missing_mask].index)].copy()
        missing_df.insert(0, "__正規化患者番号__", src_norm.loc[missing_mask, "患者番号"])  # インデックス対応

        # 8) 一致/不一致判定
        merged = src_norm.merge(cmp_norm, on="患者番号", how="inner", suffixes=("_src", "_cmp"))
        fields = ["患者氏名カナ", "患者氏名", "性別", "生年月日", "年齢", "郵便番号", "住所１", "電話番号"]

        all_eq_mask = pd.Series([True] * len(merged), index=merged.index)
        for f in fields:
            all_eq_mask &= (merged[f + "_src"] == merged[f + "_cmp"])
        matched_rows = merged.loc[all_eq_mask, ["患者番号"] + [f + "_src" for f in fields]].copy()
        matched_rows.rename(columns={f + "_src": f for f in fields}, inplace=True)

        mismatches = []
        for f in fields:
            neq = merged.loc[merged[f + "_src"] != merged[f + "_cmp"], ["患者番号", f + "_src", f + "_cmp"]].copy()
            if not neq.empty:
                neq.insert(1, "項目名", f)
                neq.rename(columns={f + "_src": "正規化_元", f + "_cmp": "正規化_突合"}, inplace=True)
                mismatches.append(neq)
        mismatch_df = pd.concat(mismatches, axis=0) if mismatches else pd.DataFrame(columns=["患者番号", "項目名", "正規化_元", "正規化_突合"])

        # 9) 出力
        tag = _dt.now().strftime("%Y%m%d")
        out_matched = out_dir / f"患者_内容_一致_{tag}.csv"
        out_mismatch = out_dir / f"患者_内容_不一致_{tag}.csv"
        out_missing = out_dir / f"患者_内容_未ヒット_{tag}.csv"
        inspection.to_csv(matched_rows, str(out_matched))
        inspection.to_csv(mismatch_df, str(out_mismatch))
        inspection.to_csv(missing_df, str(out_missing))

        self.log(
            f"[患者-内容] 一致: {len(matched_rows)} / 不一致明細行: {len(mismatch_df)} / 未ヒット: {len(missing_df)}\n"
            f"  (src={len(src_norm)}, cmp={len(cmp_norm)})"
        )
        messagebox.showinfo(
            "患者内容検収 完了",
            f"一致: {len(matched_rows)} 件\n不一致明細: {len(mismatch_df)} 行（項目単位）\n未ヒット: {len(missing_df)} 件\n\n出力先:\n{out_dir}"
        )

# ===== Entry points for InspectionActions =====
def run_patient_content_check(app, logger=None, preset=None):
    checker = PatientContentChecker(logger=logger, preset_colmap=preset)
    return bool(checker.run(app))

def run(app, logger=None, preset=None):
    checker = PatientContentChecker(logger=logger, preset_colmap=preset)
    return bool(checker.run(app))

def main(app, logger=None, preset=None):
    checker = PatientContentChecker(logger=logger, preset_colmap=preset)
    return bool(checker.run(app))