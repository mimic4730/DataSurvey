from __future__ import annotations
from tkinter import filedialog, messagebox
from pathlib import Path
import pandas as pd
from datetime import datetime as _dt
from core.io_utils import CsvLoader
from core import inspection

# 共通ルール（患者の移行対象外判定）
try:
    from core.rules.patient import evaluate_patient_exclusions, PatientRuleConfig  # type: ignore
except Exception:
    evaluate_patient_exclusions = None  # type: ignore
    PatientRuleConfig = None  # type: ignore


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
        # まず NFKC で互換正規化（半角→全角、濁点/半濁点の結合系列の平準化など）
        t = unicodedata.normalize("NFKC", str(s))
        # つづけて NFC で正規合成（例: "ウ"+"゛" → "ヴ", "ハ"+"゜" → "パ"）
        t = unicodedata.normalize("NFC", t)
        # 結合濁点/半濁点の直前に紛れたスペースを除去（例: "カ ゙" → "ガ"）
        t = re.sub(r"\s+([\u3099\u309A])", r"\1", t)
        # 再合成
        t = unicodedata.normalize("NFC", t)
        # ハイフン/ダッシュ/長音記号のゆらぎを長音「ー」に統一（例: "-" "−" "ｰ" "—" など）
        t = re.sub(r"[-‐‑‒–—―ｰ−]", "ー", t)
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
        # まず NFKC、続けて NFC（濁点合成 "゛/゜" のゆらぎも吸収）
        t = unicodedata.normalize("NFKC", str(s))
        t = unicodedata.normalize("NFC", t)
        # 結合濁点/半濁点の直前に紛れたスペースを除去
        t = re.sub(r"\s+([\u3099\u309A])", r"\1", t)
        t = unicodedata.normalize("NFC", t)
        # ハイフン/ダッシュ類を長音「ー」に統一（氏名中の「-」「−」などを吸収）
        t = re.sub(r"[-‐‑‒–—―ｰ−]", "ー", t)
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

    def _looks_mojibake(self, s: pd.Series) -> pd.Series:
        """
        典型的な文字化けの兆候を検知して True を返すブール Series。
        ・置換用文字 '�' (U+FFFD)
        ・UTF-8/Shift_JIS 誤デコードで現れやすい文字群 (例: Ã, ã, Â, ¢, ð, Š, œ, ™, €)
        """
        if s is None or len(s) == 0:
            return pd.Series([False], index=getattr(s, "index", None)).iloc[:0]
        import re
        ss = s.astype(str)
        # 目視で頻出するパターンを素直に拾う（過検出を避けるため最小限）
        pattern = "[�ÃãÂ¢ðŠœ™€\uE000-\uF8FF?]"
        return ss.str.contains(pattern)

    @staticmethod
    def _pick_phone_with_mobile_fallback(df: pd.DataFrame, tel_col: str | None, mobile_col: str | None) -> pd.Series:
        """元CSV限定: 電話が空なら携帯で補完(=電話に格上げ)。
        空の定義:
          - NaN/None
          - 空文字/空白のみ
          - 数字以外しか含まない(記号のみ, 例: "--" や "()")
        携帯側も数字が1桁以上なければ無視。
        """
        import re
        n = len(df)
        # ベース(電話)
        if tel_col and tel_col in df.columns:
            tel_raw = df[tel_col]
        else:
            tel_raw = pd.Series([""] * n, index=df.index, dtype="object")

        # 携帯
        if mobile_col and mobile_col in df.columns:
            mob_raw = df[mobile_col]
        else:
            mob_raw = pd.Series([""] * n, index=df.index, dtype="object")

        # 文字列化
        tel_str = tel_raw.astype("string").fillna("")
        mob_str = mob_raw.astype("string").fillna("")

        # 空欄判定: NaN/空白のみ/数字が1桁もない
        def _is_blank(s: pd.Series) -> pd.Series:
            s2 = s.fillna("").astype(str).str.strip()
            digits = s2.map(lambda x: re.sub(r"[^0-9]", "", x))
            return (s2 == "") | (digits.str.len() == 0) | (s2.str.lower().isin(["nan", "none"]))

        tel_blank = _is_blank(tel_str)
        mob_has_digits = _is_blank(mob_str) == False  # 携帯が有効(数字あり)

        use_mobile = tel_blank & mob_has_digits
        tel_upgraded = tel_str.mask(use_mobile, other=mob_str)
        return tel_upgraded.astype("string").fillna("")

    def _phone_upgrade_pair(self, df: pd.DataFrame, tel_col: str | None, mobile_col: str | None):
        """電話の「格上げ」前後を同時に作るユーティリティ。
        戻り値: (tel_base, tel_conv, used_mask)
          - tel_base : 生の電話列（空欄は空欄のまま、携帯補完なし）
          - tel_conv : 電話が空欄なら携帯で補完した列
          - used_mask: 補完を適用した行(True)
        空欄定義は _pick_phone_with_mobile_fallback と同じ。
        """
        import re
        n = len(df)
        # Base(電話)
        if tel_col and tel_col in df.columns:
            tel_raw = df[tel_col]
        else:
            tel_raw = pd.Series([""] * n, index=df.index, dtype="object")
        # Mobile
        if mobile_col and mobile_col in df.columns:
            mob_raw = df[mobile_col]
        else:
            mob_raw = pd.Series([""] * n, index=df.index, dtype="object")

        tel_str = tel_raw.astype("string").fillna("")
        mob_str = mob_raw.astype("string").fillna("")

        def _is_blank(s: pd.Series) -> pd.Series:
            s2 = s.fillna("").astype(str).str.strip()
            digits = s2.map(lambda x: re.sub(r"[^0-9]", "", x))
            return (s2 == "") | (digits.str.len() == 0) | (s2.str.lower().isin(["nan", "none"]))

        tel_blank = _is_blank(tel_str)
        mob_has_digits = _is_blank(mob_str) == False

        used_mask = tel_blank & mob_has_digits
        tel_conv = tel_str.mask(used_mask, other=mob_str)
        tel_base = tel_str
        return tel_base.astype("string").fillna(""), tel_conv.astype("string").fillna(""), used_mask

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

        # 2.5) 移行対象外ルールの事前適用（内容検収は「対象外を除いた母集団」で比較）
        tag = _dt.now().strftime("%Y%m%d")  # 出力ファイル名で再利用
        excluded_count = 0
        try:
            if evaluate_patient_exclusions is not None:
                cfg = PatientRuleConfig() if PatientRuleConfig is not None else None  # 既定設定
                remains_df, excluded_df = evaluate_patient_exclusions(src, colmap, cfg)  # type: ignore
                if isinstance(excluded_df, pd.DataFrame) and not excluded_df.empty:
                    excluded_count = len(excluded_df)
                    out_excluded = out_dir / f"患者_内容_対象外_{tag}.csv"
                    inspection.to_csv(excluded_df, str(out_excluded))
                    self.log(f"[患者-内容] 対象外: {excluded_count} 件 → {out_excluded}")
                if isinstance(remains_df, pd.DataFrame) and not remains_df.empty:
                    src = remains_df  # 以降の一致・未ヒット・不一致判定は対象外を除いた集合で実施
                else:
                    # 全件が対象外なら空集合で続行（後続で未ヒット/一致は0件）
                    src = remains_df
            else:
                self.log("[患者-内容] 対象外ルール(evaluate_patient_exclusions)が見つからないため、全件を比較対象とします")
        except Exception as _e:
            self.log(f"[患者-内容] 対象外ルール適用で例外（スキップして続行）: {_e}")

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
        # 元CSVの氏名（RAW と 正規化の両方を持つ）
        src_name_raw = src[colmap.get("患者氏名")] if colmap.get("患者氏名") in src.columns else pd.Series([""] * len(src))
        src_name_norm = src_name_raw.map(self._norm_name_for_compare)

        # 文字化け検出 or 空欄は、カナ氏名で補完（移行と同じ運用を反映）
        try:
            mojibake_mask = self._looks_mojibake(src_name_raw)
        except Exception:
            # 失敗時は安全に False
            mojibake_mask = pd.Series([False] * len(src_name_norm), index=src_name_norm.index)

        empty_mask = src_name_norm.str.strip().eq("")
        replace_mask = (mojibake_mask | empty_mask)

        src_name_filled = src_name_norm.mask(replace_mask, other=src_kana)

        # ログ: 何件をカナ氏名で補完したか
        try:
            repl_cnt = int(replace_mask.sum())
            if repl_cnt > 0:
                self.log(f"[患者-内容] 氏名の補完(文字化け/空欄→カナ): {repl_cnt} 件")
        except Exception:
            pass
        src_sex = src[colmap.get("性別")] if colmap.get("性別") in src.columns else pd.Series([""] * len(src))
        src_sex = src_sex.map(self._norm_gender_for_compare)
        src_birth = src[colmap.get("生年月日")] if colmap.get("生年月日") in src.columns else pd.Series([""] * len(src))
        src_birth = src_birth.map(self._norm_birth_for_compare)
        src_age = src_birth.map(self._norm_age_from_birth)
        src_zip = src[colmap.get("郵便番号")] if colmap.get("郵便番号") in src.columns else pd.Series([""] * len(src))
        src_zip = src_zip.map(self._norm_zip_for_compare)
        src_addr = src[colmap.get("住所１")] if colmap.get("住所１") in src.columns else pd.Series([""] * len(src))
        src_addr = src_addr.map(self._norm_addr_for_compare)
        # 電話番号: 補正なし/ありの2系統を用意（未補正は不一致判定用、補正ありは変換一致用）
        tel_base_str, tel_conv_str, phone_used_mask = self._phone_upgrade_pair(
            src, colmap.get("電話番号"), colmap.get("携帯電話番号")
        )
        src_tel_base = tel_base_str.map(self._norm_tel_for_compare)
        src_tel_conv  = tel_conv_str.map(self._norm_tel_for_compare)
        # 補正が適用された患者番号キー集合（後で「適用ルール」列の判定に使用）
        try:
            phone_replaced_keys = set(src_key[phone_used_mask].tolist())
        except Exception:
            phone_replaced_keys = set()

        # 2系統: src_norm_base(氏名=正規化), src_norm_conv(氏名=カナ補完)
        src_norm_base = pd.DataFrame({
            "患者番号": src_key,
            "患者氏名カナ": src_kana,
            "患者氏名": src_name_norm,
            "性別": src_sex,
            "生年月日": src_birth,
            "年齢": src_age,
            "郵便番号": src_zip,
            "住所１": src_addr,
            "電話番号": src_tel_base,
        })
        src_norm_conv = pd.DataFrame({
            "患者番号": src_key,
            "患者氏名カナ": src_kana,
            "患者氏名": src_name_filled,
            "性別": src_sex,
            "生年月日": src_birth,
            "年齢": src_age,
            "郵便番号": src_zip,
            "住所１": src_addr,
            "電話番号": src_tel_conv,
        })

        try:
            _ph_cnt = len(phone_replaced_keys)
            if _ph_cnt > 0:
                self.log(f"[患者-内容] 電話の補完(空欄→携帯へ格上げ): {_ph_cnt} 件")
        except Exception:
            pass

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
        _src_before = len(src_norm_base)
        _cmp_before = len(cmp_norm)
        src_norm_base = src_norm_base.drop_duplicates(subset=dup_subset, keep="first")
        src_norm_conv = src_norm_conv.drop_duplicates(subset=dup_subset, keep="first")
        cmp_norm = cmp_norm.drop_duplicates(subset=dup_subset, keep="first")
        try:
            self.log(f"[患者-内容] 完全重複削除: 元 { _src_before }→{ len(src_norm_base) } / 突合 { _cmp_before }→{ len(cmp_norm) }")
        except Exception:
            pass

        # 7) キー突合 & 未ヒット（src_norm_base基準）
        src_norm_base = src_norm_base[src_norm_base["患者番号"] != ""].copy()
        src_norm_conv = src_norm_conv[src_norm_conv["患者番号"] != ""].copy()
        cmp_norm = cmp_norm[cmp_norm["患者番号"] != ""].copy()

        # 未ヒット: src_norm_base側で突合キーに無いもの
        src_keys_all = set(src_norm_base.loc[src_norm_base["患者番号"] != "", "患者番号"].tolist())
        cmp_key_set = set(cmp_norm.loc[cmp_norm["患者番号"] != "", "患者番号"].tolist())
        missing_keys = src_keys_all - cmp_key_set
        missing_index = src_norm_base.index[src_norm_base["患者番号"].isin(missing_keys)]
        missing_df = src.loc[missing_index].copy()
        missing_df.insert(0, "__正規化患者番号__", src_norm_base.loc[missing_index, "患者番号"])

        # 比較対象列
        fields = ["患者氏名カナ", "患者氏名", "性別", "生年月日", "年齢", "郵便番号", "住所１", "電話番号"]

        # 8) Pass-1 (厳密一致): src_norm_base vs cmp_norm
        merged1 = src_norm_base.merge(cmp_norm, on="患者番号", how="inner", suffixes=("_src", "_cmp"))
        all_eq_mask1 = pd.Series([True] * len(merged1), index=merged1.index)
        for f in fields:
            all_eq_mask1 &= (merged1[f + "_src"] == merged1[f + "_cmp"])
        matched_rows_strict = merged1.loc[all_eq_mask1, ["患者番号"] + [f + "_src" for f in fields]].copy()
        matched_rows_strict.rename(columns={f + "_src": f for f in fields}, inplace=True)

        # 8) Pass-2 (SRCのみ補正: 氏名→カナ補完): Pass-1で一致していない患者番号
        remain_keys = set(src_norm_base["患者番号"]) - set(matched_rows_strict["患者番号"])
        src_conv_slice = src_norm_conv[src_norm_conv["患者番号"].isin(remain_keys)]
        merged2 = src_conv_slice.merge(cmp_norm, on="患者番号", how="inner", suffixes=("_src", "_cmp"))
        all_eq_mask2 = pd.Series([True] * len(merged2), index=merged2.index)
        for f in fields:
            all_eq_mask2 &= (merged2[f + "_src"] == merged2[f + "_cmp"])
        matched_rows_conv = merged2.loc[all_eq_mask2, ["患者番号"] + [f + "_src" for f in fields]].copy()
        matched_rows_conv.rename(columns={f + "_src": f for f in fields}, inplace=True)
        # Pass-2: 「適用ルール」列追加
        name_replaced_keys = set(src_key[replace_mask].tolist())
        def _rule_for_key(k):
            flags = []
            if k in name_replaced_keys:
                flags.append("氏名→カナ補完")
            if k in phone_replaced_keys:
                flags.append("電話→携帯補完")
            return " + ".join(flags) if flags else "SRC補正"
        matched_rows_conv["適用ルール"] = matched_rows_conv["患者番号"].map(_rule_for_key)

        # 不一致明細: Pass-1（未補正=ベース）の差分をすべて列挙する
        # ここで作る不一致は「SRC未補正なら不一致」= 監査視点での生の差分
        base_diff_rows = merged1.loc[~all_eq_mask1].copy()

        mismatches = []
        for f in fields:
            neq = base_diff_rows.loc[
                base_diff_rows[f + "_src"] != base_diff_rows[f + "_cmp"],
                ["患者番号", f + "_src", f + "_cmp"]
            ].copy()
            if not neq.empty:
                neq.insert(1, "項目名", f)
                neq.rename(columns={f + "_src": "正規化_元", f + "_cmp": "正規化_突合"}, inplace=True)
                mismatches.append(neq)

        mismatch_df = (
            pd.concat(mismatches, axis=0)
            if mismatches else pd.DataFrame(columns=["患者番号", "項目名", "正規化_元", "正規化_突合"])
        )

        # 変換で解消されたかのフラグ（Pass-2 で一致に回った患者番号）
        try:
            resolved_keys = set(matched_rows_conv["患者番号"].tolist())
            if not mismatch_df.empty:
                mismatch_df["変換で解消"] = mismatch_df["患者番号"].map(lambda k: "はい" if k in resolved_keys else "いいえ")
        except Exception:
            pass

        try:
            mismatch_unique = mismatch_df["患者番号"].nunique()
            self.log(f"[患者-内容] 不一致レコード数(患者番号ユニーク): {mismatch_unique}")
        except Exception:
            pass

        # 9) 出力
        tag = _dt.now().strftime("%Y%m%d")
        out_matched = out_dir / f"患者_内容_一致_{tag}.csv"
        out_conv_matched = out_dir / f"患者_内容_変換一致_{tag}.csv"
        out_mismatch = out_dir / f"患者_内容_不一致_{tag}.csv"
        out_missing = out_dir / f"患者_内容_未ヒット_{tag}.csv"
        inspection.to_csv(matched_rows_strict, str(out_matched))
        inspection.to_csv(matched_rows_conv, str(out_conv_matched))
        inspection.to_csv(mismatch_df, str(out_mismatch))
        inspection.to_csv(missing_df, str(out_missing))

        self.log(
            f"[患者-内容] 一致: {len(matched_rows_strict)} / 変換一致(SRC補正): {len(matched_rows_conv)} / 不一致明細行: {len(mismatch_df)} / 未ヒット: {len(missing_df)} / 対象外: {excluded_count}\n"
            f"  (src={len(src_norm_base)}, cmp={len(cmp_norm)})"
        )
        messagebox.showinfo(
            "患者内容検収 完了",
            f"一致: {len(matched_rows_strict)} 件\n"
            f"変換一致（SRC補正）: {len(matched_rows_conv)} 件\n"
            f"不一致明細: {len(mismatch_df)} 行（項目単位）\n"
            f"未ヒット: {len(missing_df)} 件\n"
            f"対象外: {excluded_count} 件（別CSVへ出力済み）\n\n"
            f"出力先:\n{out_dir}"
        )

def run_integrated(*, src_df: pd.DataFrame, colmap: dict, cmp_path: str, out_dir, logger=None) -> dict:
    """
    検収CSV生成フローから呼び出す、ダイアログなしの『患者・内容検収』実行関数。
    - src_df, colmap, cmp_path, out_dir をそのまま使用（UIダイアログなし）
    - 出力は out_dir 直下に
        患者_内容_一致_YYYYMMDD.csv
        患者_内容_変換一致_YYYYMMDD.csv
        患者_内容_不一致_YYYYMMDD.csv
        患者_内容_未ヒット_YYYYMMDD.csv
        患者_内容_対象外_YYYYMMDD.csv
      を保存
    """
    checker = PatientContentChecker(logger=logger, preset_colmap=None)
    def log(msg: str):
        checker.log(msg)
    tag = _dt.now().strftime("%Y%m%d")

    # 0) 入力
    src = src_df.copy()
    cmp_df = CsvLoader.read_csv_flex(cmp_path)

    # 1) 対象外（患者ルール）
    excluded_count = 0
    try:
        if evaluate_patient_exclusions is not None:
            cfg = PatientRuleConfig() if PatientRuleConfig is not None else None
            remains_df, excluded_df = evaluate_patient_exclusions(src, colmap, cfg)  # type: ignore
            if isinstance(excluded_df, pd.DataFrame) and not excluded_df.empty:
                excluded_count = len(excluded_df)
                out_ex = Path(out_dir) / f"患者_内容_対象外_{tag}.csv"
                inspection.to_csv(excluded_df, str(out_ex))
                log(f"[患者-内容] 対象外: {excluded_count} 件 → {out_ex}")
            src = remains_df if isinstance(remains_df, pd.DataFrame) else src.head(0)
        else:
            log("[患者-内容] 対象外ルールが見つからないため、全件比較します")
    except Exception as _e:
        log(f"[患者-内容] 対象外ルール適用で例外（スキップ）: {_e}")

    # 2) キー幅推定
    import re, unicodedata
    def _digits_len_max(s: pd.Series) -> int:
        return int(s.astype(str).map(lambda x: re.sub(r"[^0-9]", "", unicodedata.normalize("NFKC", x))).str.len().max() or 0)
    src_w = _digits_len_max(src[colmap.get("患者番号")]) if colmap.get("患者番号") in src.columns else 0
    cmp_w = _digits_len_max(cmp_df["患者番号"]) if "患者番号" in cmp_df.columns else 0
    width = max(src_w, cmp_w, 1)
    log(f"[患者-内容] 患者番号桁数: 元={src_w} / 突合={cmp_w} / 使用幅={width}")

    # 3) 元CSV 正規化（run と同一）
    norm = checker._normalize_codes
    src_key = norm(src[colmap.get("患者番号")].astype(str), width, mode="zfill") if colmap.get("患者番号") in src.columns else pd.Series([""]*len(src))
    src_kana = (src[colmap.get("患者氏名カナ")].map(checker._norm_kana_for_compare)
                if colmap.get("患者氏名カナ") in src.columns else pd.Series([""]*len(src)))
    src_name_raw = (src[colmap.get("患者氏名")] if colmap.get("患者氏名") in src.columns else pd.Series([""]*len(src)))
    src_name_norm = src_name_raw.map(checker._norm_name_for_compare)
    # 氏名：文字化け/空欄 → カナ補完
    try:
        mojibake_mask = checker._looks_mojibake(src_name_raw)
    except Exception:
        mojibake_mask = pd.Series([False]*len(src_name_norm), index=src_name_norm.index)
    empty_mask = src_name_norm.str.strip().eq("")
    replace_mask = (mojibake_mask | empty_mask)
    src_name_filled = src_name_norm.mask(replace_mask, other=src_kana)
    # 性別/生年月日/年齢/郵便/住所/電話（電話は携帯格上げの2系統）
    src_sex   = (src[colmap.get("性別")].map(checker._norm_gender_for_compare)
                 if colmap.get("性別") in src.columns else pd.Series([""]*len(src)))
    src_birth = (src[colmap.get("生年月日")].map(checker._norm_birth_for_compare)
                 if colmap.get("生年月日") in src.columns else pd.Series([""]*len(src)))
    src_age   = src_birth.map(checker._norm_age_from_birth)
    src_zip   = (src[colmap.get("郵便番号")].map(checker._norm_zip_for_compare)
                 if colmap.get("郵便番号") in src.columns else pd.Series([""]*len(src)))
    # 住所は「住所1」「住所１」どちらでも拾う
    _addr_key = "住所１" if "住所１" in src.columns else ("住所1" if "住所1" in src.columns else None)
    src_addr  = (src[_addr_key].map(checker._norm_addr_for_compare) if _addr_key else pd.Series([""]*len(src)))
    tel_base_str, tel_conv_str, phone_used_mask = checker._phone_upgrade_pair(
        src, colmap.get("電話番号"), colmap.get("携帯電話番号")
    )
    src_tel_base = tel_base_str.map(checker._norm_tel_for_compare)
    src_tel_conv = tel_conv_str.map(checker._norm_tel_for_compare)
    try:
        phone_replaced_keys = set(src_key[phone_used_mask].tolist())
    except Exception:
        phone_replaced_keys = set()
    name_replaced_keys = set(src_key[replace_mask].tolist())
    def _rule_for_key(k):
        flags = []
        if k in name_replaced_keys:  flags.append("氏名→カナ補完")
        if k in phone_replaced_keys: flags.append("電話→携帯補完")
        return " + ".join(flags) if flags else "SRC補正"
    # 2系統（ベース/補正）
    fields = ["患者氏名カナ","患者氏名","性別","生年月日","年齢","郵便番号","住所１","電話番号"]
    src_norm_base = pd.DataFrame({
        "患者番号": src_key, "患者氏名カナ": src_kana, "患者氏名": src_name_norm, "性別": src_sex,
        "生年月日": src_birth, "年齢": src_age, "郵便番号": src_zip, "住所１": src_addr, "電話番号": src_tel_base
    })
    src_norm_conv = pd.DataFrame({
        "患者番号": src_key, "患者氏名カナ": src_kana, "患者氏名": src_name_filled, "性別": src_sex,
        "生年月日": src_birth, "年齢": src_age, "郵便番号": src_zip, "住所１": src_addr, "電話番号": src_tel_conv
    })
    # 4) 突合側 正規化（run と同一）
    if "患者番号" not in cmp_df.columns:
        return {"error": "突合CSVに『患者番号』列がありません。"}
    cmp_key = checker._normalize_codes(cmp_df["患者番号"].astype(str), width, mode="zfill")
    CMP_ALIASES = {
        "患者氏名カナ": ["患者氏名カナ","カナ氏名","ﾌﾘｶﾞﾅ","フリガナ"],
        "患者氏名": ["患者氏名","氏名","名前"],
        "性別": ["性別","性","男女区分"],
        "生年月日": ["生年月日","誕生日","出生年月日"],
        "郵便番号": ["郵便番号","郵便No","郵便番号１","郵便番号1"],
        "住所１": ["住所１","住所1","住所","住所_1"],
        "電話番号": ["電話番号","電話","TEL","Tel","電話番号１","電話番号1"],
    }
    def _get_cmp(col: str) -> pd.Series:
        if col in cmp_df.columns: return cmp_df[col]
        for cand in CMP_ALIASES.get(col, []):
            if cand in cmp_df.columns: return cmp_df[cand]
        if col == "住所１" and "住所1" in cmp_df.columns: return cmp_df["住所1"]
        if col == "郵便番号" and "郵便番号1" in cmp_df.columns: return cmp_df["郵便番号1"]
        if col == "電話番号" and "電話番号1" in cmp_df.columns: return cmp_df["電話番号1"]
        return pd.Series([""]*len(cmp_df))
    cmp_kana_norm = _get_cmp("患者氏名カナ").map(checker._norm_kana_for_compare)
    cmp_name_norm = _get_cmp("患者氏名").map(checker._norm_name_for_compare)
    cmp_name_filled = cmp_name_norm.mask(cmp_name_norm.str.strip() == "", other=cmp_kana_norm)
    cmp_norm = pd.DataFrame({
        "患者番号": cmp_key,
        "患者氏名カナ": cmp_kana_norm,
        "患者氏名": cmp_name_filled,
        "性別": _get_cmp("性別").map(checker._norm_gender_for_compare),
        "生年月日": _get_cmp("生年月日").map(checker._norm_birth_for_compare),
        "郵便番号": _get_cmp("郵便番号").map(checker._norm_zip_for_compare),
        "住所１": _get_cmp("住所１").map(checker._norm_addr_for_compare),
        "電話番号": _get_cmp("電話番号").map(checker._norm_tel_for_compare),
    })
    cmp_norm["年齢"] = cmp_norm["生年月日"].map(checker._norm_age_from_birth)

    # 5) 完全重複の削除
    dup_subset = ["患者番号","患者氏名カナ","患者氏名","性別","生年月日","年齢","郵便番号","住所１","電話番号"]
    src_norm_base = src_norm_base.drop_duplicates(subset=dup_subset, keep="first")
    src_norm_conv = src_norm_conv.drop_duplicates(subset=dup_subset, keep="first")
    cmp_norm      = cmp_norm.drop_duplicates(subset=dup_subset, keep="first")

    # 6) 未ヒット（キー：患者番号）
    src_norm_base = src_norm_base[src_norm_base["患者番号"] != ""].copy()
    src_norm_conv = src_norm_conv[src_norm_conv["患者番号"] != ""].copy()
    cmp_norm      = cmp_norm[cmp_norm["患者番号"]      != ""].copy()
    src_keys_all  = set(src_norm_base["患者番号"].tolist())
    cmp_key_set   = set(cmp_norm["患者番号"].tolist())
    missing_keys  = src_keys_all - cmp_key_set
    missing_index = src_norm_base.index[src_norm_base["患者番号"].isin(missing_keys)]
    missing_df    = src.loc[missing_index].copy()
    missing_df.insert(0, "__正規化患者番号__", src_norm_base.loc[missing_index, "患者番号"])

    # 7) Pass-1 厳密一致 / Pass-2（SRC補正）
    merged1 = src_norm_base.merge(cmp_norm, on="患者番号", how="inner", suffixes=("_src","_cmp"))
    all_eq_mask1 = pd.Series([True]*len(merged1), index=merged1.index)
    for f in fields: all_eq_mask1 &= (merged1[f+"_src"] == merged1[f+"_cmp"])
    matched_rows_strict = merged1.loc[all_eq_mask1, ["患者番号"]+[f+"_src" for f in fields]].copy()
    matched_rows_strict.rename(columns={f+"_src": f for f in fields}, inplace=True)

    remain_keys = set(src_norm_base["患者番号"]) - set(matched_rows_strict["患者番号"])
    src_conv_slice = src_norm_conv[src_norm_conv["患者番号"].isin(remain_keys)]
    merged2 = src_conv_slice.merge(cmp_norm, on="患者番号", how="inner", suffixes=("_src","_cmp"))
    all_eq_mask2 = pd.Series([True]*len(merged2), index=merged2.index)
    for f in fields: all_eq_mask2 &= (merged2[f+"_src"] == merged2[f+"_cmp"])
    matched_rows_conv = merged2.loc[all_eq_mask2, ["患者番号"]+[f+"_src" for f in fields]].copy()
    matched_rows_conv.rename(columns={f+"_src": f for f in fields}, inplace=True)
    matched_rows_conv["適用ルール"] = matched_rows_conv["患者番号"].map(_rule_for_key)

    # 8) 不一致明細（Pass-1 基準の生差分）
    base_diff_rows = merged1.loc[~all_eq_mask1].copy()
    mismatches = []
    for f in fields:
        neq = base_diff_rows.loc[base_diff_rows[f+"_src"] != base_diff_rows[f+"_cmp"], ["患者番号", f+"_src", f+"_cmp"]].copy()
        if not neq.empty:
            neq.insert(1, "項目名", f)
            neq.rename(columns={f+"_src": "正規化_元", f+"_cmp": "正規化_突合"}, inplace=True)
            mismatches.append(neq)
    mismatch_df = pd.concat(mismatches, axis=0) if mismatches else pd.DataFrame(columns=["患者番号","項目名","正規化_元","正規化_突合"])
    try:
        resolved_keys = set(matched_rows_conv["患者番号"].tolist())
        if not mismatch_df.empty:
            mismatch_df["変換で解消"] = mismatch_df["患者番号"].map(lambda k: "はい" if k in resolved_keys else "いいえ")
    except Exception:
        pass

    # 9) 出力
    out_dir = Path(out_dir)
    out_matched  = out_dir / f"患者_内容_一致_{tag}.csv"
    out_conv     = out_dir / f"患者_内容_変換一致_{tag}.csv"
    out_mismatch = out_dir / f"患者_内容_不一致_{tag}.csv"
    out_missing  = out_dir / f"患者_内容_未ヒット_{tag}.csv"
    inspection.to_csv(matched_rows_strict, str(out_matched))
    inspection.to_csv(matched_rows_conv,  str(out_conv))
    inspection.to_csv(mismatch_df,        str(out_mismatch))
    inspection.to_csv(missing_df,         str(out_missing))
    log(f"[患者-内容] 一致: {len(matched_rows_strict)} / 変換一致: {len(matched_rows_conv)} / 不一致明細行: {len(mismatch_df)} / 未ヒット: {len(missing_df)} / 対象外: {excluded_count}")

    return {
        "matched_path": str(out_matched),
        "conv_matched_path": str(out_conv),
        "mismatch_path": str(out_mismatch),
        "missing_path": str(out_missing),
        "excluded_path": str(out_dir / f"患者_内容_対象外_{tag}.csv"),
        "matched_count": int(len(matched_rows_strict)),
        "conv_matched_count": int(len(matched_rows_conv)),
        "mismatch_count": int(len(mismatch_df)),
        "missing_count": int(len(missing_df)),
        "excluded_count": int(excluded_count),
    }

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