# core/file_keys.py

# 文字コード候補
ENCODING_CANDIDATES = [
    "cp932", "shift_jis", "utf-8-sig", "utf-8", "euc_jp",
    "latin1", "iso-8859-1", "mac_roman"
]

# 患者コード候補の列名
PATIENT_CODE_CANDIDATES = [
    "患者コード", "患者番号", "患者ID", "患者Id", "患者ｺｰﾄﾞ",
    "PatientCode", "Patient_ID", "patient_id", "ID", "コード"
]

# ファイル識別（ファイル名にこのキーが含まれている想定）
FILE_KEYS = {
    "patients": "patients",
    "health_ins": "patient_health_insurances",
    "ceiling": "patient_ceiling_amount_applications",
    "subsidies": "patient_public_subsidies",
}

# サンプリング件数
SAMPLE_SIZE = 5

# 特殊列
COL_KUBUN = "区分"
# ※ユーザーコード準拠（『地方公費制度UUID』）
COL_LOCAL_PUB_UUID = "地方公費制度UUID"