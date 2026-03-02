import pandas as pd
from pathlib import Path

EXPECTED_COLS = {"LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"}

def load_raw_credit_xls(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)

    # Probamos combinaciones típicas
    for sheet in xls.sheet_names:
        for header in [0, 1, 2]:
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=header)
            except Exception:
                continue

            df.columns = [str(c).strip() for c in df.columns]

            if EXPECTED_COLS.issubset(set(df.columns)):
                return df

    raise ValueError(
        "No pude encontrar un header/hoja que contenga columnas esperadas "
        f"{sorted(EXPECTED_COLS)}. Revisá el archivo Excel."
    )