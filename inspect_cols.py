import pandas as pd
from pathlib import Path
base = Path('pdf_embeddings')
for name in sorted(base.glob('*.xlsx')):
    df = pd.read_excel(name)
    print(name.name, df.columns.tolist())
    print(df.dtypes)
    break
