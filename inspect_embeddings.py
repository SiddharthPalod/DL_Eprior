import pandas as pd
import pathlib
base = pathlib.Path('pdf_embeddings')
for name in ['d1.xlsx','d2.xlsx','d3.xlsx','d4.xlsx']:
    df = pd.read_excel(base / name)
    print(name)
    print(df.head().to_string())
