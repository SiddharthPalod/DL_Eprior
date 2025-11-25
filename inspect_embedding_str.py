import pandas as pd
from pathlib import Path
base = Path('pdf_embeddings')
df = pd.read_excel(base / 'd1.xlsx')
val = df.loc[0, 'embedding_preview']
print(len(val))
print(val[:200])
