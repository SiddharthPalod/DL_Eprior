from pypdf import PdfReader
import pathlib
path = pathlib.Path('plan.pdf')
reader = PdfReader(str(path))
with open('plan.txt','w',encoding='utf-8') as f:
    for page in reader.pages:
        text = page.extract_text()
        if text:
            f.write(text)
            f.write('\n')
