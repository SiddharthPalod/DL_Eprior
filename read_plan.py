from pypdf import PdfReader
import pathlib
path = pathlib.Path('plan.pdf')
reader = PdfReader(str(path))
text = ''
for page in reader.pages:
    text += page.extract_text() + '\n'
print(text[:2000])
