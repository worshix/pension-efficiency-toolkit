import zipfile
import xml.etree.ElementTree as ET

def extract_docx_text(docx_path):
    with zipfile.ZipFile(docx_path, 'r') as z:
        with z.open('word/document.xml') as f:
            tree = ET.parse(f)
    root = tree.getroot()
    
    paragraphs = []
    for para in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        texts = []
        for run in para.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
            if run.text:
                texts.append(run.text)
        paragraphs.append(''.join(texts))
    return paragraphs

paras = extract_docx_text('PROJECT.docx')
for i, p in enumerate(paras):
    print(f'[{i}] {repr(p)}')
