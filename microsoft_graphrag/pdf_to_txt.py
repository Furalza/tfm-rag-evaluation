from pathlib import Path
from pypdf import PdfReader

input_dir = Path("input")

for pdf_path in input_dir.glob("*.pdf"):
    reader = PdfReader(str(pdf_path))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    txt_path = input_dir / f"{pdf_path.stem}.txt"
    txt_path.write_text(text, encoding="utf-8")

    print(f"Converted: {pdf_path.name} -> {txt_path.name}")