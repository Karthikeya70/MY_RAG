import sys
sys.stdout.reconfigure(encoding="utf-8")

from docling.document_converter import DocumentConverter

# Usage: python src/parse.py gold   OR   python src/parse.py silver
plan_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "gold"

pdf_path = f"data/raw/bajaj_{plan_arg}.pdf"
output_path = f"data/processed/bajaj_{plan_arg}_raw.md"

# Create the converter - this is Docling's main tool
# It reads the PDF and understands its layout
converter = DocumentConverter()

# Convert the PDF
print(f"Parsing {pdf_path} with Docling...")
result = converter.convert(pdf_path)

# Get the document
doc = result.document

# Export the whole document to clean markdown text
markdown_text = doc.export_to_markdown()

print("\n========== MARKDOWN EXPORT (first 3000 chars) ==========\n")
print(markdown_text[:3000])

# Print tables separately
print("\n========== TABLES FOUND ==========\n")
tables = doc.tables
print(f"Total tables found: {len(tables)}")
for i, table in enumerate(tables):
    print(f"\n--- TABLE {i+1} ---")
    print(table.export_to_dataframe(doc))

# Save the full markdown to disk for inspection in VS Code
with open(output_path, "w", encoding="utf-8") as f:
    f.write(markdown_text)
print(f"\nSaved full markdown export to {output_path}")
