import re
import json
import sys

# Usage: python src/chunk.py gold  OR  python src/chunk.py silver
plan_arg = sys.argv[1].lower() if len(sys.argv) > 1 else "gold"

plan_name = "Gold" if plan_arg == "gold" else "Silver"
input_file = f"data/processed/bajaj_{plan_arg}_raw.md"
output_file = f"data/processed/bajaj_{plan_arg}_chunks.json"
source_file = f"bajaj_{plan_arg}.pdf"
chunk_prefix = plan_arg
# Gold's UIN is printed inside the gold PDF itself (verified).
# The silver PDF contains no UIN anywhere, so we leave it as None
# instead of guessing — wrong metadata is worse than missing metadata.
uin = "BAJHLIP21185V032021" if plan_arg == "gold" else None

# We measure chunk size in TOKENS — the same units the embedding model uses.
# Counting words instead is misleading: tables tokenize terribly (every '|',
# number and email becomes several tokens), so a 300-word table can secretly
# be 1000+ tokens and the model would silently ignore everything past 512.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

# Tuning knobs:
# - a section bigger than MAX_SECTION_TOKENS gets split into smaller pieces
# - each piece aims for about TARGET_TOKENS (must stay under the model's 512)
# - a chunk smaller than MIN_CHUNK_CHARS (in characters) gets merged into its neighbor
MAX_SECTION_TOKENS = 450
TARGET_TOKENS = 320
MIN_CHUNK_CHARS = 250

# Read the markdown file
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Normalize line endings
content = content.replace('\r\n', '\n').replace('\r', '\n')

# ─── Step 0: Keep first occurrence of repeating headers, remove the rest ─────
def remove_after_first(text, pattern):
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE | re.IGNORECASE))
    if len(matches) <= 1:
        return text
    result = text
    for match in reversed(matches[1:]):
        result = result[:match.start()] + result[match.end():]
    return result

content = remove_after_first(content, r'^##\s*Health Guard\s*[-–]\s*Gold Plan\s*\n')
content = remove_after_first(content, r'^##\s*Health Guard\s*[-–]\s*Silver Plan\s*\n')
content = remove_after_first(content, r'^##\s*Bajaj Allianz General Insurance Company Limited\s*\n')

# Clean up excessive blank lines
content = re.sub(r'\n{3,}', '\n\n', content)

# ─── Step 1: Extract page markers ────────────────────────────────────────────
page_markers = []
for match in re.finditer(r'(\d+)\s*\|\s*Page', content):
    page_markers.append({
        "page_number": int(match.group(1)),
        "position": match.start()
    })
print(f"Found {len(page_markers)} page markers")

# ─── Step 2: Clean text ──────────────────────────────────────────────────────
def compact_table_row(line):
    """Docling pads table cells with spaces so columns line up on screen,
    e.g. '| AHMEDABAD ...          |  Gujarat  |'. That padding is useless
    to us and wastes embedding tokens, so we squeeze it out."""
    if not line.strip().startswith('|'):
        return line
    # Separator rows like |-----------|------| become |---|---|
    if re.match(r'^\s*\|[\s\-:|]+\|\s*$', line):
        num_cols = line.count('|') - 1
        return '|' + '---|' * num_cols
    # Normal rows: collapse runs of spaces into one space
    line = re.sub(r' {2,}', ' ', line).strip()
    # Docling sometimes duplicates a whole cell in the same row
    # ('| office | Karnataka. | Karnataka. |'). Drop the copy — but only
    # for LONG cells, because short ones ('| 5 | 5 |') can be real data.
    cells = [c.strip() for c in line.strip('|').split('|')]
    deduped = []
    for c in cells:
        if deduped and c and c == deduped[-1] and len(c) > 30:
            continue
        deduped.append(c)
    return '| ' + ' | '.join(deduped) + ' |'

def clean_text(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if re.match(r'^\d+\s*\|\s*Page', line.strip()):
            continue
        if '<!-- image -->' in line:
            continue
        if 'CIN:' in line and 'U66010PN2000PLC015329' in line:
            continue
        cleaned.append(compact_table_row(line))
    return '\n'.join(cleaned)

# ─── Step 3: Page number logic ───────────────────────────────────────────────
def get_page_for_position(position, page_markers):
    for marker in page_markers:
        if marker["position"] >= position:
            return marker["page_number"]
    return page_markers[-1]["page_number"]

# ─── Step 4: Split oversized sections without breaking tables ────────────────
def is_table_row(line):
    return line.strip().startswith('|')

def is_separator_row(line):
    # A markdown table separator looks like |----|----|
    return bool(re.match(r'^\s*\|[\s\-:|]+\|\s*$', line))

def starts_new_record(line):
    """Some tables spread ONE logical entry over SEVERAL physical rows.
    Example (ombudsman offices): 'AHMEDABAD - ...' row, then address row,
    then phone row, then email row — all one office. Cutting between them
    tears an office's contact info across two chunks.

    New entries in such tables start with an ALL-CAPS name and a dash,
    e.g. '| JAIPUR - Smt. Sandhya Baliga |'. We prefer to cut right
    before such a row, so each piece holds complete offices."""
    if not is_table_row(line):
        return False
    first_cell = line.strip().strip('|').split('|')[0].strip()
    return bool(re.match(r'^[A-Z]{3,}[A-Z\s]*\s*[-–]', first_cell))

# The absolute ceiling for one piece. The embedding model reads 512 tokens;
# we stay below that to leave room for the repeated table header and the
# model's own special start/end tokens.
HARD_CAP_TOKENS = 460

def build_units(lines):
    """Group the lines of a section into 'units' — the smallest blocks
    we are never allowed to cut apart.

    - a normal text line  -> one unit by itself
    - one table RECORD    -> one unit (all its physical rows together,
      e.g. an ombudsman office's name + address + phone + email rows)

    Each unit remembers its table header, so if a chunk starts mid-table
    we can repeat the header at the top of that chunk.
    """
    units = []  # each unit: {"lines": [...], "header": [...] or None}
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not is_table_row(line):
            units.append({"lines": [line], "header": None})
            i += 1
            continue

        # We are at the start of a table. It only truly has a header when
        # the first row is followed by a |---|---| separator row. A lone
        # table row with no separator is DATA, never a header — treating
        # it as a header would silently throw the row away.
        header = []
        if (not is_separator_row(line)
                and i + 1 < n and is_separator_row(lines[i + 1])):
            header = [line, lines[i + 1]]
            i += 2
        elif is_separator_row(line):
            i += 1  # stray separator with no header row above it — skip

        # Now walk the data rows, grouping them into records.
        record = []
        while i < n:
            row = lines[i]
            if not is_table_row(row):
                # a blank line BETWEEN rows doesn't end the table
                if not row.strip() and i + 1 < n and is_table_row(lines[i + 1]):
                    i += 1
                    continue
                break
            if starts_new_record(row) and record:
                units.append({"lines": record, "header": header})
                record = [row]
            else:
                record.append(row)
            i += 1
        if record:
            units.append({"lines": record, "header": header})
    return units

def split_oversized_unit(unit):
    """Safety net: if one single record is bigger than the hard cap
    (very rare), split it row by row so nothing exceeds the model limit."""
    small = []
    part = []
    part_tokens = 0
    for row in unit["lines"]:
        row_tokens = count_tokens(row)
        if part and part_tokens + row_tokens > HARD_CAP_TOKENS - 60:
            small.append({"lines": part, "header": unit["header"]})
            part = [row]
            part_tokens = row_tokens
        else:
            part.append(row)
            part_tokens += row_tokens
    if part:
        small.append({"lines": part, "header": unit["header"]})
    return small

def split_large_section(text, target_tokens=TARGET_TOKENS):
    """Split a big section into pieces of roughly target_tokens each.

    Think of it as packing boxes: units (text lines / whole table records)
    are the items, chunks are the boxes. We close a box when the next item
    would not fit, and we never saw an item in half. Chunks that start
    mid-table get the table's header repeated on top.
    """
    lines = text.split('\n')
    units = build_units(lines)

    # Break up any freak unit that alone exceeds the hard cap
    safe_units = []
    for u in units:
        u_tokens = count_tokens('\n'.join(u["lines"]))
        if u_tokens > HARD_CAP_TOKENS - 60:
            safe_units.extend(split_oversized_unit(u))
        else:
            safe_units.append(u)

    pieces = []
    current = []           # lines of the chunk we are filling right now
    current_tokens = 0
    emitted_headers = set()  # headers already written into this chunk

    def close_piece():
        nonlocal current, current_tokens, emitted_headers
        if any(l.strip() for l in current):
            pieces.append('\n'.join(current))
        current = []
        current_tokens = 0
        emitted_headers = set()

    for u in safe_units:
        u_text = '\n'.join(u["lines"])
        u_tokens = count_tokens(u_text)

        # Would this unit overflow the current chunk? Then close the chunk.
        if current and current_tokens + u_tokens > target_tokens:
            close_piece()

        # Before this table's first record in this chunk, write its header —
        # whether the chunk starts with the table or the table starts mid-chunk.
        if u["header"] and id(u["header"]) not in emitted_headers:
            current.extend(u["header"])
            current_tokens += count_tokens('\n'.join(u["header"]))
            emitted_headers.add(id(u["header"]))

        current.extend(u["lines"])
        current_tokens += u_tokens

    close_piece()
    return [p for p in pieces if p.strip()]

# ─── Step 5: Split document on ## headings and build chunks ──────────────────
sections = re.split(r'(?=^## )', content, flags=re.MULTILINE)

# Build position map to avoid duplicate-match bug
section_positions = []
search_start = 0
for section in sections:
    pos = content.find(section[:80], search_start)
    if pos == -1:
        pos = content.find(section[:40], 0)
    section_positions.append(pos)
    if pos != -1:
        search_start = pos + 1

chunks = []

def add_chunk(section_name, page_num, text):
    chunks.append({
        "chunk_id": "",  # filled in at the end, after merging
        "plan": plan_name,
        "source_file": source_file,
        "UIN": uin,
        "section": section_name,
        "page": page_num,
        "text": text,
        "char_count": len(text)
    })

for idx, section in enumerate(sections):
    if not section.strip():
        continue

    lines = section.strip().split('\n')
    heading_line = lines[0] if lines else ''
    heading = heading_line.replace('## ', '').strip()

    if not heading:
        continue

    section_position = section_positions[idx] if idx < len(section_positions) else 0
    if section_position == -1:
        section_position = 0
    page_num = get_page_for_position(section_position, page_markers)

    cleaned = clean_text(section).strip()

    if not cleaned or len(cleaned) < 10:
        continue

    if count_tokens(cleaned) > MAX_SECTION_TOKENS:
        pieces = split_large_section(cleaned)
        for i, piece in enumerate(pieces):
            text = piece.strip()
            # Parts after the first lost their section heading in the split,
            # so their text no longer says what topic it belongs to. Example:
            # the list of 24-month waiting diseases (cataracts...) is in part 2,
            # and without the heading a search for "cataract waiting period"
            # can't tell it's about waiting periods. Put the heading back.
            if i > 0:
                text = f"## {heading} (continued)\n\n{text}"
            add_chunk(f"{heading} (part {i+1})", page_num, text)
    else:
        add_chunk(heading, page_num, cleaned)

# ─── Step 6: Merge heading-only chunks into the next chunk ───────────────────
merged_chunks = []
i = 0
while i < len(chunks):
    current = chunks[i]

    # A chunk is heading-only if stripping the ## line leaves almost nothing
    text_without_heading = current['text'].replace(f"## {current['section']}", '').strip()
    is_heading_only = len(text_without_heading) < 20

    if is_heading_only and i + 1 < len(chunks):
        # Prepend this heading to the next chunk's text
        next_chunk = chunks[i + 1]
        next_chunk['text'] = current['text'] + '\n\n' + next_chunk['text']
        next_chunk['char_count'] = len(next_chunk['text'])
        i += 1  # skip current — it is absorbed into next
    else:
        merged_chunks.append(current)
        i += 1

chunks = merged_chunks

# ─── Step 7: Merge tiny chunks into the next chunk ───────────────────────────
# Very small chunks (like a 100-char definition) carry too little meaning to
# be found on their own, so we glue neighbors together until they are big
# enough. Example: definitions "40. Non-Network" and "41. Network Provider"
# end up in one chunk instead of two crumbs.
merged_chunks = []
i = 0
while i < len(chunks):
    current = chunks[i]

    # Keep absorbing following chunks while current is still too small —
    # but never let a merge push the result past the model's token limit
    while current['char_count'] < MIN_CHUNK_CHARS and i + 1 < len(chunks):
        nxt = chunks[i + 1]
        merged_text = current['text'] + '\n\n' + nxt['text']
        if count_tokens(merged_text) > 500:
            break  # better a small chunk than a truncated one
        current['text'] = merged_text
        current['section'] = f"{current['section']} + {nxt['section']}"
        current['char_count'] = len(current['text'])
        i += 1

    # If the very last chunk is tiny, glue it onto the previous one instead
    if (current['char_count'] < MIN_CHUNK_CHARS and merged_chunks
            and count_tokens(merged_chunks[-1]['text'] + '\n\n' + current['text']) <= 500):
        prev = merged_chunks[-1]
        prev['text'] = prev['text'] + '\n\n' + current['text']
        prev['section'] = f"{prev['section']} + {current['section']}"
        prev['char_count'] = len(prev['text'])
    else:
        merged_chunks.append(current)
    i += 1

chunks = merged_chunks

# Number the chunk IDs now that merging is finished
for idx, chunk in enumerate(chunks):
    chunk['chunk_id'] = f"{chunk_prefix}_{idx+1:03d}"

# ─── Step 8: Sanity checks ───────────────────────────────────────────────────
print(f"\nTotal chunks created: {len(chunks)}")

sizes = [c['char_count'] for c in chunks]
print(f"Chunk size (chars): min={min(sizes)}, avg={sum(sizes)//len(sizes)}, max={max(sizes)}")

token_sizes = [count_tokens(c['text']) for c in chunks]
print(f"Chunk size (tokens): min={min(token_sizes)}, avg={sum(token_sizes)//len(token_sizes)}, max={max(token_sizes)}")

# 512 is the embedding model's hard reading limit — nothing may exceed it
too_big = [(c, t) for c, t in zip(chunks, token_sizes) if t > 512]
print(f"Chunks over the model's 512-token limit: {len(too_big)}")
for c, t in too_big:
    print(f"  - {c['chunk_id']}: {c['section']} ({t} tokens)")

# Data-integrity check: every meaningful line of the cleaned document must
# appear in some chunk. If this ever reports losses, the chunker ate text.
all_chunk_text = '\n'.join(c['text'] for c in chunks)
lost_lines = []
for line in clean_text(content).split('\n'):
    line = line.strip()
    if len(line) > 15 and line not in all_chunk_text:
        lost_lines.append(line)
print(f"Document lines missing from all chunks: {len(lost_lines)}")
for line in lost_lines[:10]:
    print(f"  LOST: {line[:100]}")

too_small = [c for c in chunks if c['char_count'] < MIN_CHUNK_CHARS]
print(f"Chunks under {MIN_CHUNK_CHARS} chars: {len(too_small)}")
for c in too_small:
    print(f"  - {c['chunk_id']}: {c['section']} ({c['char_count']} chars)")

print("\n--- SAMPLE CHUNK 1 ---")
c = chunks[0]
print(f"section: {c['section']}\npage: {c['page']}\nchar_count: {c['char_count']}")
print(f"text preview:\n{c['text'][:200]}")

# ─── Step 9: Save ────────────────────────────────────────────────────────────
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(chunks)} chunks to {output_file}")
