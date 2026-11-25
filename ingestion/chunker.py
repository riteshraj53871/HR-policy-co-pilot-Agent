import re 

MAX_WORDS=250 
HEADING= re.compile(r"^(#{1,6})\s+(.+?)\s*$")

#section heading keywords  -> category. First match wins
CATEGORY_KEYWORDS = [
    ("offboarding", ["offboarding"]),
    ("onboarding", ["onboarding"]),
    ("leave", ["leave"]),
    ("remote_work", ["remote"]),
    ("attendance", ["working hours", "attendance"]),
    ("payroll", ["payroll", "salary"]),
    ("benefits", ["benefit"]),
    ("conduct", ["conduct"]),
    ("records", ["record"]),
]

#if the  documents title contains one of these words, the whole file is HR-only.range
HR_ONLY_MARKERS=("runbook","operation")

def doc_visibility(title: str)-> str:
    t=title.lower()
    return "hr_only" if any(m in t for m in HR_ONLY_MARKERS) else "all_employees"

def categorize(section: str) -> str:
    s = section.lower()
    for category, words in CATEGORY_KEYWORDS:
        if any(w in s for w in words):
            return category
    return "general"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def split_long(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Sentence-aware split with one sentence of overlap. Short text is returned unchanged."""
    if len(text.split()) <= max_words:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts, cur = [], []
    for s in sentences:
        if cur and sum(len(x.split()) for x in cur) + len(s.split()) > max_words:
            parts.append(" ".join(cur))
            cur = cur[-1:]
        cur.append(s)
    if cur:
        parts.append(" ".join(cur))
    return parts


def parse_markdown(text: str) -> tuple[str, list[dict]]:
    """Returns (document_title, sections). The first '#' heading is the title.
    Each section: {"path": ["Leave", "Annual Leave"], "text": "..."}"""
    title, sections, stack, buf = "", [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body and stack:
            sections.append({"path": [name for _, name in stack], "text": body})

    for line in text.splitlines():
        m = HEADING.match(line)
        if not m:
            buf.append(line)
            continue
        flush()
        buf.clear()
        level, name = len(m.group(1)), m.group(2)
        if level == 1 and not title:
            title = name
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, name))
    flush()
    return title, sections


def chunk_document(text: str, fallback_title: str) -> dict:
    """Returns {"title", "visibility", "chunks": [{"section", "heading_path",
    "category", "part", "text"}]}"""
    title, sections = parse_markdown(text)
    title = title or fallback_title
    chunks = []
    for sec in sections:
        for part, body in enumerate(split_long(sec["text"])):
            chunks.append(
                {
                    "section": sec["path"][-1],
                    "heading_path": " > ".join(sec["path"]),
                    "category": categorize(sec["path"][0]),
                    "part": part,
                    "text": body,
                }
            )
    return {"title": title, "visibility": doc_visibility(title), "chunks": chunks}