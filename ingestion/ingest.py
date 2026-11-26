import argparse
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from app.services import embeddings, vector_store
from ingestion.chunker import chunk_document, slugify

DOCS_DIR = Path("data/hr_documents")
SUPPORTED = {".md", ".txt", ".pdf", ".docx"}


def load_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if ext == ".pdf":
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    if ext == ".docx":
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    return ""


def build_records(path: Path) -> list[dict]:
    doc_id = path.stem
    doc = chunk_document(load_text(path), fallback_title=doc_id.replace("_", " ").title())
    chunks = doc["chunks"]
    if not chunks:
        return []

    # Title + heading path go into the embedded text so each chunk carries its own context
    embed_inputs = [f"{doc['title']} - {c['heading_path']}\n{c['text']}" for c in chunks]
    vectors = embeddings.embed_documents(embed_inputs)

    return [
        {
            # Stable ID from the section name, so reordering sections never leaves stale vectors
            "id": f"{doc_id}-{slugify(c['heading_path'])}-{c['part']}",
            "values": vec,
            "metadata": {
                "text": c["text"],
                "doc_id": doc_id,
                "title": doc["title"],
                "section": c["section"],
                "heading_path": c["heading_path"],
                "category": c["category"],
                "visibility": doc["visibility"],
                "country": "ALL",
                "version": "n/a",
                "effective_date": "n/a",
                "source_path": path.relative_to(DOCS_DIR).as_posix(),
            },
        }
        for c, vec in zip(chunks, vectors)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest HR documents into Pinecone")
    parser.add_argument("--reset", action="store_true", help="Clear the namespace before ingesting")
    args = parser.parse_args()

    files = [p for p in sorted(DOCS_DIR.rglob("*")) if p.suffix.lower() in SUPPORTED]
    if not files:
        print(f"No documents found in {DOCS_DIR}")
        return

    vector_store.ensure_index()
    if args.reset:
        vector_store.reset_namespace()
        print("Namespace cleared.")

    total = 0
    for path in files:
        records = build_records(path)
        if not records:
            print(f"  WARNING {path.name}: no sections found (needs '##' headings), skipped")
            continue
        vector_store.upsert_chunks(records)
        total += len(records)
        visibility = records[0]["metadata"]["visibility"]
        print(f"  {path.name}: {len(records)} chunks [{visibility}]")
        for r in records:
            print(f"      - {r['metadata']['heading_path']} ({r['metadata']['category']})")

    print(f"\nDone. {total} chunks from {len(files)} files.")


if __name__ == "__main__":
    main()