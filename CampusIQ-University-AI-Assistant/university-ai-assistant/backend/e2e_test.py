import sys
sys.path.insert(0, '.')
from app.core import settings
from app.answer import generate
from app.rag import retriever
from app.db import SessionLocal, Document, Chunk
from sqlalchemy import select

print("Settings loaded:")
print("  API key:", settings.gemini_api_key[:10] + "...")
print("  Model:", settings.gemini_model)
print("  Demo mode:", settings.demo_mode)

# Get published docs
with SessionLocal() as db:
    docs = db.scalars(select(Document).where(Document.status=="published")).all()
    print(f"\nPublished documents: {len(docs)}")
    for d in docs:
        print(f"  - {d.title} ({d.chunks} chunks)")
    chunks = db.scalars(select(Chunk)).all()
    print(f"Total chunks: {len(chunks)}")
    candidates = [(c, next(d for d in docs if d.id == c.document_id)) for c in chunks]

# Test retrieval
print("\nTesting retrieval for 'attendance requirement'...")
ranked = retriever.search("attendance requirement", candidates)
print(f"Retrieved {len(ranked)} passages total")
above = [r for r in ranked if r[0] >= 0.38]
print(f"Above 0.38 threshold: {len(above)}")
for score, c, d in ranked[:3]:
    print(f"  [{score:.3f}] {d.title} p{c.page}: {c.text[:100]}...")

# Test answer generation
evidence = [
    {
        "chunk_id": c.id,
        "document_id": d.id,
        "title": d.title,
        "page": c.page,
        "text": c.text,
        "score": round(float(score), 3),
    }
    for score, c, d in ranked if score >= 0.38
]
print(f"\nEvidence passed to LLM: {len(evidence)} passages")
if evidence:
    result = generate("What is the attendance requirement?", evidence)
    print(f"Status: {result['status']}")
    print(f"Answer: {result['answer'][:300]}")
else:
    print("WARNING: No evidence above 0.38 threshold!")
    print("The bot will always say 'I could not find enough verified information'")
    print("Either the documents don't have relevant content, or embeddings need re-indexing.")
