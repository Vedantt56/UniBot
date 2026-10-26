"""
Quick fix script: publishes all draft documents so the bot can search them.
Run from the backend directory:
  .venv\Scripts\python.exe publish_all_docs.py
"""
import sqlite3

conn = sqlite3.connect('data/app.db')
cur = conn.cursor()

cur.execute("SELECT id, title, status, chunks FROM documents WHERE status='draft'")
drafts = cur.fetchall()

if not drafts:
    print("No draft documents found.")
else:
    print(f"Publishing {len(drafts)} draft document(s):")
    for doc_id, title, status, chunks in drafts:
        print(f"  - [{doc_id[:8]}...] '{title}' ({chunks} chunks)")
        cur.execute("UPDATE documents SET status='published' WHERE id=?", (doc_id,))
    conn.commit()
    print("\n✅ All documents published. The bot can now search them.")

conn.close()
