import sqlite3

conn = sqlite3.connect('data/app.db')
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", cur.fetchall())

cur.execute("SELECT id, title, status, chunks FROM documents")
rows = cur.fetchall()
print("\nDocuments:")
for r in rows:
    print(r)

cur.execute("SELECT COUNT(*) FROM chunks")
print("\nTotal chunks in DB:", cur.fetchone()[0])

conn.close()
