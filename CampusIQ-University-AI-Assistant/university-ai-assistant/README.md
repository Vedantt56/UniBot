# CampusIQ — University Document-Grounded Assistant

A runnable full-stack MVP with a **React student chatbot**, protected **React admin dashboard**, **FastAPI**, **SQLite**, **ChromaDB** and **Gemini 2.5 Flash**. It answers only using approved documents and refuses when the retrieved evidence is inadequate. This is a demonstration, not a certified university policy system.

## Requirements

- Python 3.11–3.13, Node.js 20+, npm
- Gemini API key from Google AI Studio for embedding and answer generation
- Internet access for the Google Gemini API (unless using offline demo mode)

## Run it (Windows PowerShell)

```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env: set GEMINI_API_KEY, ADMIN_EMAIL, ADMIN_PASSWORD, SESSION_SECRET
uvicorn app.main:app --reload
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. Student chat is public; admin UI is at **/admin**. Never put the Gemini API key in the frontend.

### Linux/macOS

Replace the activation command with `source .venv/bin/activate` and `Copy-Item` with `cp`.

### Upload the example documents

1. Go to `/admin`; sign in with the admin email/password in `backend/.env`.
2. Select **Upload document** and upload text files from `sample_documents/`.
3. Repeat for each file; the Gemini Embedding API indexes the text in ChromaDB.
4. Go to **Knowledge base**, preview chunks, **Publish** each approved document.
5. Return to the student chat. Ask: *What is the minimum attendance requirement?*
6. Test: *What is the dean's personal phone number?* The bot should abstain.

The files are **fictional**; never present them as real university policy. In production, upload genuine approved documents with proper effective dates and academic years.

## Offline demo mode (without API key)

Set `DEMO_MODE=true` in `backend/.env` and restart. Upload and publish the sample text files; offline retrieval uses a lightweight lexical ranker. Answers are intentionally **not AI-generated** in offline mode, and the UI displays the retrieved passages rather than inventing policy answers. Switch `DEMO_MODE=false` and supply `GEMINI_API_KEY` for the full Gemini + ChromaDB RAG workflow. Re-upload/re-index documents indexed during demo mode because demo mode intentionally does not populate ChromaDB.

## Tests

```bash
cd backend
python -m pytest -q
```

Frontend production build: `cd frontend && npm run build`.

## API

- `GET /api/health`
- `POST /api/chat`: `{ "question":"...", "history":[], "academic_year":"all", "program":"all" }`
- `POST /api/feedback`
- `POST /api/admin/login` (returns expiring HMAC-authenticated bearer token)
- `GET /api/admin/documents`, `POST /api/admin/documents` (multipart file + metadata)
- `GET /api/admin/documents/{id}/chunks`, `POST /api/admin/documents/{id}/publish`, `POST /api/admin/documents/{id}/archive`
- `POST /api/admin/test-query`, `GET /api/admin/feedback`
- `GET /api/sources/{id}` (only published documents)

Swagger UI: **http://localhost:8000/docs**.

## Reliability and security boundaries

- Draft or archived documents cannot be used by the public retrieval service.
- Document files are stored under random server-generated names, not user-controlled paths.
- PDFs are processed as text. Scanned PDFs without embedded text are rejected; OCR is not implemented.
- Gemini generation receives bounded retrieved evidence with passage IDs and a strict grounding prompt.
- A second Gemini call checks whether claims in the draft have support. This is a best-effort check, **not a mathematical guarantee of no hallucinations**.
- Missing or conflicting evidence should cause refusal; source-level conflict detection and citation correctness should be tested against your real policies before actual university use.
- API keys are server-side only. Passwords and secrets belong in `.env`, never git.
- Basic admin token authentication is for **local MVP use**. For production, add password hashing, dedicated admin accounts, robust session management, CSRF protections where needed, rate limiting, abuse protection and comprehensive audit logging. The current example uses a shared credential configured via environment, not an account management system.
- Public chat is stateless. Feedback is stored but student conversations are not saved by default.
- Document metadata validity and embedding generation affect output. Monitor costs/rate limits when using the Gemini API.

## Known MVP limitations

- The source verification step is LLM-based rather than independently formally proven; evaluate using a held-out QA set.
- Archived chunks remain in ChromaDB but are blocked by the SQLite status filter; purge them during maintenance before broad deployment.
- No OCR, multilingual UI, individual student accounts, private student records, streaming, university website widget or background queue.
- Documents are indexed sequentially during upload, so large PDFs can exceed HTTP timeouts. For university deployment, add asynchronous ingestion workers.
- Only documents with extracted text and correctly configured embedding models are retrievable.
