import hashlib, hmac, shutil, uuid
from datetime import date
from pathlib import Path
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from .core import settings, UPLOADS
from .db import SessionLocal, Document, Chunk, Feedback
from .security import admin_required, token
from .rag import extract_pages, chunk_text, retriever
from .answer import generate, REFUSAL

app=FastAPI(title='University AI Assistant', version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=[settings.frontend_origin],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
class Login(BaseModel): email:str; password:str
class Ask(BaseModel):
    question: str = Field(min_length=2,max_length=1200)
    history: list[dict] = Field(default_factory=list,max_length=8)
    academic_year: str = 'all'
    program: str = 'all'
class Vote(BaseModel): question:str=''; answer:str=''; helpful:bool; comment:str=''
@app.get('/api/health')
def health(): return {'ok':True,'mode':'demo' if settings.demo_mode else 'gemini','configured':bool(settings.gemini_api_key)}
@app.post('/api/admin/login')
def login(data:Login):
    if not hmac.compare_digest(data.email, settings.admin_email) or not hmac.compare_digest(data.password,settings.admin_password):
        raise HTTPException(401,'Invalid credentials')
    return {'token':token()}

def serialize(d):
    return {k:getattr(d,k) for k in ['id','title','filename','category','academic_year','program','department','status','chunks','error','effective_from','effective_to']}
@app.get('/api/admin/documents')
def documents(_=Depends(admin_required)):
    with SessionLocal() as db: return [serialize(d) for d in db.scalars(select(Document).order_by(Document.created_at.desc())).all()]
@app.post('/api/admin/documents')
async def upload(file:UploadFile=File(...),title:str=Form(''),category:str=Form('policy'),academic_year:str=Form('all'),program:str=Form('all'),department:str=Form('all'), effective_from:str=Form(''),effective_to:str=Form(''),_=Depends(admin_required)):
    ext=Path(file.filename or '').suffix.lower()
    if ext not in {'.pdf','.docx','.txt'}:raise HTTPException(400,'Only PDF, DOCX and TXT documents are supported')
    data=await file.read(settings.max_upload_mb*1024*1024+1)
    if len(data)>settings.max_upload_mb*1024*1024: raise HTTPException(413,'File exceeds upload limit')
    if not data: raise HTTPException(400,'Empty file')
    digest=hashlib.sha256(data).hexdigest()
    with SessionLocal() as db:
        if db.scalar(select(Document).where(Document.checksum==digest,Document.status!='archived')):raise HTTPException(409,'This document has already been uploaded')
    if effective_from and effective_to and effective_from>effective_to:raise HTTPException(400,'Invalid effective date range')
    for value in [effective_from,effective_to]:
        if value:
            try: date.fromisoformat(value)
            except ValueError:raise HTTPException(400,'Dates must use YYYY-MM-DD')
    doc_id=uuid.uuid4().hex
    path=UPLOADS/(doc_id+ext)
    path.write_bytes(data)
    doc=Document(id=doc_id,title=(title.strip() or file.filename or 'Untitled')[:180],filename=file.filename,category=category[:60],academic_year=academic_year[:30],program=program[:70],department=department[:70],checksum=digest,effective_from=effective_from,effective_to=effective_to)
    try:
        chunks=[]
        for page,text in extract_pages(path):
            for part in chunk_text(text): chunks.append(Chunk(id=uuid.uuid4().hex,document_id=doc.id,text=part,page=page,chunk_index=len(chunks)))
        if not chunks:raise ValueError('No extractable text. Scanned PDFs need OCR, which is not supported in this MVP.')
        retriever.index(chunks)
        doc.chunks=len(chunks)
        with SessionLocal() as db:
            db.add(doc); db.add_all(chunks); db.commit(); db.refresh(doc); result=serialize(doc)
    except Exception as e:
        try: retriever.delete([c.id for c in locals().get('chunks',[])])
        except Exception: pass
        path.unlink(missing_ok=True)
        raise HTTPException(422,f'Indexing failed: {str(e)[:260]}')
    return result
@app.post('/api/admin/documents/{doc_id}/publish')
def publish(doc_id:str,_=Depends(admin_required)):
    with SessionLocal() as db:
        d=db.get(Document,doc_id)
        if not d:raise HTTPException(404,'Document not found')
        if not d.chunks:raise HTTPException(409,'Document has no indexed chunks')
        d.status='published';db.commit();db.refresh(d);return serialize(d)
@app.post('/api/admin/documents/{doc_id}/archive')
def archive(doc_id:str,_=Depends(admin_required)):
    with SessionLocal() as db:
        d=db.get(Document,doc_id)
        if not d:raise HTTPException(404,'Document not found')
        d.status='archived';db.commit();return serialize(d)
@app.get('/api/admin/documents/{doc_id}/chunks')
def view_chunks(doc_id:str,_=Depends(admin_required)):
    with SessionLocal() as db:
        if not db.get(Document,doc_id):raise HTTPException(404)
        return [{'id':c.id,'page':c.page,'text':c.text} for c in db.scalars(select(Chunk).where(Chunk.document_id==doc_id)).all()]
@app.get('/api/sources/{doc_id}')
def source_file(doc_id:str):
    with SessionLocal() as db:
        d=db.get(Document,doc_id)
        if not d or d.status!='published':raise HTTPException(404,'Source not found')
        path=UPLOADS/(d.id+Path(d.filename).suffix.lower())
        if not path.exists():raise HTTPException(404)
        return FileResponse(path,filename=d.filename,content_disposition_type='inline')

def evidence_for(question,academic_year='all',program='all'):
    today=date.today().isoformat()
    with SessionLocal() as db:
        docs=db.scalars(select(Document).where(Document.status=='published')).all()
        valid={d.id:d for d in docs if (d.academic_year in ('all',academic_year) or academic_year=='all') and (d.program in ('all',program) or program=='all') and (not d.effective_from or d.effective_from<=today) and (not d.effective_to or d.effective_to>=today)}
        candidates=[(c,valid[c.document_id]) for c in db.scalars(select(Chunk).where(Chunk.document_id.in_(list(valid)))).all()] if valid else []
        ranked=retriever.search(question,candidates)
        if settings.demo_mode: ranked=[r for r in ranked if r[0]>=.04]
        else: ranked=[r for r in ranked if r[0]>=.38]
        return [{'chunk_id':c.id,'document_id':d.id,'title':d.title,'page':c.page,'text':c.text,'score':round(float(score),3)} for score,c,d in ranked[:6]]
@app.post('/api/chat')
def chat(data:Ask):
    question=data.question.strip()
    history=data.history[-4:]
    # Previous conversation can disambiguate the search but never supplies policy facts.
    query=question
    if len(question.split())<8 and history:
        previous=next((str(m.get('content',''))[:180] for m in reversed(history) if m.get('role')=='user'), '')
        if previous: query=previous+' '+question
    try:
        evidence=evidence_for(query,data.academic_year,data.program)
        answer=generate(question,evidence)
        cited=set(answer['source_ids'])
        sources=[{'chunk_id':e['chunk_id'],'document_id':e['document_id'],'title':e['title'],'page':e['page'],'excerpt':e['text'][:1000],'score':e['score']} for e in evidence if e['chunk_id'] in cited]
        return {'status':answer['status'],'answer':answer['answer'],'sources':sources,'retrieved_count':len(evidence)}
    except Exception:
        return {'status':'service_error','answer':'I cannot verify the answer right now because the knowledge service is unavailable. Please try again later or contact the university office.','sources':[],'retrieved_count':0}
@app.post('/api/admin/test-query')
def test_query(data:Ask,_=Depends(admin_required)):
    try:return {'evidence':evidence_for(data.question,data.academic_year,data.program), 'response':chat(data)}
    except Exception as exc: raise HTTPException(503,str(exc)[:200])
@app.post('/api/feedback')
def feedback(data:Vote):
    with SessionLocal() as db:
        db.add(Feedback(id=uuid.uuid4().hex,question=data.question[:1200],answer=data.answer[:3000],helpful=int(data.helpful),comment=data.comment[:500]));db.commit()
    return {'ok':True}
@app.get('/api/admin/feedback')
def feedback_list(_=Depends(admin_required)):
    with SessionLocal() as db:
        return [{'question':f.question,'answer':f.answer,'helpful':bool(f.helpful),'comment':f.comment} for f in db.scalars(select(Feedback).order_by(Feedback.created_at.desc()).limit(100)).all()]
