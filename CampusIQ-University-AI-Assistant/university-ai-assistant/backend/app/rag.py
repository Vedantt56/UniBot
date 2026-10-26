import re, math
from collections import Counter
from .core import DATA, settings

STOP={'the','for','and','are','is','in','a','an','do','does','to','of','me','what','how','can','my','i','you','about','tell','please','that','with','be'}
def tokens(t): return [v for v in re.findall(r'[a-z0-9]+',t.lower()) if v not in STOP]

def chunk_text(text, target=1500, overlap=180):
    text = re.sub(r'[ \t]+', ' ', text).strip()
    paragraphs = [x.strip() for x in re.split(r'\n\s*\n', text) if x.strip()]
    out=[]; current=''
    for para in paragraphs:
        if len(current)+len(para)+2 <= target: current = (current+'\n\n'+para).strip(); continue
        if current: out.append(current)
        if len(para)>target:
            step=target-overlap
            for i in range(0,len(para),step): out.append(para[i:i+target])
            current=''
        else: current=(out[-1][-overlap:]+'\n'+para if out else para)
    if current: out.append(current)
    return [x for x in out if len(x)>15]

def extract_pages(path):
    ext=path.suffix.lower()
    if ext=='.pdf':
        import fitz
        with fitz.open(path) as pdf: return [(i+1,p.get_text(sort=True)) for i,p in enumerate(pdf)]
    if ext=='.docx':
        from docx import Document
        return [(1, '\n\n'.join(p.text for p in Document(path).paragraphs))]
    if ext=='.txt': return [(1,path.read_text('utf-8'))]
    raise ValueError('Only PDF, DOCX and TXT files are supported')

class Retriever:
    def __init__(self): self.collection=None; self.client=None
    def _collection(self):
        if self.collection is None:
            import chromadb
            self.client=chromadb.PersistentClient(path=str(DATA/'chroma'))
            self.collection=self.client.get_or_create_collection('university_chunks', metadata={'hnsw:space':'cosine'})
        return self.collection
    def _embed(self, texts, task):
        from google import genai
        from google.genai import types
        client=genai.Client(api_key=settings.gemini_api_key)
        result=[]
        for text in texts:
            r=client.models.embed_content(model=settings.embedding_model, contents=text,
                config=types.EmbedContentConfig(task_type=task,output_dimensionality=settings.embedding_dimensions))
            v=r.embeddings[0].values
            norm=math.sqrt(sum(z*z for z in v)) or 1
            result.append([float(z/norm) for z in v])
        return result
    def index(self, chunks):
        if settings.demo_mode: return
        if not settings.gemini_api_key: raise ValueError('GEMINI_API_KEY must be configured to index documents')
        coll=self._collection()
        for chunk in chunks:
            vec=self._embed([chunk.text], 'RETRIEVAL_DOCUMENT')[0]
            coll.upsert(ids=[chunk.id], embeddings=[vec], documents=[chunk.text], metadatas=[{'document_id':chunk.document_id,'page':chunk.page}])
    def delete(self, chunk_ids):
        if not settings.demo_mode and chunk_ids: self._collection().delete(ids=chunk_ids)
    def search(self, question, candidates, limit=6):
        if not candidates: return []
        if settings.demo_mode:
            q=Counter(tokens(question))
            scored=[]
            for c,d in candidates:
                t=Counter(tokens(c.text)); intersection=sum(min(t[w],n) for w,n in q.items())
                score=intersection/max(1,math.sqrt(sum(t.values())*sum(q.values())))
                if score>0: scored.append((score,c,d))
            return sorted(scored, key=lambda a:-a[0])[:limit]
        vec=self._embed([question], 'RETRIEVAL_QUERY')[0]
        coll=self._collection()
        count=min(coll.count(), max(12, limit*5))
        if not count:return []
        results=coll.query(query_embeddings=[vec], n_results=count)
        lookup={c.id:(c,d) for c,d in candidates}
        scored=[]
        for chunk_id,distance in zip(results['ids'][0],results['distances'][0]):
            if chunk_id in lookup:
                c,d=lookup[chunk_id]; scored.append((1-float(distance),c,d))
        return sorted(scored,key=lambda a:-a[0])[:limit]
retriever=Retriever()
