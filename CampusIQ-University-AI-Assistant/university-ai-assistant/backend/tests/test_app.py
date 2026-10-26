import os
os.environ['DEMO_MODE']='true'
os.environ['ADMIN_PASSWORD']='test-admin-password'
from fastapi.testclient import TestClient
from app.main import app
from app.rag import chunk_text, tokens
client=TestClient(app)
def test_health(): assert client.get('/api/health').status_code==200
def test_auth_required(): assert client.get('/api/admin/documents').status_code==401
def test_bad_password(): assert client.post('/api/admin/login',json={'email':'admin@demo.edu','password':'wrong'}).status_code==401
def test_admin_login():
    r=client.post('/api/admin/login',json={'email':'admin@demo.edu','password':'test-admin-password'})
    assert 'token' in r.json()
def test_empty_document_rejected():
    key=client.post('/api/admin/login',json={'email':'admin@demo.edu','password':'test-admin-password'}).json()['token']
    r=client.post('/api/admin/documents',headers={'Authorization':'Bearer '+key},files={'file':('blank.txt',b'','text/plain')})
    assert r.status_code==400
def test_hallucination_refusal_no_evidence():
    r=client.post('/api/chat',json={'question':'What is the imaginary intergalactic tuition waiver?'})
    assert r.json()['status'] in ('insufficient_evidence','service_error')
def test_chunking(): assert chunk_text('Attendance policy\n\n'+('Students must attend classes. '*50))
def test_feedback(): assert client.post('/api/feedback',json={'helpful':False,'question':'x','answer':'y'}).status_code==200

def test_document_lifecycle(tmp_path):
    from uuid import uuid4
    key=client.post('/api/admin/login',json={'email':'admin@demo.edu','password':'test-admin-password'}).json()['token']
    auth={'Authorization':'Bearer '+key}
    content=f'FICTIONAL POLICY {uuid4().hex}\n\nStudents must complete required coursework before examination registration. The academic office verifies registration.'.encode()
    uploaded=client.post('/api/admin/documents',headers=auth,files={'file':('policy.txt',content,'text/plain')})
    assert uploaded.status_code==200
    ident=uploaded.json()['id']
    assert uploaded.json()['status']=='draft'
    assert client.get(f'/api/sources/{ident}').status_code==404
    assert client.post(f'/api/admin/documents/{ident}/publish',headers=auth).status_code==200
    assert client.get(f'/api/sources/{ident}').status_code==200
    assert client.post(f'/api/admin/documents/{ident}/archive',headers=auth).status_code==200
    assert client.get(f'/api/sources/{ident}').status_code==404
