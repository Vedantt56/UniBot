import hashlib, hmac, time
from fastapi import HTTPException, Request
from .core import settings

def token():
    expiry = str(int(time.time()) + 8*3600)
    mac = hmac.new(settings.session_secret.encode(), expiry.encode(), hashlib.sha256).hexdigest()
    return expiry+'.'+mac

def admin_required(request: Request):
    header = request.headers.get('authorization', '')
    value = header.removeprefix('Bearer ').strip()
    try:
        expiry, mac = value.split('.', 1)
        expected = hmac.new(settings.session_secret.encode(), expiry.encode(), hashlib.sha256).hexdigest()
        if int(expiry) > time.time() and hmac.compare_digest(expected, mac): return True
    except Exception: pass
    raise HTTPException(status_code=401, detail='Admin authorization required')
