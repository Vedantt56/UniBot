from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE = Path(__file__).resolve().parents[1]
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE/'.env'), extra='ignore')
    gemini_api_key: str = ''
    gemini_model: str = 'gemini-2.5-flash'
    embedding_model: str = 'gemini-embedding-001'
    embedding_dimensions: int = 768
    admin_email: str = 'admin@demo.edu'
    admin_password: str = 'change-this-long-admin-password'
    session_secret: str = 'change-this-to-a-random-32-character-secret'
    frontend_origin: str = 'http://localhost:5173'
    demo_mode: bool = False
    max_upload_mb: int = 20
settings = Settings()
DATA = BASE/'data'
UPLOADS = DATA/'uploads'
DATA.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)
