from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from .core import DATA

engine = create_engine(f'sqlite:///{DATA / "app.db"}', connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(bind=engine)
class Base(DeclarativeBase): pass
class Document(Base):
    __tablename__ = 'documents'
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    category = Column(String, default='policy')
    academic_year = Column(String, default='all')
    program = Column(String, default='all')
    department = Column(String, default='all')
    status = Column(String, default='draft')
    checksum = Column(String, nullable=False)
    effective_from = Column(String, default='')
    effective_to = Column(String, default='')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    chunks = Column(Integer, default=0)
    error = Column(Text, default='')
class Chunk(Base):
    __tablename__ = 'chunks'
    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey('documents.id'), index=True)
    text = Column(Text, nullable=False)
    page = Column(Integer, default=1)
    chunk_index = Column(Integer, default=0)
class Feedback(Base):
    __tablename__ = 'feedback'
    id = Column(String, primary_key=True)
    question = Column(Text)
    answer = Column(Text)
    helpful = Column(Integer)
    comment = Column(Text, default='')
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
Base.metadata.create_all(engine)
