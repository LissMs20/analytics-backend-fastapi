from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = 'postgresql://neondb_owner:npg_8AQlGHSkbPq5@ep-lucky-term-ac693w1y-pooler.sa-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'

if not SQLALCHEMY_DATABASE_URL:
    print("ERRO: Variável de ambiente DATABASE_URL não encontrada.")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True 
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()