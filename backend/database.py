from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Aici decidem ca baza de date se numeste sql_app.db si se creaza automat in folder
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# O functie helper pe care o vom folosi la fiecare request pentru a vorbi cu Baza de Date
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
