from sqlalchemy import Column, Integer, String, DateTime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String)
    
    # Campuri adaugate pentru Resetare Parola
    reset_token = Column(String, index=True, nullable=True)
    reset_token_expiry = Column(DateTime, nullable=True)
