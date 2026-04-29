from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from schemas.user import UserCreate, UserLogin, UserResponse, ForgotPasswordRequest, ResetPasswordRequest
from services import auth_service

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/signup", response_model=UserResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = auth_service.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_user_email = auth_service.get_user_by_email(db, email=user.email)
    if db_user_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    return auth_service.create_user(db=db, user=user)

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = auth_service.get_user_by_username(db, username=user.username)
    if not db_user or not auth_service.verify_password(user.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    return {"message": "Login successful", "username": db_user.username}

from utils import email_sender

@router.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = auth_service.get_user_by_email(db, email=req.email)
    
    # Conform dorinței explicite a user-ului, dăm eroare clară dacă nu există contul
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Această adresă de email nu este înregistrată în baza de date!"
        )
    
    token = auth_service.set_password_reset_token(db, user)
    reset_link = f"http://localhost:5173/reset-password?token={token}"
    
    # Trimitem ACUM ACEL E-MAIL REAL 
    email_sent = email_sender.send_reset_email(to_email=user.email, reset_link=reset_link)
    
    if not email_sent:
        # A crapat probabil pentru ca i-am zis utilizatorului sa puna parola si inca nu a pus-o
        return {"message": "Domeniul a generat linkul... DAR emailul real nu a fost setat in Utils. Verifica terminalul!"}

    return {"message": f"Link-ul de resetare a fost trimis cu succes catre adresa ta: {req.email}"}

@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    success = auth_service.reset_password_with_token(db, token=req.token, new_password=req.new_password)
    if not success:
        raise HTTPException(status_code=400, detail="Token invalid sau expirat.")
    
    return {"message": "Parola a fost schimbata cu succes!"}
