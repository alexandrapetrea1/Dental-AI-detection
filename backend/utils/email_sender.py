import smtplib
from email.message import EmailMessage
import os

# Acestea vor fi completate cu adresa ta reala
# Ideal ar fi sa le luam din variabile de mediu (.env), dar pentru demo le punem aici sau le cerem la configurare.

SENDER_EMAIL = "dentalclinicalexandra@gmail.com" # Vei schimba cu adresa pe care o creezi 
SENDER_PASSWORD = "aici-pui-parola-de-aplicatie" # Vei pune parola de aplicatie furnizata de Google
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465 # SSL port

def send_reset_email(to_email: str, reset_link: str):
    """Trimite emailul de resetare in mod real."""
    if SENDER_PASSWORD == "aici-pui-parola-de-aplicatie":
        print(f"\n[AVERTISMENT] Emailul NU s-a trimis cu adevarat. Trebuie sa completezi SENDER_EMAIL si SENDER_PASSWORD in utils/email_sender.py\nLink-ul este: {reset_link}\n")
        return False

    msg = EmailMessage()
    msg['Subject'] = 'Dental AI Clinic - Password Reset Request'
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email

    content = f"""
    Draga Doctorule,

    Am primit o cerere pentru a reseta parola contului tau de pe platforma Dental AI Clinic.
    
    Te rugam sa dai click pe link-ul de mai jos pentru a seta o noua parola. Link-ul este valabil timp de o ora:
    {reset_link}

    Daca nu ai solicitat tu aceasta resetare, te rugam sa ignori acest email.

    Cu respect,
    Echipa Dental AI Clinic
    """
    msg.set_content(content)

    try:
        # Ne conectam la serverul Gmail in mod securizat
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"Eroare la trimiterea emailului: {e}")
        return False
