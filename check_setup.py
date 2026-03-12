import os
from dotenv import load_dotenv

def check():
    load_dotenv()
    
    keys = [
        "OPENROUTER_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "TEAM_NAME"
    ]
    
    print("--- Verifica Setup ---")
    all_ok = True
    for key in keys:
        value = os.getenv(key)
        if value:
            print(f"[OK] {key} è presente.")
        else:
            print(f"[ERRORE] {key} manca nel file .env!")
            all_ok = False
            
    if all_ok:
        print("\nSetup completato con successo! Puoi iniziare a sviluppare.")
    else:
        print("\nCi sono dei problemi nel file .env. Controlla i messaggi sopra.")

if __name__ == "__main__":
    check()
