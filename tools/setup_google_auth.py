"""
Autenticador interativo do Google Workspace (Google Calendar & Gmail) para a Aigis.
Gera o token.json para permitir leitura de compromissos e triagem de emails.
"""

import os
import sys
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow

# Escopos necessários para a Aigis (Leitura + Criação de Eventos no Calendário)
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
]

def authenticate_google():
    cred_path = Path("credentials.json")
    token_path = Path("token.json")

    if not cred_path.exists():
        print("\n❌ Arquivo 'credentials.json' não encontrado na raiz do projeto!")
        print("Para conectar o Google Workspace:")
        print("1. Acesse o Google Cloud Console: https://console.cloud.google.com")
        print("2. Crie um projeto ou selecione um existente.")
        print("3. Ative as APIs 'Google Calendar API' e 'Gmail API'.")
        print("4. Em 'APIs & Services' > 'Credentials', crie um OAuth 2.0 Client ID (tipo: Desktop App).")
        print("5. Baixe o arquivo JSON e salve como 'credentials.json' nesta pasta:\n   ", cred_path.resolve())
        return False

    print("\n🔑 Iniciando fluxo OAuth2 local do Google...")
    flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), SCOPES)
    creds = flow.run_local_server(port=0)

    with open(token_path, "w", encoding="utf-8") as token_file:
        token_file.write(creds.to_json())

    print(f"✅ Autenticação concluída com sucesso! Token salvo em: {token_path.resolve()}")
    return True

if __name__ == "__main__":
    authenticate_google()
