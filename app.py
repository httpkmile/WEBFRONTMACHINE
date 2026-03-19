import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from apps.mobile_app import app, serve

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Iniciando porta {port}...")
    serve(port=port)
