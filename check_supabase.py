"""Supabase connectivity and schema verification script."""

from supabase import create_client
from rag_platform.config import get_settings

settings = get_settings()

url = settings.SUPABASE_URL
key = settings.effective_supabase_key

print("URL set:", bool(url))
print("Key set:", bool(key))

if url and key:
    try:
        supabase = create_client(url, key)
        print("Client initialized OK")
        resp = supabase.table("documents").select("id").limit(1).execute()
        print("Table check OK, rows:", len(resp.data) if resp.data else 0)
    except Exception as exc:
        print("Supabase check failed:", exc)
else:
    print("Supabase credentials not fully configured in environment.")
