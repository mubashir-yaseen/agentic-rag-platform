"""Upload every annual-report PDF in a folder that is NOT yet indexed. Safe to run again and again."""
import argparse
import os
import subprocess
import time
from pathlib import Path

from supabase import create_client

COMPANY_MAPPING = {
    "BAHL": "483c8ed5-408d-4033-97a5-6f39628d477b",
    "SYS": "08051842-5cec-4d38-8b9e-5fdb936ab10a",
    "UBL": "53fee763-1e22-4c97-841a-b2036a8763ff",
    "MEBL": "7155bd3a-88c6-4cc3-b380-ae2c04641082",
    "OGDC": "85e9459e-a43c-42d0-a845-74c7302c5f6c",
    "FFC": "8fa6cb6f-b532-4795-8162-9011ab194bd6",
    "PPL": "93b3bd6f-960e-427c-948f-d4fbe80b49b1",
    "HBL": "a68c0d55-f57a-4edf-b79a-2d5b644414cc",
    "MARI": "b88224a5-a126-4862-ab4a-274f7d0d55d2",
    "EFERT": "eb30fb49-4170-4427-b838-fb9f19027722",
}
QUOTA_EXIT_CODE = 3  # upload_company_report.py exits with 3 when the free daily quota/budget is used up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--force", action="store_true", help="re-process reports that are already indexed")
    args = ap.parse_args()
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    def already_indexed(company_id, filename):
        try:
            rows = (client.table("documents").select("doc_id").eq("filename", filename)
                    .filter("metadata->>company_id", "eq", company_id).execute().data) or []
            for row in rows:
                got = client.table("chunks").select("chunk_id", count="exact").eq("doc_id", row["doc_id"]).limit(1).execute()
                if (got.count or 0) > 0:
                    return True
        except Exception as exc:
            print(f"   (could not check Supabase: {exc}; the file will be processed)")
        return False

    pending, done, unmatched = [], [], []
    for pdf in sorted(Path(args.folder).glob("*.pdf")):
        cid = next((c for k, c in COMPANY_MAPPING.items() if k in pdf.name.upper()), None)
        if not cid:
            unmatched.append(pdf.name)
        elif not args.force and already_indexed(cid, pdf.name):
            done.append(pdf.name)
        else:
            pending.append((pdf, cid))

    print(f"Already indexed ({len(done)}): {', '.join(done) or '-'}")
    print(f"To upload now  ({len(pending)}): {', '.join(p.name for p, _ in pending) or '-'}")
    if unmatched:
        print(f"No company match, ignored ({len(unmatched)}): {', '.join(unmatched)}")

    ok, failed, not_tried = [], [], []
    for i, (pdf, cid) in enumerate(pending):
        print(f"\n[*] Processing: {pdf.name}")
        code = subprocess.run(["python", "upload_company_report.py", "--pdf", str(pdf.absolute()),
                               "--company-id", cid, "--year", str(args.year)]).returncode
        if code == 0:
            ok.append(pdf.name)
            time.sleep(10)
        elif code == QUOTA_EXIT_CODE:
            failed.append(pdf.name + " (quota)")
            not_tried = [p.name for p, _ in pending[i + 1:]]
            print("\n[!] Free daily quota/budget used up. Stop here and run this cell again after 05:00 Pakistan time.")
            break
        else:
            failed.append(pdf.name)
            print(f"[x] {pdf.name} failed (not a quota problem). Continuing with the next report.")

    print("\n===== SUMMARY =====")
    print(f"Uploaded now : {', '.join(ok) or '-'}")
    print(f"Failed       : {', '.join(failed) or '-'}")
    print(f"Not tried    : {', '.join(not_tried) or '-'}")


if __name__ == "__main__":
    main()
