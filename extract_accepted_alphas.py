import json
import os

db_path = r"C:\Users\hardi\wq-alpha-research\alpha_db.json"
out_path = r"c:\Users\hardi\Downloads\Project Chronos\data\accepted_alphas.json"

if not os.path.exists(db_path):
    print(f"Error: {db_path} does not exist!")
    exit(1)

with open(db_path, "r", encoding="utf-8") as f:
    db = json.load(f)

alphas = db.get("alphas", {})
accepted = {}

for aid, a in alphas.items():
    st = a.get("status", "").strip().upper()
    if st in ["ACTIVE", "SUBMITTED"]:
        accepted[aid] = {
            "id": aid,
            "status": st,
            "expression": a.get("expression", ""),
            "original_sharpe": a.get("sharpe"),
            "original_fitness": a.get("fitness"),
            "original_returns": a.get("returns"),
            "original_drawdown": a.get("drawdown"),
            "original_turnover": a.get("turnover"),
            "settings": a.get("settings", {}),
            "date_ingested": a.get("date_submitted") or a.get("date_created") or "2026-09-29"
        }

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(accepted, f, indent=2)

print(f"Total alphas in DB: {len(alphas)}")
print(f"Accepted alphas extracted: {len(accepted)}")
status_counts = {}
for a in accepted.values():
    s = a["status"]
    status_counts[s] = status_counts.get(s, 0) + 1
print(f"Breakdown: {status_counts}")
