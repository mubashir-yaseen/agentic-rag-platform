# Evaluating the platform on real annual reports

`run_live_eval.py` calls your **deployed** API (`POST /api/v1/chat`, Company Research mode) with questions you
wrote from the real PDFs, and measures what actually matters:

| Metric | What it means |
|---|---|
| Answer accuracy | the answer contains every expected fact (numbers compared without commas) |
| Correct report / page cited | the sources list contains the right PDF (and page, if you give one) |
| Abstention | for questions the report cannot answer, the app says so instead of inventing an answer |
| Latency p50 / p95 | typical and worst-case seconds per question |

Questions that fail because of infrastructure (daily quota used up, server asleep, timeouts) are counted as
**errors**, not as wrong answers, and are excluded from the scores.

## 1. Write 20 questions (about 45 minutes)
Open each report in a PDF viewer and pick facts you can verify on a specific page.

| Type | How many | Example | Fields |
|---|---|---|---|
| Single figure | 8 | "What was profit after tax in 2025?" | `expected`: the figure exactly as printed |
| Table value or two figures | 4 | "What were total deposits and total assets?" | `expected`: both figures |
| Facts in text | 4 | "Who is the chairman?" | `expected`: the name |
| **Cannot be answered** | 4 | "What was revenue in Japan in 2019?" | `"should_abstain": true` |

Spread them over at least 3 different companies. Copy the page number into `"pages": [N]` when you know it.
Use the company's `id` from the Supabase `companies` table as `company_id`.

Tips: write the figure exactly as the report prints it (commas do not matter). If a fact can be written two
ways (a name spelled differently), use `expected_any`.

## 2. Run it
```bash
cp eval/questions.example.json eval/questions.json     # then edit it
python eval/run_live_eval.py --base-url https://<your-api>.onrender.com --questions eval/questions.json
```
The first call may take about a minute if the free server is asleep. The runner waits 4 seconds between
questions by default because the free LLMs are rate limited (`--delay`).

## 3. Use the result
It prints a table and saves `eval/results.json` with every question, answer and source. Paste the table into
the main README, and use the real numbers in your resume ("answers X% of questions correctly with the correct
report cited"). Read the WRONG rows: they show what to improve (chunking, retrieval, prompt).

Run it again after every change that could affect quality, and note the date and the number of questions.
