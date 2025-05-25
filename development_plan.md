# Development Plan

Here’s a lean roadmap the coding assistant can follow. 

⸻

1. Project Skeleton (first 30 min)
	•	Create a fresh Git repo with app.py, tests/, README.md, and (optionally) requirements.txt.
	•	Add a .streamlit/config.toml that forces wide-mode so the four reports sit comfortably side-by-side when the window is large.
	•	Reflective check: Do we prefer the repo to live under the company’s GitHub org or a personal fork for faster iteration?

⸻

2. CSV Upload & Validation (≈1 hr)
	•	Wrap the entire import flow in one @st.experimental_singleton to avoid re-reading the file on every rerun.
	•	Use st.file_uploader with type=["csv"] and accept_multiple_files=False.
	•	Decode to StringIO and feed to pd.read_csv.
	•	Verify all 20 headers via set(headers) == expected_set; if any differ, raise st.error(f"Missing/incorrect header: {name}") and st.stop().
	•	Lower-case and filter Company Name == "rei".
	•	Parse Created At with errors='raise'; wrap in try/except to collect bad rows and display them.
	•	Coaching prompt: Do we want a spinner or a bare progress bar while parsing 50 k rows?

⸻

3. Common Data Prep (≈45 min)
	•	Derive customer_name = df["Shipping Name"].where(df["Shipping Name"].ne(""), df["Company Name"]).
	•	Group by customer_name and aggregate:
	•	order_count = size
	•	lifetime_total = Total.sum()
	•	first_order_date = Created At.min()
	•	last_order_date = Created At.max()
	•	Compute avg_days_between with np.where(order_count>1, (last−first)/(order_count−1), np.nan) and round to one decimal.
	•	Cache the final customer-level DataFrame with @st.cache_data(ttl=3600) to keep reruns <100 ms.

⸻

4. Report Builders (≈2 hr)
	•	One-timers: filter order_count==1 & last_order_date < today-180d; sort ascending by date.
	•	Dormant multi-buyers: filter order_count>=3 & last_order_date < today-90d; same sort.
	•	Repeat-buyer cadence: filter order_count>2; sort by avg_days_between descending; round cadence to an int for readability.
	•	High-value patterns:
	•	Grab nlargest(20, "lifetime_total").
	•	Convert to JSON via records orientation.
	•	Call openai.ChatCompletion.create(...) with your earlier prompt, timeout=10, max_retries=1.
	•	On failure: st.warning("LLM summary unavailable.").
	•	Each report lives in its own st.expander, followed by st.download_button.
	•	Reflective check: Do we want the raw tables hidden by default to keep the page tidy?

⸻

5. UX & Performance (≈30 min)
	•	Add st.status at the top after parsing: "Parsed {len(df)} orders for {n_customers} customers in {elapsed:.1f}s."
	•	Wrap each report render in st.spinner so all four appear within the 3 s target.
	•	Use st.dataframe over st.table for scrollable grids; pre-set column widths so money fields right-align.
	•	Insert st.caption("n/a") if a filter returns an empty DataFrame (single-row table looks clunky).

⸻

6. Testing (≈1 hr)
	•	Create tests/test_reports.py with fixtures for:
	•	Header-mismatch error.
	•	Bad datetime row detection.
	•	Each report filter returns known counts when fed a synthetic sample CSV.
	•	Run pytest -q locally; add a GitHub Actions workflow if you want CI later.
	•	Prompt: Will we eventually want CSV golden-files to guard against metric drift?

⸻

7. README & Deployment (≈25 min)
	•	Quick-start:
	•	pip install -r requirements.txt
	•	streamlit run app.py
	•	Vercel notes:
	•	Add vercel.json pointing to an api/index.py function that shells out streamlit run --server.port $PORT.
	•	Remind devs to set OPENAI_API_KEY in Vercel’s dashboard.  ￼
	•	Mention Streamlit Community Cloud as a fallback if Vercel’s beta Python runtime proves flaky.

⸻

8. requirements.txt (≈5 min)
streamlit==1.35.0
pandas==2.3.2
openai==1.25.0
pytest==8.2.1
Pin versions; add python-dotenv if you like local .env files.


9. Final Sanity Pass
	•	Run with a 50 k-row synthetic file; confirm total latency <3 s and memory <512 MB.
	•	Lint with ruff or black (optional).
	•	Push to GitHub and hand repo URL to the coding assistant.

