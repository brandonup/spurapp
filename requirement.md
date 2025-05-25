### **Spurcycle Sales-Insight App — v0 Requirements**

**1. CSV Upload**

- Accept exactly **one** file per session, size capped at 20 MB.
- The CSV must include all 20 headers with exact spelling and case.
    - If a header is missing or misspelled, show a red Streamlit error:
        
        “Missing/incorrect header: <name> – please fix the CSV and re-upload.”
        
- Drop any row where Company Name equals “REI” (case-insensitive).
- Parse Created At with pd.to_datetime. If any row fails, abort import and list the bad line numbers for the user.

**2. Common Data Prep**

- Use **Shipping Name** as the customer key; if it’s blank, fall back to **Company Name**.
- Derive:
    1. order_count
    2. lifetime_total (sum of Total)
    3. first_order_date and last_order_date
    4. avg_days_between (only if order_count > 1)

**3. Report A – Churn-Risk One-Timers**

- Criteria: order_count == 1 **and** last_order_date < today − 180 days.
- Show customer_name, last_order_date, and last_order_amount, sorted by the oldest last_order_date.

**4. Report B – Dormant Multi-Buyers**

- Criteria: order_count ≥ 3 **and** last_order_date < today − 90 days.
- Same columns and sort order as Report A.

**5. Report C – High-Value Patterns (LLM)**

1. Select the top 20 customers by lifetime_total.
2. Pass their JSON rows to OpenAI using the prompt provided earlier.
3. Display the LLM’s executive summary above the raw table.
4. If the first OpenAI call fails, retry once; otherwise show “LLM summary unavailable.”
5. It’s acceptable in v0 to send customer names to the LLM.

**6. Report D – Active Repeat-Buyer Cadence**

- Criteria: order_count > 2.
- Show customer_name, order_count, and rounded avg_days_between, sorted with the slowest cadence first.

**7. UX Touches**

- Each report has a “Download CSV” button.
- If a report returns no rows, render a single-row table that says “n/a”.
- A status line at the top reads, for example:
    
    “Parsed 12,483 orders for 8,211 customers in 1.8 s.”
    
- All four reports must render within three seconds on a 50 k-row file.

**8. Non-Functional Notes**

- Tech stack: Python 3.11, pandas, Streamlit, OpenAI API.
- No persistent storage; everything resets on page refresh.
- Store OPENAI_API_KEY as a Vercel project environment variable, and note that step in the README.

**9. Deliverables for the Coding Assistant**

1. app.py — fully commented Streamlit script.
2. tests/ — pytest cases for each report filter and key error paths.
3. README.md — instructions for local run and Vercel deploy, including the env-var setup.
4. requirements.txt (optional but recommended) with pinned package versions.