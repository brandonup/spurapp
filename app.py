# Spurcycle Sales-Insight App
# v0

import streamlit as st
import pandas as pd
import numpy as np
import openai
import time
import os
import httpx # Added for explicit HTTP client control
from io import StringIO
from dotenv import load_dotenv
from datetime import date, timedelta

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY") # Not strictly needed if client is initialized with key

# --- Configuration ---
EXPECTED_HEADERS = {
    "Name", "Company Name", "Subtotal", "Shipping", "Taxes", "Total",
    "Discount Code", "Discount Amount", "Created at", "Shipping Name",
    "Shipping Street", "Shipping Address1", "Shipping Address2",
    "Shipping Company", "Shipping City", "Shipping Zip", "Shipping Province",
    "Shipping Country", "Shipping Phone", "Tags", "Source"
}

# --- Helper Functions ---
@st.cache_data # Reverted to cache_data, more suitable for file processing
def load_and_validate_csv(uploaded_file_obj): # Renamed parameter
    """Loads, validates, and preprocesses the CSV file."""
    # Using cache_data for functions that process file-like objects or return dataframes.
    # Streamlit handles hashing of UploadedFile objects for cache_data.
    if uploaded_file_obj is None:
        return None, None, None

    start_time = time.time()
    
    try:
        # Decode to StringIO and feed to pd.read_csv
        stringio = StringIO(uploaded_file_obj.getvalue().decode("utf-8")) # Use renamed parameter
        df = pd.read_csv(stringio)
        
        # 1. Header Validation
        actual_headers = set(df.columns)
        if not EXPECTED_HEADERS.issubset(actual_headers): # Check if all expected headers are present
            missing_headers = EXPECTED_HEADERS - actual_headers
            incorrect_headers = actual_headers - EXPECTED_HEADERS # Headers in file but not expected (if strict)
            # For now, only error on missing expected headers
            if missing_headers:
                st.error(f"Missing/incorrect header(s): {', '.join(sorted(list(missing_headers)))} – please fix the CSV and re-upload.")
                return None, None, None
        
        # 2. Drop rows where Company Name is "REI" (case-insensitive)
        if "Company Name" in df.columns:
            df = df[df["Company Name"].str.lower() != "rei"]
        else:
            # This case should ideally be caught by header validation if "Company Name" is expected
            st.warning("Column 'Company Name' not found, cannot filter out 'REI'.")

        # 3. Parse Created at
        if "Created at" in df.columns:
            try:
                # Attempt to parse, store original index for error reporting
                df['original_index'] = df.index
                
                # Try to infer format first, then try specific formats if it fails broadly
                # Common date formats to try
                common_formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%m/%d/%Y %H:%M:%S",
                    "%Y-%m-%d",
                    "%m/%d/%Y",
                    "%Y-%m-%dT%H:%M:%S", 
                    "%Y-%m-%dT%H:%M:%S%z", # ISO 8601 with timezone
                    "%a, %d %b %Y %H:%M:%S %Z" # RFC 822
                ]
                
                parsed_dates = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
                temp_created_at = df["Created at"].astype(str) # Ensure it's string

                # Attempt with automatic parsing first
                try:
                    parsed_dates_auto = pd.to_datetime(temp_created_at, errors='coerce')
                except Exception: # Broad exception if to_datetime itself fails on the whole series
                    parsed_dates_auto = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')

                # For rows that failed auto parsing, try specific formats
                # This iterative approach can be slow for very large datasets with many failed rows
                # but is more robust for varied formats.
                # A more performant way for known varied formats would be to apply pd.to_datetime
                # with a specific format in a loop, only on the subset of rows that haven't been parsed yet.
                
                # For simplicity and given the error message implies a block of similar errors,
                # let's try a more direct approach: apply to_datetime and if it fails with the initial
                # infer, it will show the error as before. The user warning suggests it's already trying.
                # The key is that `errors='coerce'` should turn unparseable dates into NaT.
                
                parsed_dates = pd.to_datetime(df["Created at"], errors='coerce')
                # The UserWarning "/Users/brandonupchuch/Projects/spurcycleapp/app.py:65: UserWarning: Could not infer format..."
                # indicates that pandas is already trying its best. The issue is likely truly unparseable data
                # or a format it cannot guess.
                # No change to parsing logic here, as the current `errors='coerce'` is the correct behavior
                # for the requirement ("If any row fails, abort import and list the bad line numbers").

                bad_rows_indices = df[parsed_dates.isna()]['original_index'].tolist()
                
                if bad_rows_indices:
                    # Convert to 1-based line numbers (header is line 1, first data row is line 2)
                    bad_line_numbers = [idx + 2 for idx in bad_rows_indices]
                    st.error(f"Failed to parse 'Created at' for the following line number(s) in your CSV: {', '.join(map(str, bad_line_numbers))}. Please fix these rows and re-upload.")
                    return None, None, None
                df["Created at"] = parsed_dates
                df.drop(columns=['original_index'], inplace=True)
            except Exception as e:
                st.error(f"An unexpected error occurred during 'Created at' parsing: {e}")
                return None, None, None
        else:
            # This case should be caught by header validation if "Created at" is in EXPECTED_HEADERS
            st.warning("Column 'Created at' not found, cannot parse dates.")

        # If all checks pass:
        processing_time = time.time() - start_time
        return df, len(df), processing_time

    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return None, None, None

@st.cache_data(ttl=3600)
def prepare_customer_data(_df_orders):
    """Prepares the customer-level aggregated DataFrame."""
    if _df_orders is None or _df_orders.empty:
        return None

    df = _df_orders.copy()

    # 1. Derive customer_name
    # Always use "Company Name" as the customer key.
    if "Company Name" in df.columns:
        df["customer_name"] = df["Company Name"]
    else:
        # This should ideally be caught by header validation if "Company Name" is expected.
        # If not, we need a fallback or raise an error.
        # For now, if "Company Name" is missing (which it shouldn't be if headers are validated),
        # we'll create an 'Unknown Customer' or similar, or let it fail if groupby fails.
        # Given EXPECTED_HEADERS includes "Company Name", this path is unlikely.
        st.error("Critical Error: 'Company Name' column is missing, cannot derive customer_name.")
        return None # Or handle by creating a placeholder customer_name if appropriate
    
    # Ensure 'Total' is numeric for aggregation
    if 'Total' in df.columns:
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)
    else:
        st.error("Critical Error: 'Total' column is missing for customer data preparation.")
        return None

    # Ensure 'Created at' is datetime
    if 'Created at' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Created at']):
        pass # Correctly typed
    else:
        st.error("Critical Error: 'Created at' column is not correctly parsed as datetime for customer data preparation.")
        return None

    # Get last order details for each customer to fetch last_order_amount
    # Sort by 'Created at' to ensure 'last()' picks the actual last order details
    last_order_details = df.sort_values(by="Created at", ascending=True).groupby("customer_name").last()

    # 2. Group by customer_name and aggregate
    customer_agg = df.groupby("customer_name").agg(
        order_count=("Name", "size"),  # Assuming "Name" is a unique order identifier
        lifetime_total=("Total", "sum"),
        first_order_date=("Created at", "min"),
        last_order_date=("Created at", "max")
    ).reset_index()

    # Add last_order_amount from the last_order_details
    customer_agg = pd.merge(customer_agg, last_order_details[['Total']].rename(columns={'Total': 'last_order_amount'}), on="customer_name", how="left")

    # 3. Compute avg_days_between
    # (last_order_date - first_order_date).dt.days / (order_count - 1)
    customer_agg["avg_days_between"] = np.where(
        customer_agg["order_count"] > 1,
        (customer_agg["last_order_date"] - customer_agg["first_order_date"]).dt.days / (customer_agg["order_count"] - 1),
        pd.NA # Use pd.NA for missing numeric data if appropriate, or np.nan
    )
    # As per dev plan: round to one decimal. Requirement for Report D says "rounded avg_days_between"
    customer_agg["avg_days_between"] = customer_agg["avg_days_between"].round(1)
    
    return customer_agg

# --- Report Builder Functions ---

def get_churn_risk_one_timers(df_customers):
    """Report A: Churn-Risk One-Timers"""
    if df_customers is None or df_customers.empty:
        return pd.DataFrame(columns=["customer_name", "last_order_date", "last_order_amount"])
    
    today = pd.to_datetime(date.today())
    report_df = df_customers[
        (df_customers["order_count"] == 1) &
        (df_customers["last_order_date"] < today - timedelta(days=180))
    ].copy()
    report_df = report_df[["customer_name", "last_order_date", "last_order_amount"]]
    return report_df.sort_values(by="last_order_date", ascending=True)

def get_dormant_multi_buyers(df_customers):
    """Report B: Dormant Multi-Buyers"""
    if df_customers is None or df_customers.empty:
        return pd.DataFrame(columns=["customer_name", "last_order_date", "last_order_amount"])
        
    today = pd.to_datetime(date.today())
    report_df = df_customers[
        (df_customers["order_count"] >= 3) &
        (df_customers["last_order_date"] < today - timedelta(days=90))
    ].copy()
    report_df = report_df[["customer_name", "last_order_date", "last_order_amount"]]
    return report_df.sort_values(by="last_order_date", ascending=True)

def get_active_repeat_buyer_cadence(df_customers):
    """Report D: Active Repeat-Buyer Cadence"""
    if df_customers is None or df_customers.empty:
        return pd.DataFrame(columns=["customer_name", "order_count", "avg_days_between"])
        
    report_df = df_customers[df_customers["order_count"] > 2].copy()
    
    # Handle potential infinity values from division by zero if order_count - 1 was 0 (though filtered by order_count > 2)
    # or if dates were identical leading to 0 / 0 = NaN, which round() handles.
    # The main concern for 'object' dtype would be if NaNs were strings or other objects.
    # However, avg_days_between should be numeric or pd.NA.
    # Let's ensure it's numeric before rounding and converting.
    report_df["avg_days_between"] = pd.to_numeric(report_df["avg_days_between"], errors='coerce')
    
    # Replace any infinities with pd.NA before rounding and converting
    report_df["avg_days_between"] = report_df["avg_days_between"].replace([np.inf, -np.inf], pd.NA)
    
    # Requirement: "rounded avg_days_between" - dev plan says round to int for readability
    # Round to 0 decimal places. Result will be float (e.g., 30.0) or pd.NA.
    # Then convert to Int64 (nullable integer).
    report_df["avg_days_between"] = report_df["avg_days_between"].round(0).astype('Int64')
    
    report_df = report_df[["customer_name", "order_count", "avg_days_between"]]
    return report_df.sort_values(by="avg_days_between", ascending=False) # Slowest cadence first

def get_high_value_patterns_llm(df_customers):
    """Report C: High-Value Patterns (LLM)"""
    if df_customers is None or df_customers.empty:
        return "LLM summary unavailable.", pd.DataFrame(columns=["customer_name", "lifetime_total", "order_count", "first_order_date", "last_order_date"])

    retrieved_api_key = os.getenv("OPENAI_API_KEY")
    # print(f"[DEBUG] OPENAI_API_KEY in get_high_value_patterns_llm: {retrieved_api_key}") # Debug print removed

    if not retrieved_api_key: # Check the retrieved key
        st.warning("OpenAI API key not configured. LLM summary will be unavailable.")
        return "LLM summary unavailable (API key missing).", pd.DataFrame()

    top_20_customers = df_customers.nlargest(20, "lifetime_total")
    
    # Select relevant columns for the LLM prompt
    # The prompt itself is not provided, so we'll send key customer data.
    # Requirements: "Pass their JSON rows to OpenAI using the prompt provided earlier."
    # For now, let's select a few key fields.
    llm_data = top_20_customers[["customer_name", "lifetime_total", "order_count", "first_order_date", "last_order_date"]].copy()
    llm_data["first_order_date"] = llm_data["first_order_date"].dt.strftime('%Y-%m-%d')
    llm_data["last_order_date"] = llm_data["last_order_date"].dt.strftime('%Y-%m-%d')
    
    json_rows = llm_data.to_json(orient="records", indent=2)

    # Placeholder for the actual prompt
    prompt_template = (
        "You are a sesoned marketing anlaysit."
        "Analyze the following top 20 customer data and provide an executive summary of high-value patterns. "
        "Focus on commonalities, purchasing behaviors, and any insights that could inform marketing or retention strategies.\n\n"
        "Customer Data (JSON):\n{json_data}\n\nExecutive Summary:"
    )
    full_prompt = prompt_template.format(json_data=json_rows)

    summary = "LLM summary unavailable."
    
    # Explicitly create an httpx client with trust_env=False
    # to prevent it from picking up system proxy environment variables.
    # Streamlit is async, so an AsyncClient is appropriate if the OpenAI client can use it.
    # The OpenAI client can accept a pre-configured httpx.Client or httpx.AsyncClient.
    # Let's use a synchronous client for simplicity unless async is strictly required by OpenAI's sync client.
    # The default OpenAI() client uses a synchronous httpx.Client.
    
    custom_http_client = httpx.Client(trust_env=False)

    client = openai.OpenAI(
        api_key=retrieved_api_key, # Use the key checked above
        http_client=custom_http_client
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06", 
            messages=[
                {"role": "system", "content": "You are an expert data analyst providing executive summaries."},
                {"role": "user", "content": full_prompt}
            ],
            timeout=10,
        )
        summary = response.choices[0].message.content.strip()
    except openai.APIConnectionError as e: 
        st.warning(f"LLM call failed (connection error), retrying once: {e}")
        try:
            # Retry with the same explicitly initialized client
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an expert data analyst."},
                    {"role": "user", "content": full_prompt}
                ],
                timeout=15 
            )
            summary = response.choices[0].message.content.strip()
        except Exception as retry_e:
            st.error(f"LLM call failed on retry: {retry_e}")
            summary = "LLM summary unavailable (failed on retry)."
    except Exception as e:
        st.error(f"LLM call failed: {e}") # This will catch the proxies error if it persists
        summary = f"LLM summary unavailable ({type(e).__name__} - {e})." # Add error message
        
    return summary, llm_data

# --- UX Helper Functions ---
def display_report_table(df_report, report_name):
    """Helper to display a report table or 'n/a' message with currency formatting."""
    if df_report.empty:
        st.caption("n/a")
    else:
        column_config = {
            "lifetime_total": st.column_config.NumberColumn(format="$%.2f"),
            "last_order_amount": st.column_config.NumberColumn(format="$%.2f"),
            "Total": st.column_config.NumberColumn(format="$%.2f"), # If 'Total' appears
            "avg_days_between": st.column_config.NumberColumn(format="%.0f days"), # Rounded for Report D
            "last_order_date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "first_order_date": st.column_config.DateColumn(format="YYYY-MM-DD"),
        }
        # Filter config for columns present in the specific report
        report_specific_column_config = {
            k: v for k, v in column_config.items() if k in df_report.columns
        }

        st.dataframe(df_report, use_container_width=True, column_config=report_specific_column_config)
        
        csv = df_report.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {report_name} CSV",
            data=csv,
            file_name=f"{report_name.lower().replace(' ', '_')}.csv",
            mime='text/csv',
        )

# --- Main App Logic ---
def main():
    st.set_page_config(layout="wide") 
    st.title("Spurcycle Sales-Insight App")

    # Initialize session state for data if not present
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = None
    if 'df_customers' not in st.session_state:
        st.session_state.df_customers = None
    if 'num_raw_orders' not in st.session_state:
        st.session_state.num_raw_orders = 0
    if 'parsing_time' not in st.session_state:
        st.session_state.parsing_time = 0


    uploaded_file = st.file_uploader(
        "Upload your sales CSV file (max 20MB)", 
        type=["csv"], 
        accept_multiple_files=False
    )

    if uploaded_file:
        # Check file size (Streamlit's default max is 200MB, but requirement is 20MB)
        if uploaded_file.size > 20 * 1024 * 1024: # 20 MB in bytes
            st.error("File size exceeds 20MB limit. Please upload a smaller file.")
            st.stop()
            
        # Use a spinner while parsing and preparing data
        with st.spinner("Processing your CSV file... This may take a moment."):
            df_processed_loaded, num_orders_loaded, proc_time_loaded = load_and_validate_csv(uploaded_file)

        if df_processed_loaded is not None:
            st.session_state.df_processed = df_processed_loaded
            st.session_state.num_raw_orders = num_orders_loaded
            st.session_state.parsing_time = proc_time_loaded
            
            with st.spinner("Preparing customer analytics..."):
                st.session_state.df_customers = prepare_customer_data(st.session_state.df_processed)
            
            if st.session_state.df_customers is not None:
                 st.success(f"Successfully processed your file.") # Simpler success message
            else:
                st.error("Could not prepare customer data after parsing. Please check CSV content.")
                # Clear data if customer prep fails
                st.session_state.df_processed = None
                st.session_state.df_customers = None
        else:
            # Error messages are shown by load_and_validate_csv
            # Clear data if loading fails
            st.session_state.df_processed = None
            st.session_state.df_customers = None


    if st.session_state.df_customers is not None:
        # Display status line (Requirement 7)
        # "Parsed 12,483 orders for 8,211 customers in 1.8 s.”
        status_message = (
            f"Parsed {st.session_state.num_raw_orders} order lines "
            f"for {len(st.session_state.df_customers)} customers "
            f"in {st.session_state.parsing_time:.1f}s."
        )
        st.caption(status_message) # Using st.caption for a less prominent status line

        # --- Display Reports ---
        # Dev plan: "Each report lives in its own st.expander"
        # Dev plan: "Wrap each report render in st.spinner so all four appear within the 3s target"
        # The 3s target is for all reports combined after initial parsing.
        # Individual spinners might be too much if data is already computed.
        # Let's compute all report data first, then display.

        with st.spinner("Generating reports..."):
            report_a_data = get_churn_risk_one_timers(st.session_state.df_customers)
            report_b_data = get_dormant_multi_buyers(st.session_state.df_customers)
            llm_summary, report_c_data = get_high_value_patterns_llm(st.session_state.df_customers)
            report_d_data = get_active_repeat_buyer_cadence(st.session_state.df_customers)

        # Layout for reports (Dev plan: "four reports sit comfortably side-by-side")
        # This is hard with expanders. Let's use columns for titles and then expanders within.
        # Or simply list them vertically. Side-by-side might be too cramped with expanders.
        # For now, vertical layout.

        with st.expander("Report A: Churn-Risk One-Timers", expanded=True):
            display_report_table(report_a_data, "Churn-Risk One-Timers")

        with st.expander("Report B: Dormant Multi-Buyers", expanded=True):
            display_report_table(report_b_data, "Dormant Multi-Buyers")

        with st.expander("Report C: High-Value Patterns (LLM)", expanded=True):
            st.markdown("#### LLM Executive Summary")
            st.markdown(llm_summary if llm_summary else "Summary not available.")
            st.markdown("---")
            st.markdown("#### Top 20 Customers Data (used for LLM)")
            display_report_table(report_c_data, "High-Value Patterns Data")
            
        with st.expander("Report D: Active Repeat-Buyer Cadence", expanded=True):
            display_report_table(report_d_data, "Active Repeat-Buyer Cadence")
            
    elif uploaded_file is None:
        st.info("Upload a CSV file to begin analysis.")
    # If uploaded_file is not None but df_customers is None, errors were already shown.


if __name__ == "__main__":
    main()
