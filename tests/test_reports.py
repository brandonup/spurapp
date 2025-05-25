import pytest
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
import app # Import the Streamlit app module

# --- Fixtures ---

@pytest.fixture
def all_expected_headers_string():
    # Must match EXPECTED_HEADERS in app.py
    headers = [
        "Name", "Company Name", "Subtotal", "Shipping", "Taxes", "Total",
        "Discount Code", "Discount Amount", "Created at", "Shipping Name",
        "Shipping Street", "Shipping Address1", "Shipping Address2",
        "Shipping Company", "Shipping City", "Shipping Zip", "Shipping Province",
        "Shipping Country", "Shipping Phone", "Tags", "Source"
    ]
    return ",".join(headers)

@pytest.fixture
def sample_csv_data_valid(all_expected_headers_string):
    """Provides a string representation of a valid sample CSV."""
    return f"""{all_expected_headers_string}
Order1,CompA,100,10,5,115,DISC1,5,2023-01-01,ShipNameA,StreetA,Addr1A,Addr2A,ShipCompA,CityA,ZipA,ProvA,CountryA,PhoneA,TagA,SourceA
Order2,CompB,200,20,10,230,,,2023-02-01,ShipNameB,StreetB,Addr1B,Addr2B,ShipCompB,CityB,ZipB,ProvB,CountryB,PhoneB,TagB,SourceB
Order3,REI,300,30,15,345,,,2023-03-01,ShipNameC,StreetC,Addr1C,Addr2C,ShipCompC,CityC,ZipC,ProvC,CountryC,PhoneC,TagC,SourceC
Order4,CompA,150,15,7,172,,,2024-01-15,ShipNameA,StreetA,Addr1A,Addr2A,ShipCompA,CityA,ZipA,ProvA,CountryA,PhoneA,TagA,SourceA
Order5,CompD,50,5,2,57,,,2022-05-01,ShipNameD,StreetD,Addr1D,Addr2D,ShipCompD,CityD,ZipD,ProvD,CountryD,PhoneD,TagD,SourceD
Order6,CompE,70,7,3,80,,,2023-10-01,ShipNameE,StreetE,Addr1E,Addr2E,ShipCompE,CityE,ZipE,ProvE,CountryE,PhoneE,TagE,SourceE
Order7,CompE,80,8,4,92,,,2023-11-01,ShipNameE,StreetE,Addr1E,Addr2E,ShipCompE,CityE,ZipE,ProvE,CountryE,PhoneE,TagE,SourceE
Order8,CompE,90,9,5,104,,,2023-12-01,ShipNameE,StreetE,Addr1E,Addr2E,ShipCompE,CityE,ZipE,ProvE,CountryE,PhoneE,TagE,SourceE
Order9,CompF,10,1,0.5,11.5,,,2024-03-01,ShipNameF,StreetF,Addr1F,Addr2F,ShipCompF,CityF,ZipF,ProvF,CountryF,PhoneF,TagF,SourceF
Order10,CompF,20,2,1,23,,,2024-03-15,ShipNameF,StreetF,Addr1F,Addr2F,ShipCompF,CityF,ZipF,ProvF,CountryF,PhoneF,TagF,SourceF
"""

@pytest.fixture
def sample_csv_data_missing_header(all_expected_headers_string):
    """CSV with 'Total' header missing."""
    headers_list = all_expected_headers_string.split(',')
    headers_list.remove("Total")
    return f"""{",".join(headers_list)}
Order1,CompA,100,10,5,DISC1,5,2023-01-01,ShipNameA,StreetA,Addr1A,Addr2A,ShipCompA,CityA,ZipA,ProvA,CountryA,PhoneA,TagA,SourceA
"""

@pytest.fixture
def sample_csv_data_bad_date_format(all_expected_headers_string):
    """CSV with a badly formatted date."""
    return f"""{all_expected_headers_string}
Order1,CompA,100,10,5,115,DISC1,5,NOT_A_DATE,ShipNameA,StreetA,Addr1A,Addr2A,ShipCompA,CityA,ZipA,ProvA,CountryA,PhoneA,TagA,SourceA
"""

# Mock UploadedFile object
class MockUploadedFile:
    def __init__(self, content_string):
        self.content_string = content_string
        self.name = "test.csv"
        self.size = len(content_string.encode('utf-8'))

    def getvalue(self):
        return self.content_string.encode('utf-8')

# --- Test Cases for load_and_validate_csv ---

def test_load_valid_csv(sample_csv_data_valid):
    mock_file = MockUploadedFile(sample_csv_data_valid)
    df, num_orders, proc_time = app.load_and_validate_csv(mock_file)
    assert df is not None
    assert num_orders == 9 # 10 original rows, 1 REI row filtered out
    assert "REI" not in df["Company Name"].str.upper().tolist()
    assert pd.api.types.is_datetime64_any_dtype(df["Created at"])

def test_load_csv_missing_header(sample_csv_data_missing_header):
    mock_file = MockUploadedFile(sample_csv_data_missing_header)
    # This test needs to check st.error, which is tricky without Streamlit context.
    # For now, we check the direct return. In a real Streamlit test env, you'd mock st.error.
    df, num_orders, proc_time = app.load_and_validate_csv(mock_file)
    assert df is None # Expecting failure due to missing header

def test_load_csv_bad_date(sample_csv_data_bad_date_format):
    mock_file = MockUploadedFile(sample_csv_data_bad_date_format)
    df, num_orders, proc_time = app.load_and_validate_csv(mock_file)
    assert df is None # Expecting failure due to bad date

# --- Test Cases for prepare_customer_data ---

@pytest.fixture
def sample_processed_df(sample_csv_data_valid):
    # Get a valid processed DataFrame first
    mock_file = MockUploadedFile(sample_csv_data_valid)
    df, _, _ = app.load_and_validate_csv(mock_file)
    return df

def test_prepare_customer_data_structure(sample_processed_df):
    customer_df = app.prepare_customer_data(sample_processed_df)
    assert customer_df is not None
    expected_cols = {"customer_name", "order_count", "lifetime_total", 
                     "first_order_date", "last_order_date", "last_order_amount", "avg_days_between"}
    assert expected_cols.issubset(set(customer_df.columns))
    
    # Test CompA data (ShipNameA)
    comp_a_data = customer_df[customer_df["customer_name"] == "ShipNameA"]
    assert not comp_a_data.empty
    assert comp_a_data["order_count"].iloc[0] == 2
    assert comp_a_data["lifetime_total"].iloc[0] == 115 + 172
    assert comp_a_data["last_order_amount"].iloc[0] == 172 # Last order for CompA
    assert pd.to_datetime(comp_a_data["first_order_date"].iloc[0]) == datetime(2023,1,1)
    assert pd.to_datetime(comp_a_data["last_order_date"].iloc[0]) == datetime(2024,1,15)
    days_diff = (datetime(2024,1,15) - datetime(2023,1,1)).days
    assert comp_a_data["avg_days_between"].iloc[0] == days_diff / (2-1)


# --- Test Cases for Report Filters ---
# These tests will use the customer_df generated by prepare_customer_data

@pytest.fixture
def sample_customer_df(sample_processed_df):
    return app.prepare_customer_data(sample_processed_df)

def test_report_a_churn_risk_one_timers(sample_customer_df):
    # ShipNameD: 1 order, 2022-05-01 (older than 180 days from a fixed 'today')
    # We need to mock 'date.today()' or pass it into the function for consistent testing
    # For simplicity, let's assume today is 2024-06-01 for this test.
    # Monkeypatching date.today can be complex. Alternative: modify function to accept 'today'.
    # For now, let's manually check based on the data.
    # CompD (ShipNameD) made one order on 2022-05-01. This should be a churn risk.
    # CompB (ShipNameB) made one order on 2023-02-01. If today is 2024-06-01, this is > 180 days.
    
    # To make this test robust, we'd ideally inject 'today'
    # For now, let's assume the logic is correct and check if it produces *some* output
    # or filter based on known old dates in the fixture.
    
    # Modify CompD's last_order_date to be very old for a clear test
    idx_comp_d = sample_customer_df[sample_customer_df['customer_name'] == 'ShipNameD'].index
    if not idx_comp_d.empty:
         sample_customer_df.loc[idx_comp_d, 'last_order_date'] = pd.to_datetime('2020-01-01')
         sample_customer_df.loc[idx_comp_d, 'order_count'] = 1
         sample_customer_df.loc[idx_comp_d, 'last_order_amount'] = 57

    report_a = app.get_churn_risk_one_timers(sample_customer_df)
    assert "ShipNameD" in report_a["customer_name"].tolist()
    assert report_a[report_a["customer_name"] == "ShipNameD"]["last_order_amount"].iloc[0] == 57

def test_report_b_dormant_multi_buyers(sample_customer_df):
    # CompE: 3 orders, last on 2023-12-01. If today is 2024-06-01, this is < 90 days from last order.
    # Let's make CompE dormant for testing.
    idx_comp_e = sample_customer_df[sample_customer_df['customer_name'] == 'ShipNameE'].index
    if not idx_comp_e.empty:
        sample_customer_df.loc[idx_comp_e, 'last_order_date'] = pd.to_datetime('2023-06-01') # Make it > 90 days from a hypothetical 2024-06-01
        sample_customer_df.loc[idx_comp_e, 'order_count'] = 3
        sample_customer_df.loc[idx_comp_e, 'last_order_amount'] = 104 # last order amount for CompE

    report_b = app.get_dormant_multi_buyers(sample_customer_df)
    assert "ShipNameE" in report_b["customer_name"].tolist()
    assert report_b[report_b["customer_name"] == "ShipNameE"]["last_order_amount"].iloc[0] == 104


def test_report_d_active_repeat_buyer_cadence(sample_customer_df):
    # CompE has 3 orders. Should appear.
    # CompA has 2 orders. Should NOT appear (needs > 2 orders).
    # CompF has 2 orders. Should NOT appear.
    report_d = app.get_active_repeat_buyer_cadence(sample_customer_df)
    assert "ShipNameE" in report_d["customer_name"].tolist()
    assert "ShipNameA" not in report_d["customer_name"].tolist()
    assert "ShipNameF" not in report_d["customer_name"].tolist()
    
    # Check cadence calculation for CompE
    # Orders: 2023-10-01, 2023-11-01, 2023-12-01
    # first: 2023-10-01, last: 2023-12-01. Diff = 61 days. order_count = 3.
    # avg_days_between = 61 / (3-1) = 30.5. Rounded to 31.
    comp_e_cadence = report_d[report_d["customer_name"] == "ShipNameE"]["avg_days_between"].iloc[0]
    assert comp_e_cadence == 31 # 30.5 rounded to 0 decimal places is 31

# Test for LLM (Report C) would require mocking openai.ChatCompletion.create
# This is more involved and might be skipped for brevity unless specifically requested.
# For now, we can test that it returns the "unavailable" message if API key is missing.

def test_report_c_high_value_llm_no_api_key(sample_customer_df, monkeypatch):
    # Temporarily remove API key for this test
    monkeypatch.setattr(app.openai, "api_key", None)
    summary, df_top_20 = app.get_high_value_patterns_llm(sample_customer_df)
    assert "LLM summary unavailable (API key missing)" in summary
    assert not df_top_20.empty # Should still return the top 20 data

# (Add more tests for edge cases, empty dataframes, etc.)
