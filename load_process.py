import yfinance as yf
import pandas as pd

def load_and_process_nvda_data(start_date, end_date, csv_filename):
    """
    This function downloads NVDA stock data from Yahoo Finance and stores it in a CSV file.

    Parameters:
    - start_date: The start date for downloading the stock data (e.g., '2019-01-01').
    - end_date: The end date for downloading the stock data (e.g., '2024-12-03').
    - csv_filename: The name of the CSV file to save the data (e.g., 'nvda_stock_data.csv').
    """
    try:
        # Download the stock data for NVDA from Yahoo Finance
        df = yf.download("NVDA", start=start_date, end=end_date)

        # Check if data was fetched
        if df.empty:
            print("No data returned for NVDA. Please check the date range or symbol.")
            return

        # Print the first few rows to verify data structure
        print("Fetched Data Preview:")
        print(df.head())

        # Flatten the multi-level columns by keeping only the second level (Open, Close, etc.)
        # We will select the correct columns from the second level of the multi-index
        df.columns = df.columns.droplevel(1)  # Drop the first level (Ticker)

        # Now the columns are 'Open', 'Close', etc.
        print("\nColumn Names After Dropping Multi-level Index:")
        print(df.columns)

        # Filter only 'Open' and 'Close' columns
        df_filtered = df[['Open', 'Close']]  # Select Open and Close columns

        # Remove rows with missing values in 'Open' or 'Close'
        df_filtered.dropna(subset=['Open', 'Close'], inplace=True)

        # Save the data to a CSV file with the correct index (Date)
        df_filtered.index.name = "Date"  # Label the index as 'Date'
        df_filtered.to_csv(csv_filename, date_format="%Y-%m-%d")

        print(f"Data for NVDA successfully saved to {csv_filename}")

    except Exception as e:
        print(f"Error occurred: {e}")

# Example usage
load_and_process_nvda_data(start_date="2019-01-01", end_date="2024-12-03", csv_filename="nvda_stock_data.csv")
