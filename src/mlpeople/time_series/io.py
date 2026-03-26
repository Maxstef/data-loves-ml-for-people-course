import pandas as pd
import numpy as np


def read_df(file_name, datetime_col="Month", drop_orignial_datetime_col=True):
    df = pd.read_csv("./data/" + file_name)

    # build datetime index
    df["Date"] = pd.to_datetime(df[datetime_col], errors="coerce")
    df.set_index("Date", inplace=True)

    # optionally drop original datetime column
    if drop_orignial_datetime_col and datetime_col != "Date":
        df.drop(columns=[datetime_col], inplace=True)

    # auto-convert columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # sort index (important for time series)
    df.sort_index(inplace=True)

    return df


def read_shampoo_df(file_name):
    df = pd.read_csv("./data/" + file_name)

    # split "1-Jan" → ["1", "Jan"]
    parts = df["Month"].str.split("-", expand=True)
    year_number = parts[0].astype(int)
    month_name = parts[1]

    # convert month name to month number
    month_number = pd.to_datetime(month_name, format="%b").dt.month

    # construct real date (use 2000 as base)
    df["Date"] = pd.to_datetime(
        dict(year=2000 + year_number, month=month_number, day=1)
    )

    df.set_index("Date", inplace=True)
    df.drop(columns=["Month"], inplace=True)

    return df
