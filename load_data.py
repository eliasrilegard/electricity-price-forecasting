import pandas as pd

def load_csv_day_ahead_prices(region: int, year: int) -> pd.DataFrame:
  df = pd.read_csv(f"~/Downloads/kex_data/day_ahead_prices/dap-se{region}-{year}.csv")
  df.index = pd.to_datetime(
    df["MTU (CET/CEST)"].str.split(" - ").str[0],
    format="%d.%m.%Y %H:%M"
  )
  df.drop(
    columns=["MTU (CET/CEST)", f"BZN|SE{region}", "Currency"],
    inplace=True
  )
  return df

def load_csv_total_load_forecast(region: int, year: int) -> pd.DataFrame:
  df = pd.read_csv(f"~/Downloads/kex_data/total_load_forecast/tl-se{region}-{year}.csv")
  df.index = pd.to_datetime(
    df["Time (CET/CEST)"].str.split(" - ").str[0],
    format="%d.%m.%Y %H:%M"
  )
  df.drop(
    columns=["Time (CET/CEST)", f"Actual Total Load [MW] - BZN|SE{region}"],
    inplace=True
  )
  df.rename(
    columns={ f"Day-ahead Total Load Forecast [MW] - BZN|SE{region}": "Day-ahead Total Load Forecast [MW]" },
    inplace=True
  )
  return df


def day_ahead_prices(region: int = 3, start: int = 2015, end: int = 2023) -> pd.DataFrame:
  df = pd.concat([load_csv_day_ahead_prices(region, year) for year in range(start, end + 1)])
  df = df.ffill()
  df = df[~df.index.duplicated(keep="first")]
  return df

def total_load_forecast(region: int = 3, start: int = 2015, end: int = 2023) -> pd.DataFrame:
  df = pd.concat([load_csv_total_load_forecast(region, year) for year in range(start, end + 1)])
  df = df.ffill()
  df = df[~df.index.duplicated(keep="first")]
  return df