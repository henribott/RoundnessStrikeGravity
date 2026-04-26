import yfinance as yf

df = yf.download("EURUSD=X", period="1mo", interval="1d")
print(df.tail())