import yfinance as yf
import pandas as pd

# 股票名：
# 1.AAPL:apple 2.NVDA:NVIDIA 3.JPM:摩根大通(金融) 4.XOM:ExxonMobil(傳統油氣) 5.DIS:迪士尼
#stocks = {"AAPL", "NVDA", "JPM", "XOM", "DIS"}
stocks = {"JPM"}
count = 1
start_time = "2026-01-01"
end_time = "2026-04-30"
save_path = "dataset/validTest"

for stock in stocks:
    print(f"{count}.{stock}")
    count+=1

    # prepost 是否顯示盤前盤後資料
    data = yf.download(stock, start=start_time, end=end_time, interval="1h", prepost=True)
    data_wo_prepost = yf.download(stock, start=start_time, end=end_time, interval="1h", prepost=False)

    # 刪掉標頭的Ticker 行
    data.columns = data.columns.droplevel('Ticker')
    data_wo_prepost.columns = data_wo_prepost.columns.droplevel('Ticker')

    # 存成csv，一般是包含盤前盤後資料、wo_prepost 是沒有盤前盤後資料
    data.to_csv(f"{save_path}/{stock}.csv", float_format="%.2f")
    data_wo_prepost.to_csv(f"{save_path}/{stock}_wo_prepost.csv", float_format="%.2f")

#print(data)

