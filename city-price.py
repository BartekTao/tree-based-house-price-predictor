import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案
file_path = 'train.csv'
df = pd.read_csv(file_path)

# 將 '縣市' 轉換為 category 並使用數字編碼
city_mapping = {category: code for code, category in enumerate(df['縣市'].astype('category').cat.categories, 1)}
df['縣市'] = df['縣市'].map(city_mapping)
print("縣市 mapping:", city_mapping)

# 2. 處理缺失值
df = df.dropna(subset=['縣市', '單價'])

# 3. 確保 '縣市' 和 '單價' 是正確的資料型態
df['單價'] = pd.to_numeric(df['單價'], errors='coerce')
df = df.dropna(subset=['單價'])

# 繪製散點圖
plt.figure(figsize=(12, 6))
plt.scatter(df['縣市'], df['單價'])
plt.title('The relationship between housing price and city')
plt.xlabel('City')
plt.ylabel('Housing Price')
plt.xticks(range(1, df['縣市'].nunique() + 1))
plt.tight_layout()
plt.show()
