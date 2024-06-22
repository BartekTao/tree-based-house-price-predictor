import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案
file_path = 'train.csv'
df = pd.read_csv(file_path)

# 前處理步驟
use_mapping = {category: code for code, category in enumerate(df['主要用途'].astype('category').cat.categories, 1)}
df['主要用途'] = df['主要用途'].map(use_mapping)
print("主要用途 mapping:", use_mapping)

# 2. 處理缺失值
df = df.dropna(subset=['主要用途', '單價'])

# 3. 確保 '單價' 是數字型態
df['單價'] = pd.to_numeric(df['單價'], errors='coerce')
df = df.dropna(subset=['單價'])

# 4. 計算每個主要用途的平均單價
usage_price_avg = df.groupby('主要用途')['單價'].mean().reset_index()

# 繪製散點圖
plt.figure(figsize=(12, 6))
plt.scatter(df['主要用途'], df['單價'])
plt.title('The relationship between primary use and housing price')
plt.xlabel('Primary Use')
plt.ylabel('Housing Price')
plt.xticks(range(1, df['主要用途'].nunique() + 1))
plt.tight_layout()
plt.show()
