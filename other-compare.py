import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取 CSV 檔案
file_path = 'train.csv'
df = pd.read_csv(file_path)

# 前處理步驟
# 去除土地面積為負的資料
df = df[df['土地面積'] >= 0]

# 處理缺失值
df = df.dropna(subset=['縣市', '主要用途', '單價', '建物型態'])

# 確保 '單價' 是數字型態
df['單價'] = pd.to_numeric(df['單價'], errors='coerce')
df = df.dropna(subset=['單價'])


# 各縣市的平均單價 - 條形圖
city_price_avg = df.groupby('縣市')['單價'].mean().reset_index()
plt.figure(figsize=(14, 8))
sns.barplot(x='縣市', y='單價', data=city_price_avg, palette='viridis')
plt.title('Average Unit Price by City')
plt.xlabel('City')
plt.ylabel('Average Unit Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 主要用途的房價分佈 - 箱形圖
plt.figure(figsize=(14, 8))
sns.boxplot(x='主要用途', y='單價', data=df, palette='Set2')
plt.title('Housing Price Distribution by Primary Use')
plt.xlabel('Primary Use')
plt.ylabel('Housing Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 建物型態的單價分佈 - 小提琴圖
plt.figure(figsize=(14, 8))
sns.violinplot(x='建物型態', y='單價', data=df, palette='Set3')
plt.title('Housing Price Distribution by Building Type')
plt.xlabel('Building Type')
plt.ylabel('Housing Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

