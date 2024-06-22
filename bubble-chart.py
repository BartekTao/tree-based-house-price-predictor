import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取 CSV 檔案
file_path = 'train.csv'
df = pd.read_csv(file_path)

# 前處理步驟
df = df.dropna(subset=['橫坐標', '縱坐標', '單價'])
df['單價'] = pd.to_numeric(df['單價'], errors='coerce')
df = df.dropna(subset=['單價'])

# 繪製氣泡圖
plt.figure(figsize=(14, 10))
sns.scatterplot(x='橫坐標', y='縱坐標', size='單價', hue='單價', data=df, alpha=0.6, edgecolor='w', sizes=(20, 200), palette='viridis')
plt.title('Bubble Chart of Housing Locations (Bubble Size: Unit Price)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Unit Price', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
