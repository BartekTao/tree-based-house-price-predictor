import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# 載入訓練和測試資料集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 創建映射
city_mapping = {category: code for code, category in enumerate(train_data['縣市'].astype('category').cat.categories, 1)}
use_mapping = {category: code for code, category in enumerate(train_data['主要用途'].astype('category').cat.categories, 1)}
material_mapping = {category: code for code, category in enumerate(train_data['主要建材'].astype('category').cat.categories, 1)}
building_type_mapping = {category: code for code, category in enumerate(train_data['建物型態'].astype('category').cat.categories, 1)}

# 將映射應用於訓練資料
train_data['縣市'] = train_data['縣市'].map(city_mapping)
train_data['主要用途'] = train_data['主要用途'].map(use_mapping)
train_data['主要建材'] = train_data['主要建材'].map(material_mapping)
train_data['建物型態'] = train_data['建物型態'].map(building_type_mapping)

# 將映射應用於測試資料
test_data['縣市'] = test_data['縣市'].map(city_mapping)
test_data['主要用途'] = test_data['主要用途'].map(use_mapping)
test_data['主要建材'] = test_data['主要建材'].map(material_mapping)
test_data['建物型態'] = test_data['建物型態'].map(building_type_mapping)

# 打印映射
print("縣市 mapping:", city_mapping)
print("主要用途 mapping:", use_mapping)
print("主要建材 mapping:", material_mapping)
print("建物型態 mapping:", building_type_mapping)

# 分離特徵和目標變數
X_train = train_data.drop('單價', axis=1)
y_train = train_data['單價']

# 確保所有特徵都是數值型資料
X_train = X_train.select_dtypes(include=[np.number])
X_test = test_data.select_dtypes(include=[np.number])

# 定義管道以預處理資料並建立模型
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # 處理缺失值
    ('scaler', StandardScaler()),  # 特徵縮放
    ('regressor', DecisionTreeRegressor())  # 基於樹的回歸模型
])

try:
    # 交叉驗證
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error', error_score='raise')  # 使用均方誤差作為評估指標

    # 輸出交叉驗證結果
    print("交叉驗證均方誤差:", -cv_scores)
    print("平均交叉驗證均方誤差:", -cv_scores.mean())

    # 訓練最終模型
    pipeline.fit(X_train, y_train)

    # 對測試資料進行預測
    test_predictions = pipeline.predict(X_test)

    # 將預測結果寫入 csv 檔案
    output = pd.DataFrame({'ID': test_data["ID"], 'predicted_price': test_predictions})
    output.to_csv('test_predictions.csv', index=False)

    print("預測結果已寫入 /mnt/data/test_predictions.csv")
except NotFittedError as e:
    print(f"模型訓練失敗: {e}")
except Exception as e:
    print(f"發生錯誤: {e}")
