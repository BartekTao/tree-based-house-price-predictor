import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

# 載入訓練和測試資料集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 選擇要使用的欄位
features = ['縣市', '土地面積', '移轉層次', '總樓層數', '主要用途', '主要建材', '建物型態', '屋齡', '建物面積', '車位面積', '車位個數', '橫坐標', '縱坐標', '主建物面積', '陽台面積', '附屬建物面積']
X_train = train_data[features]
X_test = test_data[features]
y_train = train_data['單價']

# 確定非數值型和數值型特徵
categorical_features = ['縣市', '主要用途', '主要建材', '建物型態']
numeric_features = [col for col in features if col not in categorical_features]

# 數值特徵處理
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 類別特徵處理
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 組合預處理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 定義最終管道
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
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

    output.to_csv('test_predictions_onehot_2.csv', index=False)

    print("預測結果已寫入 test_predictions.csv")
except NotFittedError as e:
    print(f"模型訓練失敗: {e}")
except Exception as e:
    print(f"發生錯誤: {e}")
