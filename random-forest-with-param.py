import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 載入訓練和測試資料集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

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
    ('regressor', RandomForestRegressor())  # 使用隨機森林回歸模型
])

# 定義超參數網格
param_grid = {
    'regressor__n_estimators': [100, 150, 200],
    'regressor__max_depth': [10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2']
}

# 使用 GridSearchCV 搜索最佳超參數組合
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 輸出最佳參數組合
print("最佳參數組合:", grid_search.best_params_)

# 使用最佳參數進行最終訓練
best_pipeline = grid_search.best_estimator_

# 交叉驗證
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# 輸出交叉驗證結果
print("交叉驗證均方誤差:", -cv_scores)
print("平均交叉驗證均方誤差:", -cv_scores.mean())

# 對測試資料進行預測
test_predictions = best_pipeline.predict(X_test)

# 將預測結果寫入 csv 檔案
output = pd.DataFrame({'ID': test_data["ID"], 'predicted_price': test_predictions})

output.to_csv('test_predictions3.csv', index=False)

print("預測結果已寫入 /mnt/data/test_predictions.csv")
