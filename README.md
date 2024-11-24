

# Data Science HW1 - 信用卡詐騙偵測系統

## 如何執行程式

1. **安裝所需的Python套件：**

   ```
   pip install pandas numpy scikit-learn lightgbm
   ```

2. **確保有兩個CSV檔案：**

   - `train_data.csv`：訓練資料集
   - `test_data.csv`：測試資料集

3. **並執行以下命令：**

   ```
   python fraud_detection.py
   ```

## 程式架構與演算法流程

### 1. 環境檢查

`check_environment()` 函數檢查所需的Python套件版本，確保所有必要的庫已安裝並可以正常使用。

```python
def check_environment():
    print("開始環境檢查...")
    try:
        print("pandas 版本:", pd.__version__)
        print("numpy 版本:", np.__version__)
        print("lightgbm 版本:", lgb.__version__)
        print("環境檢查完成！\n")
        return True
    except Exception as e:
        print("環境檢查失敗:", str(e))
        return False
```

### 2. 讀取數據

`load_data()` 函數讀取CSV文件，將訓練和測試數據載入Pandas DataFrame中，並輸出數據集的大小。

```python
def load_data():
    print("開始讀取數據...")
    try:
        # 定義每個欄位的數據類型
        dtypes = {
            'att2': str, 'att3': str, 'att4': float, 'att5': int,
            'att6': str, 'att7': str, 'att8': str, 'att9': str,
            'att10': int, 'att11': int, 'att12': float, 'att13': float,
            'att14': str, 'att15': float, 'att16': float, 'fraud': int
        }
        train_df = pd.read_csv('train_data.csv', dtype=dtypes)
        test_df = pd.read_csv('test_data.csv', dtype=dtypes)
        print(f"訓練集形狀: {train_df.shape}")
        print(f"測試集形狀: {test_df.shape}")
        print("數據讀取完成！\n")
        return train_df, test_df
    except Exception as e:
        print("數據讀取失敗:", str(e))
        return None, None
```

### 3. 數據處理

`basic_preprocessing()` 函數對數據進行預處理，包括處理時間特徵、衍生特徵的生成、數值標準化和類別特徵編碼。

```python
def basic_preprocessing(train_df, test_df):
    print("開始數據處理...")
    try:
        # 結合訓練集和測試集進行處理
        all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        
        # 處理時間特徵
        all_data['hour_decimal'] = all_data['att1'].apply(process_time)
        all_data['hour_sin'] = np.sin(2 * np.pi * all_data['hour_decimal'] / 24)
        all_data['hour_cos'] = np.cos(2 * np.pi * all_data['hour_decimal'] / 24)
        
        # 信用卡衍生特徵
        all_data['card_first_4'] = all_data['att2'].str[:4]
        all_data['card_last_4'] = all_data['att2'].str[-4:]
        
        # 距離計算
        all_data['distance'] = np.sqrt(
            (all_data['att12'] - all_data['att15'])**2 +
            (all_data['att13'] - all_data['att16'])**2
        )
        
        # 數值標準化
        scaler = StandardScaler()
        all_data['scaled_distance'] = scaler.fit_transform(all_data[['distance']])
        all_data['scaled_amount'] = scaler.fit_transform(all_data[['att4']])
        all_data['scaled_age'] = scaler.fit_transform(all_data[['att5']])
        
        # 類別特徵編碼
        le = LabelEncoder()
        categorical_features = ['att3', 'att6', 'att7', 'att8', 'att9', 'card_first_4', 'card_last_4', 'att14']
        for feature in categorical_features:
            all_data[feature] = le.fit_transform(all_data[feature].astype(str))
        
        # 刪除無用列
        all_data.drop(['att1', 'att2', 'distance'], axis=1, inplace=True)
        
        # 分割數據集
        train_processed = all_data[:len(train_df)]
        test_processed = all_data[len(train_df):]
        
        print("數據處理完成！\n")
        return train_processed, test_processed
    except Exception as e:
        print("數據處理失敗:", str(e))
        return None, None
```

### 4. 模型訓練

`train_simple_model()` 函數使用LightGBM訓練分類模型。它分割訓練數據集，訓練模型並驗證AUC分數。

```python
def train_simple_model(train_data, test_data):
    print("開始訓練模型...")
    try:
        # 準備特徵和標籤
        features = [col for col in train_data.columns if col not in ['fraud', 'Id']]
        X = train_data[features]
        y = train_data['fraud']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 建立LightGBM分類器
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=15,
            num_leaves=31,
            random_state=42
        )
        
        # 訓練模型
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[early_stopping(stopping_rounds=80)]
        )
        
        # 預測
        test_features = test_data[features]
        predictions = model.predict_proba(test_features)[:, 1]
        
        print("\n模型訓練完成！")
        print(f"AUC分數: {roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]):.4f}")
        return predictions
    except Exception as e:
        print("模型訓練失敗:", str(e))
        return None
```

### 5. 生成提交文件

`create_submission()` 函數根據模型預測生成提交文件。預測結果轉換為二進制（0或1），並保存為CSV格式。

```python
def create_submission(test_df, predictions, filename='submission.csv'):
    print("開始生成提交文件...")
    try:
        binary_predictions = (predictions >= 0.5).astype(int)
        submission = pd.DataFrame({'Id': range(1, len(binary_predictions) + 1), 'fraud': binary_predictions})
        submission.to_csv(filename, index=False)
        print(f"提交文件已生成: {filename}")
    except Exception as e:
        print("提交文件生成失敗:", str(e))
```

### 6. 主函數

`main()` 函數是程式的入口，依序執行環境檢查、數據讀取、數據處理、模型訓練及提交文件生成。

```python
def main():
    print("=== 信用卡詐騙偵測系統 ===")
    if not check_environment():
        return
    train_df, test_df = load_data()
    if train_df is None or test_df is None:
        return
    train_processed, test_processed = basic_preprocessing(train_df, test_df)
    if train_processed is None or test_processed is None:
        return
    predictions = train_simple_model(train_processed, test_processed)
    if predictions is None:
        return
    create_submission(test_df, predictions)
    print("=== 處理完成 ===")
```
