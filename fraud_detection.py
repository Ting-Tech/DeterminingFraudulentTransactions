import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import train_test_split

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

def load_data():
    print("開始讀取數據...")
    try:
        # 指定列的數據類型
        dtypes = {
            'att2': str,  # 信用卡號碼作為字符串處理
            'att3': str,  # 交易類別
            'att4': float,  # 交易金額
            'att5': int,  # 年齡
            'att6': str,  # 性別
            'att7': str,  # 職業
            'att8': str,  # 城市
            'att9': str,  # 州
            'att10': int,  # 人口數量
            'att11': int,  # 地址編號
            'att12': float,  # 經度
            'att13': float,  # 緯度
            'att14': str,  # 收款者編號
            'att15': float,  # 收款者經度
            'att16': float,  # 收款者緯度
            'fraud': int  # 詐騙標記
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

def process_time(time_str):
    """處理時間格式 'HH:MM'"""
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours + minutes/60
    except:
        return None

def basic_preprocessing(train_df, test_df):
    print("開始數據處理...")
    try:
        # 合併數據集進行處理
        test_df['fraud'] = -1
        all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        
        # 處理時間特徵
        all_data['hour_decimal'] = all_data['att1'].apply(process_time)
        
        # 提取信用卡號碼特徵
        all_data['card_first_4'] = all_data['att2'].str[:4]
        all_data['card_last_4'] = all_data['att2'].str[-4:]
        
        # 計算地理距離
        all_data['distance'] = np.sqrt(
            (all_data['att12'] - all_data['att15'])**2 + 
            (all_data['att13'] - all_data['att16'])**2
        )
        
        # 處理類別特徵
        le = LabelEncoder()
        categorical_features = ['att3', 'att6', 'att7', 'att8', 'att9', 'card_first_4', 'card_last_4', 'att14']
        for feature in categorical_features:
            all_data[feature] = le.fit_transform(all_data[feature].astype(str))
        
        # 刪除原始的時間列和卡號列
        all_data = all_data.drop(['att1', 'att2'], axis=1)
        
        # 分割回訓練集和測試集
        train_processed = all_data[:len(train_df)]
        test_processed = all_data[len(train_df):]
        
        print("數據處理完成！\n")
        return train_processed, test_processed
    
    except Exception as e:
        print("數據處理失敗:", str(e))
        print("錯誤詳情:", str(e))
        return None, None

def train_simple_model(train_data, test_data):
    print("開始訓練模型...")
    try:
        # 準備特徵
        features = [col for col in train_data.columns if col not in ['fraud', 'Id']]
        X = train_data[features]
        y = train_data['fraud']
        
        # 分割訓練集和驗證集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 建立並訓練模型
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        # 使用 eval_set 而不使用 verbose 參數
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc'
        )
        
        # 預測測試集
        test_features = test_data[features]
        predictions = model.predict_proba(test_features)[:, 1]
        
        print("模型訓練完成！\n")
        
        # 輸出特徵重要性
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        print("\n特徵重要性排序：")
        print(feature_importance)
        
        # 輸出驗證集的效能指標
        val_pred = model.predict_proba(X_val)[:, 1]
        val_pred_binary = (val_pred >= 0.5).astype(int)
        from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
        
        print("\n驗證集效能報告：")
        print(f"準確率: {accuracy_score(y_val, val_pred_binary):.4f}")
        print(f"AUC分數: {roc_auc_score(y_val, val_pred):.4f}")
        print("\n詳細分類報告：")
        print(classification_report(y_val, val_pred_binary))
        
        return predictions
        
    except Exception as e:
        print("模型訓練失敗:", str(e))
        print("錯誤詳情:", e)
        import traceback
        traceback.print_exc()
        return None

def create_submission(test_df, predictions, filename='submission.csv'):
    print("開始生成提交文件...")
    try:
        # 將預測機率轉換為二元分類結果
        binary_predictions = (predictions >= 0.5).astype(int)
        
        submission = pd.DataFrame({
            'Id': range(1, len(binary_predictions) + 1),  # 生成連續的Id
            'fraud': binary_predictions
        })
        submission.to_csv(filename, index=False)
        print(f"提交文件已生成: {filename}\n")
        
        # 顯示預測結果統計
        print("\n預測結果統計：")
        print(f"預測為詐騙交易的數量: {sum(binary_predictions)}")
        print(f"預測為正常交易的數量: {len(binary_predictions) - sum(binary_predictions)}")
        print(f"詐騙交易比例: {(sum(binary_predictions)/len(binary_predictions))*100:.2f}%")
        
    except Exception as e:
        print("提交文件生成失敗:", str(e))
        print("錯誤詳情:", str(e))

def main():
    print("=== 信用卡詐騙偵測系統 ===")
    
    # 檢查環境
    if not check_environment():
        return
    
    # 讀取數據
    train_df, test_df = load_data()
    if train_df is None or test_df is None:
        return
    
    # 數據處理
    train_processed, test_processed = basic_preprocessing(train_df, test_df)
    if train_processed is None or test_processed is None:
        return
    
    # 訓練模型和預測
    predictions = train_simple_model(train_processed, test_processed)
    if predictions is None:
        return
    
    # 生成提交文件
    create_submission(test_df, predictions)
    
    print("=== 處理完成 ===")

if __name__ == "__main__":
    main()