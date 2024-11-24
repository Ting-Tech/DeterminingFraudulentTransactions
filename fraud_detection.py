import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import early_stopping
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

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

def process_time(time_str):
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours + minutes / 60
    except:
        return np.nan

def basic_preprocessing(train_df, test_df):
    print("開始數據處理...")
    try:
        test_df['fraud'] = -1
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
        all_data['scaled_amount'] = scaler.fit_transform(all_data[['att4']])  # 交易金額標準化
        all_data['scaled_age'] = scaler.fit_transform(all_data[['att5']])  # 年齡標準化
        
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

def train_simple_model(train_data, test_data):
    print("開始訓練模型...")
    try:
        features = [col for col in train_data.columns if col not in ['fraud', 'Id']]
        X = train_data[features]
        y = train_data['fraud']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=15,
            num_leaves=31,
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[early_stopping(stopping_rounds=80)]
        )
        
        test_features = test_data[features]
        predictions = model.predict_proba(test_features)[:, 1]
        
        print("\n模型訓練完成！")
        print(f"AUC分數: {roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]):.4f}")
        return predictions
    except Exception as e:
        print("模型訓練失敗:", str(e))
        return None

def create_submission(test_df, predictions, filename='submission.csv'):
    print("開始生成提交文件...")
    try:
        binary_predictions = (predictions >= 0.5).astype(int)
        submission = pd.DataFrame({'Id': range(1, len(binary_predictions) + 1), 'fraud': binary_predictions})
        submission.to_csv(filename, index=False)
        print(f"提交文件已生成: {filename}")
    except Exception as e:
        print("提交文件生成失敗:", str(e))

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

if __name__ == "__main__":
    main()
