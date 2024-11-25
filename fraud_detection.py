import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

def load_data():
    train_data = pd.read_csv('train_data.csv')
    test_data = pd.read_csv('test_data.csv')
    return train_data, test_data

def feature_engineering(df):
    df['hour'] = pd.to_datetime(df['att1'], format='%H:%M').dt.hour
    df['is_night'] = (df['hour'] >= 22) | (df['hour'] <= 5)
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    df['merchant_distance'] = haversine_distance(
        df['att12'], df['att13'], 
        df['att15'], df['att16']
    )
    
    return df

def create_preprocessor():
    numeric_features = ['att4', 'att5', 'att10', 'merchant_distance', 'hour']
    categorical_features = ['att3', 'att6', 'att7', 'att8', 'att9']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])
    
    return preprocessor

def train_model(X_train, y_train, preprocessor):
    # Preprocess the data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train_processed, label=y_train)
    
    # Parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'is_unbalance': True
    }
    
    # Train model
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=200
    )
    
    return preprocessor, model

def evaluate_model(preprocessor, model, X_test, y_test):
    # Preprocess test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Predict
    y_pred = (model.predict(X_test_processed) > 0.5).astype(int)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def predict_submission(preprocessor, model, test_data):
    test_data = feature_engineering(test_data)
    
    # Preprocess test data
    X_test_processed = preprocessor.transform(test_data)
    
    # Predict
    predictions = (model.predict(X_test_processed) > 0.5).astype(int)
    
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'fraud': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    return submission

def main():
    # Load Data
    train_data, test_data = load_data()
    
    # Feature Engineering
    train_data = feature_engineering(train_data)
    
    # Split Features and Target
    X = train_data.drop('fraud', axis=1)
    y = train_data['fraud']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create Preprocessor
    preprocessor = create_preprocessor()
    
    # Train Model
    preprocessor, model = train_model(X_train, y_train, preprocessor)
    
    # Evaluate Model
    evaluate_model(preprocessor, model, X_test, y_test)
    
    # Generate Submission
    submission = predict_submission(preprocessor, model, test_data)

if __name__ == "__main__":
    main()