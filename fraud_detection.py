import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

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

def create_preprocessing_pipeline():
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
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=1  # Balance for imbalanced classes
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def predict_submission(pipeline, test_data):
    test_data = feature_engineering(test_data)
    predictions = pipeline.predict(test_data)
    
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'fraud': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    return submission

def main():
    train_data, test_data = load_data()
    train_data = feature_engineering(train_data)
    
    X = train_data.drop('fraud', axis=1)
    y = train_data['fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    preprocessor = create_preprocessing_pipeline()
    pipeline = train_model(X_train, y_train, preprocessor)
    
    evaluate_model(pipeline, X_test, y_test)
    submission = predict_submission(pipeline, test_data)

if __name__ == "__main__":
    main()