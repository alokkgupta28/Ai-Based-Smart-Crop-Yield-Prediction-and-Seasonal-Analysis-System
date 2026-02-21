import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    return df_clean

def encode_categorical_variables(df: pd.DataFrame) -> tuple:
    
    df_encoded = df.copy()
    
    crop_encoder = LabelEncoder()
    season_encoder = LabelEncoder()
    
    df_encoded['Crop_Encoded'] = crop_encoder.fit_transform(df_encoded['Crop'])
    
    df_encoded['Season_Encoded'] = season_encoder.fit_transform(df_encoded['Season'])
    
    return df_encoded, crop_encoder, season_encoder

def preprocess_data(filepath: str) -> tuple:
    
    df = load_data(filepath)
    original_df = df.copy()
    
    df_clean = handle_missing_values(df)
    
    df_encoded, crop_encoder, season_encoder = encode_categorical_variables(df_clean)
    
    return df_encoded, original_df, crop_encoder, season_encoder

def get_features_and_target(df: pd.DataFrame, scale_features: bool = False) -> tuple:
    
    feature_cols = ['Crop_Encoded', 'Season_Encoded', 'Rainfall', 'Temperature', 'Area']
    
    X = df[feature_cols].copy()
    y = df['Production']
    
    if scale_features:
        scaler = StandardScaler()
        numeric_cols = ['Rainfall', 'Temperature', 'Area']
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        return X, y, scaler
    
    return X, y

def get_unique_values(df: pd.DataFrame) -> dict:
    
    return {
        'crops': sorted(df['Crop'].unique().tolist()),
        'seasons': sorted(df['Season'].unique().tolist())
    }
