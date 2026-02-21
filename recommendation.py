import pandas as pd
import numpy as np

def get_crop_statistics(df: pd.DataFrame) -> pd.DataFrame:
    
    crop_stats = df.groupby('Crop').agg({
        'Rainfall': ['mean', 'min', 'max'],
        'Temperature': ['mean', 'min', 'max'],
        'Production': 'mean',
        'Area': 'mean'
    }).round(2)
    
    crop_stats.columns = ['_'.join(col).strip() for col in crop_stats.columns.values]
    crop_stats = crop_stats.reset_index()
    
    return crop_stats

def get_seasonal_crop_data(df: pd.DataFrame, season: str) -> pd.DataFrame:
    
    return df[df['Season'] == season].copy()

def calculate_similarity_score(crop_data: pd.Series, rainfall: float, temperature: float) -> float:
    
    rainfall_diff = abs(rainfall - crop_data['Rainfall_mean']) / (crop_data['Rainfall_max'] - crop_data['Rainfall_min'] + 1)
    temp_diff = abs(temperature - crop_data['Temperature_mean']) / (crop_data['Temperature_max'] - crop_data['Temperature_min'] + 1)
    
    score = rainfall_diff + temp_diff
    
    return score

def recommend_crop(df: pd.DataFrame, rainfall: float, temperature: float, season: str) -> dict:
    
    seasonal_data = get_seasonal_crop_data(df, season)
    
    if seasonal_data.empty:
        return {
            'recommended_crop': None,
            'confidence_score': 0,
            'expected_production': 0,
            'all_recommendations': [],
            'message': f'No data available for {season} season'
        }
    
    crop_stats = seasonal_data.groupby('Crop').agg({
        'Rainfall': ['mean', 'min', 'max'],
        'Temperature': ['mean', 'min', 'max'],
        'Production': 'mean'
    }).round(2)
    
    crop_stats.columns = ['_'.join(col).strip() for col in crop_stats.columns.values]
    crop_stats = crop_stats.reset_index()
    
    recommendations = []
    
    for _, row in crop_stats.iterrows():
        rainfall_in_range = row['Rainfall_min'] <= rainfall <= row['Rainfall_max']
        temp_in_range = row['Temperature_min'] <= temperature <= row['Temperature_max']
        
        score = calculate_similarity_score(row, rainfall, temperature)
        
        confidence = max(0, 100 - (score * 50)) 
        
        recommendations.append({
            'crop': row['Crop'],
            'score': round(score, 4),
            'confidence': round(confidence, 2),
            'expected_production': round(row['Production_mean'], 2),
            'rainfall_match': rainfall_in_range,
            'temp_match': temp_in_range,
            'ideal_rainfall': f"{row['Rainfall_min']}-{row['Rainfall_max']} mm",
            'ideal_temp': f"{row['Temperature_min']}-{row['Temperature_max']} °C"
        })
    
    recommendations.sort(key=lambda x: x['score'])
    
    best = recommendations[0]
    
    return {
        'recommended_crop': best['crop'],
        'confidence_score': best['confidence'],
        'expected_production': best['expected_production'],
        'all_recommendations': recommendations,
        'message': 'Recommendation generated successfully'
    }

def get_seasonal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    
    seasonal_avg = df.groupby('Season')['Production'].mean().reset_index()
    seasonal_avg.columns = ['Season', 'Average Production']
    seasonal_avg['Average Production'] = seasonal_avg['Average Production'].round(2)
    
    return seasonal_avg

def get_crop_season_production(df: pd.DataFrame) -> pd.DataFrame:
    
    grouped = df.groupby(['Crop', 'Season'])['Production'].mean().reset_index()
    grouped['Production'] = grouped['Production'].round(2)
    
    return grouped

def get_top_crops_by_season(df: pd.DataFrame, season: str, top_n: int = 5) -> pd.DataFrame:
    
    seasonal_data = get_seasonal_crop_data(df, season)
    
    top_crops = seasonal_data.groupby('Crop').agg({
        'Production': 'mean',
        'Rainfall': 'mean',
        'Temperature': 'mean'
    }).round(2).reset_index()
    
    top_crops = top_crops.sort_values('Production', ascending=False).head(top_n)
    top_crops.columns = ['Crop', 'Avg Production', 'Avg Rainfall', 'Avg Temperature']
    
    return top_crops
