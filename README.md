# 🌾 AI-Based Smart Crop Yield Prediction and Seasonal Analysis System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive data analytics project using Python and Streamlit for predicting crop yields, analyzing seasonal trends, and providing smart crop recommendations using Machine Learning.

Deployment link-https://ai-based-smart-crop-yield-prediction-and-seasonal-analysis-sys.streamlit.app/
## 📋 Features

### 1. 🌱 Crop Yield Prediction
- Predict crop production based on crop type, season, rainfall, temperature, and area
- Uses multiple ML models and automatically selects the best one
- Real-time predictions with confidence metrics

### 2. 🤖 Machine Learning Models
- **Linear Regression** - Baseline model
- **Ridge Regression** - L2 regularized linear model
- **Random Forest Regressor** - Ensemble learning with 200 trees
- **Extra Trees Regressor** - Extremely randomized trees
- **Gradient Boosting Regressor** - Sequential ensemble method
- **XGBoost Regressor** - Advanced gradient boosting (optional)

### 3. 📊 Model Evaluation
- **R² Score** - Coefficient of Determination
- **MAE** - Mean Absolute Error
- **RMSE** - Root Mean Square Error
- **Cross-Validation** - 5-fold CV for robust evaluation

### 4. 📈 Seasonal Analysis
- Average production by season (bar chart)
- Rainfall vs Production relationship (scatter plot)
- Seasonal statistics and insights

### 5. 💡 Crop Recommendation
- Smart recommendations based on rainfall, temperature, and season
- Confidence scores for each recommendation
- Expected production estimates

## 📁 Project Structure

```
AI-Smart-Crop-Prediction/
│
├── app.py                 # Main Streamlit application
├── preprocessing.py       # Data preprocessing module
├── model.py              # ML model training and evaluation
├── recommendation.py     # Crop recommendation logic
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── LICENSE              # MIT License
├── .gitignore           # Git ignore rules
│
└── data/
    └── crop_data.csv    # Sample dataset
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alokkgupta28/Ai-Based-Smart-Crop-Yield-Prediction-and-Seasonal-Analysis-System.git
   cd Ai-Based-Smart-Crop-Yield-Prediction-and-Seasonal-Analysis-System
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 Dataset

The project includes a sample dataset (`data/crop_data.csv`) with the following columns:

| Column | Description | Type |
|--------|-------------|------|
| Crop | Type of crop | Categorical |
| Season | Growing season | Categorical |
| Rainfall | Rainfall in mm | Numeric |
| Temperature | Temperature in °C | Numeric |
| Area | Cultivation area in hectares | Numeric |
| Production | Crop production in tons | Numeric (Target) |

### Crops Included (14 Types)
Rice, Wheat, Maize, Cotton, Sugarcane, Soybean, Groundnut, Potato, Tomato, Onion, Mustard, Barley, Chickpea, Lentil

### Seasons
- **Kharif** (Monsoon: June-October)
- **Rabi** (Winter: October-March)
- **Summer** (March-June)

## 🛠️ Technical Details

### Hyperparameter Configuration

| Model | Key Parameters |
|-------|---------------|
| Random Forest | n_estimators=200, max_depth=15, max_features='sqrt' |
| Gradient Boosting | n_estimators=200, learning_rate=0.1, max_depth=5 |
| XGBoost | n_estimators=200, learning_rate=0.1, reg_alpha=0.1 |
| Extra Trees | n_estimators=200, max_depth=15 |

### Module Functions

#### preprocessing.py
- `load_data()` - Load CSV data
- `handle_missing_values()` - Fill missing values (median/mode)
- `encode_categorical_variables()` - Label encoding for Crop and Season
- `preprocess_data()` - Complete preprocessing pipeline
- `get_features_and_target()` - Extract features (X) and target (y)

#### model.py
- `train_*()` - Train individual models with optimized hyperparameters
- `evaluate_model()` - Calculate R², MAE, RMSE with cross-validation
- `select_best_model()` - Select model with highest R²
- `predict_yield()` - Make predictions for new data

#### recommendation.py
- `recommend_crop()` - Get crop recommendations based on conditions
- `get_seasonal_analysis()` - Average production by season
- `get_top_crops_by_season()` - Top performing crops per season

## 📈 Model Performance

The system automatically evaluates all models and selects the best one based on R² score. Typical performance:

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| XGBoost | ~0.95+ | Low | Low |
| Random Forest | ~0.93+ | Low | Low |
| Gradient Boosting | ~0.92+ | Low | Low |

*Actual results may vary based on dataset*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👩‍🎓 Educational Purpose

This project is designed to be:

- **Easy to understand** - Clean, well-commented code
- **Modular** - Separate files for different functionalities
- **Educational** - Demonstrates ML concepts and data analytics

### Key Learning Points
1. Data preprocessing with Pandas
2. Machine Learning with Scikit-learn & XGBoost
3. Web applications with Streamlit
4. Data visualization with Matplotlib
5. Cross-validation and model evaluation
6. Hyperparameter tuning

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Scikit-learn](https://scikit-learn.org/) for ML algorithms
- [XGBoost](https://xgboost.ai/) for advanced gradient boosting

---

**Built with ❤️ using Python, Streamlit, Scikit-learn, and XGBoost**
