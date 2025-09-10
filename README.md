# Machine Learning Utilities by Me

A comprehensive utility library for machine learning workflows, featuring ensemble methods, data visualization, feature engineering, and model evaluation tools.

## Overview

This utility library provides essential functions for data science and machine learning projects, with a focus on:
- **Stacking Ensembles**: Advanced ensemble methods for improved model performance
- **Data Visualization**: Comprehensive plotting functions for exploratory data analysis
- **Feature Engineering**: Tools for creating and analyzing feature combinations
- **Model Evaluation**: Cross-validation and performance assessment utilities
- **Interpretability**: SHAP integration for model explainability

## Dependencies

```python
sklearn
matplotlib
seaborn
numpy
pandas
itertools
shap
```

## Functions

### Ensemble Methods

#### `generate_stack(base_models, X, y, X_test, meta_model=None, folds=5, seed=42, eval_metric=root_mean_squared_error)`
Creates a stacking ensemble using cross-validation.

**Parameters:**
- `base_models`: List of sklearn-like models for level-0 predictions
- `X, y`: Training data and targets
- `X_test`: Test data for generating predictions
- `meta_model`: Meta-learner (default: LinearRegression)
- `folds`: Number of cross-validation folds (default: 5)
- `seed`: Random seed for reproducibility (default: 42)
- `eval_metric`: Evaluation function (default: RMSE)

**Returns:**
- Trained meta-model, meta-features for train/test, and final CV score

#### `oof_cross_val(model_class, X_train, y_train, X_test, folds=5, model_params=None, eval_metric=root_mean_squared_error, seed=42)`
Performs out-of-fold cross-validation for a single model.

**Parameters:**
- `model_class`: Model class to instantiate
- `X_train, y_train`: Training data
- `X_test`: Test data for predictions
- `folds`: Number of CV folds (default: 5)
- `model_params`: Model hyperparameters dictionary
- `eval_metric`: Evaluation metric (default: RMSE)
- `seed`: Random seed (default: 42)

**Returns:**
- Final OOF score, averaged test predictions, and OOF predictions

### Data Visualization

#### `plot_nums(dataframe)`
Creates histograms with KDE for all numerical columns in a DataFrame.

#### `plot_cats(dataframe)`
Generates pie charts for categorical features showing value distributions.

#### `heatmap_nums(dataframe)`
Displays correlation heatmap for numerical features and identifies highly correlated pairs (>0.8 threshold).

#### `plot_feature_importance(model, X_train, y_train)`
Visualizes feature importance from tree-based models or linear model coefficients.

#### `plot_model_performance(y_true, y_pred, model_name)`
Creates scatter plot of predictions vs true values with comprehensive regression metrics.

#### `plot_shap_values(model, X_train, y_train)`
Generates SHAP summary plots for model interpretability.

### Feature Engineering

#### `create_combination_features(df, features)`
Creates new features by averaging pairs of existing features.

**Parameters:**
- `df`: Input DataFrame
- `features`: List of feature names to combine

**Returns:**
- DataFrame with additional combination features

## Usage Examples

### Basic Stacking Ensemble
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Define base models
base_models = [
    RandomForestRegressor(n_estimators=100),
    Ridge(alpha=1.0),
    XGBRegressor(n_estimators=100)
]

# Create stacking ensemble
meta_model, meta_train, meta_test, score = generate_stack(
    base_models, X_train, y_train, X_test
)
```

### Data Exploration
```python
# Visualize numerical distributions
plot_nums(df)

# Analyze categorical features
plot_cats(df)

# Check feature correlations
heatmap_nums(df)
```

### Model Evaluation
```python
# Perform cross-validation
cv_score, test_preds, oof_preds = oof_cross_val(
    XGBRegressor, X_train, y_train, X_test,
    model_params={'n_estimators': 100, 'max_depth': 6}
)

# Evaluate model performance
plot_model_performance(y_true, y_pred, "XGBoost")
```

### Feature Engineering
```python
# Create combination features
df_enhanced = create_combination_features(df, ['feature1', 'feature2', 'feature3'])
```

## Key Features

- **Cross-validation**: Built-in K-fold cross-validation with customizable metrics
- **Ensemble Methods**: Professional-grade stacking implementation
- **Visualization**: Production-ready plotting functions
- **Model Agnostic**: Works with any sklearn-compatible estimator
- **Comprehensive Metrics**: 12+ regression evaluation metrics
- **Interpretability**: SHAP integration for feature importance analysis

## License

This utility library is designed for machine learning competitions and research projects.
