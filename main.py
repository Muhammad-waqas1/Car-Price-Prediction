import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('CAR_DETAILS.csv')

# Feature and target columns
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

# Preprocessing for numerical and categorical data
numerical_features = ['year', 'km_driven']
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']

# Preprocessing pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Models
models = {
    'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regression': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Training and evaluation
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    results[name] = {
        'CV MSE': -np.mean(cv_scores)
    }
    
    # Fit model
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name].update({
        'MAE': mae,
        'RMSE': rmse
    })

     # Visualization: Scatter Plot with Regression Line
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'{name}: Actual vs Predicted Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualization: Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title(f'{name}: Residual Plot')
    plt.grid(True)
    plt.show()

    # Visualization: Histogram of Errors
    plt.figure(figsize=(12, 6))
    plt.hist(residuals, bins=50, edgecolor='k', alpha=0.7)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(f'{name}: Histogram of Prediction Errors')
    plt.grid(True)
    plt.show()
    
# Print results
for name, metrics in results.items():
    print(f'{name} - CV MSE: {metrics["CV MSE"]}')
    print(f'{name} - MAE: {metrics["MAE"]}')
    print(f'{name} - RMSE: {metrics["RMSE"]}\n')