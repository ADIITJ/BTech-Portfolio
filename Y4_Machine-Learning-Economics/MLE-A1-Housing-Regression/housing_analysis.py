#!/usr/bin/env python3
"""
Housing Price Analysis Using Linear Regression
Machine Learning for Economics - Assignment 1
Student: Atharva Date (B22AI045)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns

def main():
    """Main function to run the complete housing price analysis"""
    
    print("="*60)
    print("Housing Price Analysis Using Linear Regression")
    print("="*60)
    
    # Load and explore dataset
    print("\n1. Loading and exploring dataset...")
    df = load_and_explore_data()
    
    # Correlation analysis
    print("\n2. Generating correlation analysis...")
    correlation_analysis(df)
    
    # Data preprocessing
    print("\n3. Preprocessing data...")
    df_encoded, df_with_dummies = preprocess_data(df)
    
    # Question 1: Price Semi-Elasticity of Bedrooms
    print("\n4. Question 1: Price Semi-Elasticity of Bedrooms")
    question_1_analysis(df)
    
    # Question 2: Effect of School Quality on Prices
    print("\n5. Question 2: Effect of School Quality on Prices")
    question_2_analysis(df)
    
    # Question 3: Price Prediction for Specific House
    print("\n6. Question 3: Price Prediction for Specific House")
    question_3_analysis(df_encoded)
    
    # Question 4: Zoning Type Effects Across Zones
    print("\n7. Question 4: Zoning Type Effects Across Zones")
    question_4_analysis(df_with_dummies)
    
    print("\n" + "="*60)
    print("Analysis Complete! Check generated plots in current directory.")
    print("="*60)

def load_and_explore_data():
    """Load housing dataset and perform basic exploration"""
    df = pd.read_csv('housing_prices.csv')
    print(f"Dataset shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    df.info()
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nZoning types:")
    print(df['zoning_type'].value_counts())
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df

def correlation_analysis(df):
    """Generate correlation heatmap for numeric variables"""
    numeric_cols = ['price', 'sqft_living', 'bedrooms', 'bathrooms', 'year_built', 
                    'lot_size', 'garage_spaces', 'distance_to_downtown', 
                    'school_quality', 'crime_rate', 'median_income']
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Housing Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Correlation heatmap saved as 'correlation_heatmap.png'")

def preprocess_data(df):
    """Encode categorical variables and create dummy variables"""
    # Encode categorical variables
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['zoning_encoded'] = le.fit_transform(df['zoning_type'])
    
    # Create dummy variables for zoning type (properly encoded as 0/1)
    zoning_dummies = pd.get_dummies(df['zoning_type'], prefix='zoning', dtype=int)
    df_with_dummies = pd.concat([df, zoning_dummies], axis=1)
    
    print("Zoning encoding mapping:")
    for i, zone in enumerate(le.classes_):
        print(f"{zone}: {i}")
    
    print(f"\nZoning dummy columns created: {list(zoning_dummies.columns)}")
    print(f"Sample of dummy variables:")
    print(zoning_dummies.head())
    
    return df_encoded, df_with_dummies

def question_1_analysis(df):
    """Analyze price semi-elasticity of bedrooms"""
    print("\nQuestion 1: Price Semi-Elasticity of Bedrooms")
    print("-" * 50)
    
    # Semi-elasticity requires log transformation of dependent variable
    df_q1 = df.copy()
    df_q1['log_price'] = np.log(df_q1['price'])
    
    # Features for the model
    X_q1 = df_q1[['bedrooms']]
    y_q1 = df_q1['log_price']
    
    # Fit linear regression
    model_q1 = LinearRegression()
    model_q1.fit(X_q1, y_q1)
    
    # Semi-elasticity is the coefficient of bedrooms
    semi_elasticity = model_q1.coef_[0]
    r2_q1 = r2_score(y_q1, model_q1.predict(X_q1))
    
    print(f"Price semi-elasticity of bedrooms: {semi_elasticity:.4f}")
    print(f"This means a 1-unit increase in bedrooms increases price by {semi_elasticity*100:.2f}%")
    print(f"R-squared: {r2_q1:.4f}")
    
    # Visualization for bedrooms vs price relationship
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['bedrooms'], df['price'], alpha=0.6, color='darkgreen')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Price ($)')
    plt.title('Housing Price vs Number of Bedrooms')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['bedrooms'], np.log(df['price']), alpha=0.6, color='darkgreen')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Log Price')
    plt.title('Log Housing Price vs Number of Bedrooms')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bedrooms_vs_price_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Bedrooms analysis plot saved as 'bedrooms_vs_price_analysis.png'")

def question_2_analysis(df):
    """Analyze effect of school quality on housing prices"""
    print("\nQuestion 2: Effect of School Quality on Prices")
    print("-" * 50)
    
    # Linear regression with school quality as predictor
    X_q2 = df[['school_quality']]
    y_q2 = df['price']
    
    model_q2 = LinearRegression()
    model_q2.fit(X_q2, y_q2)
    
    school_coef = model_q2.coef_[0]
    intercept_q2 = model_q2.intercept_
    r2_q2 = r2_score(y_q2, model_q2.predict(X_q2))
    
    print(f"School quality coefficient: {school_coef:.2f}")
    print(f"Intercept: {intercept_q2:.2f}")
    print(f"R-squared: {r2_q2:.4f}")
    
    if school_coef > 0:
        print(f"\nYES, housing prices increase with higher school quality score.")
        print(f"Each 1-point increase in school quality increases price by ${school_coef:.2f}")
    else:
        print(f"\nNO, housing prices decrease with higher school quality score.")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(df['school_quality'], df['price'], alpha=0.6, color='steelblue')
    plt.plot(df['school_quality'], model_q2.predict(X_q2), color='red', linewidth=2)
    plt.xlabel('School Quality Score')
    plt.ylabel('Price ($)')
    plt.title('Housing Price vs School Quality Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('school_quality_vs_price.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("School quality analysis plot saved as 'school_quality_vs_price.png'")

def question_3_analysis(df_encoded):
    """Predict price for specific house characteristics"""
    print("\nQuestion 3: Price Prediction for Specific House")
    print("-" * 50)
    
    # Build comprehensive model with all relevant features
    features = ['sqft_living', 'bedrooms', 'bathrooms', 'year_built', 'lot_size', 
               'garage_spaces', 'distance_to_downtown', 'school_quality', 
               'crime_rate', 'median_income']
    
    # Add encoded zoning
    X_full = df_encoded[features + ['zoning_encoded']]
    y_full = df_encoded['price']
    
    model_full = LinearRegression()
    model_full.fit(X_full, y_full)
    
    r2_full = r2_score(y_full, model_full.predict(X_full))
    print(f"Full model R-squared: {r2_full:.4f}")
    
    # House specifications
    house_specs = {
        'sqft_living': 1200,
        'bedrooms': 3,
        'bathrooms': 2,
        'year_built': 1980,
        'lot_size': 1857,
        'garage_spaces': 0,
        'distance_to_downtown': 13.5,
        'school_quality': 1,
        'crime_rate': 14.2,
        'median_income': 67262,
        'zoning_encoded': 2  # Residential = 2 based on encoding
    }
    
    # Create prediction array
    house_features = np.array([[house_specs[feature] for feature in features + ['zoning_encoded']]])
    
    predicted_price = model_full.predict(house_features)[0]
    
    print(f"\nHouse specifications:")
    for key, value in house_specs.items():
        if key != 'zoning_encoded':
            print(f"{key}: {value}")
    print(f"zoning_type: Residential")
    
    print(f"\nPredicted price: ${predicted_price:,.2f}")

def question_4_analysis(df_with_dummies):
    """Analyze zoning type effects across different zones"""
    print("\nQuestion 4: Zoning Type Effects Across Zones")
    print("-" * 50)
    
    # Debug: Check data types and structure
    print("All columns starting with 'zoning':")
    all_zoning_cols = [col for col in df_with_dummies.columns if col.startswith('zoning')]
    for col in all_zoning_cols:
        print(f"{col}: {df_with_dummies[col].dtype}, unique values: {df_with_dummies[col].unique()}")
    
    print(f"\nNumeric zoning dummy columns only:")
    zoning_dummy_cols = [col for col in df_with_dummies.columns if col.startswith('zoning_') and col != 'zoning_type']
    for col in zoning_dummy_cols:
        print(f"{col}: {df_with_dummies[col].dtype}")
    
    print(f"\nColumns to use in model (dropping last dummy): {zoning_dummy_cols[:-1]}")
    
    # Analyze effect of zoning type using dummy variables
    zoning_features = ['sqft_living', 'bedrooms', 'bathrooms', 'year_built', 'lot_size',
                      'garage_spaces', 'distance_to_downtown', 'school_quality', 
                      'crime_rate', 'median_income']
    
    # Get only the numeric dummy columns (exclude original zoning_type column)
    zoning_dummy_cols = [col for col in df_with_dummies.columns if col.startswith('zoning_') and col != 'zoning_type']
    print(f"\nZoning dummy columns: {zoning_dummy_cols}")
    
    # Use all features plus zoning dummies (drop one dummy to avoid multicollinearity)
    X_zoning = df_with_dummies[zoning_features + zoning_dummy_cols[:-1]]
    y_zoning = df_with_dummies['price']
    
    print(f"Features used: {zoning_features + zoning_dummy_cols[:-1]}")
    print(f"X_zoning shape: {X_zoning.shape}")
    
    model_zoning = LinearRegression()
    model_zoning.fit(X_zoning, y_zoning)
    
    r2_zoning = r2_score(y_zoning, model_zoning.predict(X_zoning))
    print(f"Zoning model R-squared: {r2_zoning:.4f}")
    
    # Extract zoning coefficients
    zoning_coefs = {}
    for i, col in enumerate(zoning_dummy_cols[:-1]):
        zoning_coefs[col] = model_zoning.coef_[len(zoning_features) + i]
    
    print("\nZoning type effects (relative to baseline):")
    for zone, coef in zoning_coefs.items():
        print(f"{zone}: ${coef:,.2f}")
    
    # Check if effects are significantly different
    coef_values = list(zoning_coefs.values())
    coef_range = max(coef_values) - min(coef_values)
    print(f"\nRange of zoning effects: ${coef_range:,.2f}")
    
    if coef_range > 10000:  # Arbitrary threshold for practical significance
        print("YES, the effect of zoning type on prices is different across zones.")
    else:
        print("NO, the effect of zoning type on prices is similar across zones.")
    
    # Additional analysis: Average prices by zone
    zone_analysis = df_with_dummies.groupby('zoning_type')['price'].agg(['mean', 'std', 'count'])
    print("\nAverage prices by zoning type:")
    print(zone_analysis)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    df_with_dummies.boxplot(column='price', by='zoning_type', ax=plt.gca())
    plt.title('Price Distribution by Zoning Type')
    plt.suptitle('')  # Remove default title
    plt.ylabel('Price ($)')
    plt.xlabel('Zoning Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('price_distribution_by_zoning.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Zoning analysis plot saved as 'price_distribution_by_zoning.png'")

if __name__ == "__main__":
    main()
