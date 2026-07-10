
import os
import random
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

# Define device for model operations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing and feature engineering
def preprocess_data(df):
    # Handle missing values using mean imputation
    df['feature1'] = df['feature1'].fillna(df['feature1'].mean())
    
    # Encode categorical variables using one-hot encoding
    encoded_df = pd.get_dummies(df, columns=['categorical_feature'])
    
    # Scale numerical features using standardization
    scaler = StandardScaler()
    scaled_df = encoded_df[['numerical_feature']]
    scaled_df[['numerical_feature']] = scaler.fit_transform(scaled_df[['numerical_feature']])
    
    return scaled_df

def balance_classes(df):
    ros = RandomOverSampler(random_state=42)
    rus = RandomUnderSampler(random_state=42)
    df_balanced = pd.concat([ros.fit_resample(df, df['target']), rus.fit_resample(df, df['target'])], axis=0)
    
    return df_balanced

def generate_new_features(df):
    pca = PCA(n_components=5)
    new_features = pca.fit_transform(df[['numerical_feature']])
    new_df = pd.DataFrame(new_features, columns=['new_feature1', 'new_feature2', 'new_feature3', 'new_feature4', 'new_feature5'])
    
    return new_df

def select_most_relevant_features(df):
    X = df.drop('target', axis=1)
    y = df['target']
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=200), n_features_to_select=10, step=0.2)
    rfe.fit(X, y)
    
    return X.columns[rfe.support_]

# Model selection
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, f1

# Model deployment
def prepare_model_for_deployment(model):
    # Save the trained model to "./agent_workspace/trained_models" directory
    torch.save(model.state_dict(), './agent_workspace/trained_models/random_forest_classifier.pth')
    
    return model

def deploy_model():
    # Load the saved model from "./agent_workspace/trained_models" directory
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    model.load_state_dict(torch.load('./agent_workspace/trained_models/random_forest_classifier.pth'))
    
    return model

# Main function to orchestrate the data loading, data preprocessing, feature engineering, model training, model preparation, model deployment, and model evaluation
def main():
    # Load dataset from "./agent_workspace/datasets/sample_banana.csv"
    df = pd.read_csv('./agent_workspace/datasets/sample_banana.csv')
    
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)
    
    # Preprocess data using the "preprocess_data" function
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    
    # Balance classes using the "balance_classes" function
    X_train_balanced = balance_classes(X_train)
    y_train_balanced = y_train
    
    # Generate new features using the "generate_new_features" function
    X_train_new = generate_new_features(X_train_balanced)
    X_test_new = generate_new_features(X_test)
    
    # Select most relevant features using the "select_most_relevant_features" function
    X_train_selected = select_most_relevant_features(X_train_new)
    X_test_selected = select_most_relevant_features(X_test_new)
    
    # Train model using the "train_model" function
    model = train_model(X_train_selected, y_train_balanced)
    
    # Evaluate model performance using the "evaluate_model" function
    accuracy, f1 = evaluate_model(model, X_test_selected, y_test)
    
    print("Model Performance on Test Set:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-score: {f1:.3f}")
    
    # Prepare model for deployment using the "prepare_model_for_deployment" function
    deployable_model = prepare_model_for_deployment(model)
    
    return deployable_model

if __name__ == "__main__":
    deployable_model = main()
