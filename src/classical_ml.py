import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import plot_confusion_matrix

def load_iris_data():
    """
    Load and return Iris dataset
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    return X, y, feature_names, target_names

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocess data: split into train/test sets and scale features
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train a decision tree classifier
    """
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluate model and return metrics
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Plot confusion matrix
    plt = plot_confusion_matrix(y_test, y_pred, target_names)
    plt.show()
    
    return accuracy, precision, recall, y_pred

def run_iris_classification():
    """
    Complete workflow for Iris classification
    """
    print("Loading Iris dataset...")
    X, y, feature_names, target_names = load_iris_data()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    print("Training Decision Tree model...")
    model = train_decision_tree(X_train, y_train)
    
    print("Evaluating model...")
    accuracy, precision, recall, y_pred = evaluate_model(model, X_test, y_test, target_names)
    
    return model, accuracy, precision, recall

if __name__ == "__main__":
    run_iris_classification()