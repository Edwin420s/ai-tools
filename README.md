AI Tools : Mastering the AI Toolkit


https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/TensorFlow-2.12-orange
https://img.shields.io/badge/Scikit--learn-1.2-green
https://img.shields.io/badge/spaCy-3.5-lightblue
https://img.shields.io/badge/License-MIT-yellow


A comprehensive implementation of an AI Tools Assignment demonstrating proficiency with various AI frameworks and tools through theoretical analysis and practical implementation.

📁 Project Structure

```
ai-tools-assignment/
│
├── notebooks/                          # Jupyter notebooks for each task
│   ├── 1_iris_classical_ml.ipynb      # Classical ML with Scikit-learn
│   ├── 2_mnist_deep_learning.ipynb    # Deep Learning with TensorFlow
│   ├── 3_spacy_nlp_analysis.ipynb     # NLP with spaCy
│   └── 4_tensorflow_debugging.ipynb   # Debugging exercises
│
├── src/                                # Python source code
│   ├── classical_ml.py                # Iris classification functions
│   ├── deep_learning.py               # MNIST CNN implementation
│   ├── nlp_processing.py              # NLP processing with spaCy
│   ├── deployment_app.py              # Streamlit deployment app
│   └── utils.py                       # Utility functions
│
├── data/                               # Dataset files
│   ├── amazon_reviews_sample.csv      # Sample product reviews
│   └── iris_data.csv                  # Iris dataset
│
├── models/                             # Trained model files (generated)
│   ├── mnist_cnn_model.h5             # Trained CNN model
│   └── decision_tree_iris.pkl         # Trained decision tree
│
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
└── .gitignore                         # Git ignore rules

```

🛠️ Installation

Prerequisites
Python 3.8 or higher
pip (Python package manager)

Step-by-Step Installation
Clone the repository:
```
git clone https://github.com/Edwin420s/ai-tools.git 
cd ai-tools
```

Create a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```
Install dependencies:
```
pip install -r requirements.txt
```

Download spaCy language model:
```
python -m spacy download en_core_web_sm
```
Generate sample data (if needed):
```
python -c "
import pandas as pd
from sklearn.datasets import load_iris

# Create Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
iris_df.to_csv('data/iris_data.csv', index=False)

# Create sample reviews
reviews = [
    {'review_id': 1, 'product_id': 'IPHONE13', 'review_text': 'This iPhone is amazing!', 'rating': 5},
    {'review_id': 2, 'product_id': 'SAMSUNGTVC', 'review_text': 'Terrible picture quality.', 'rating': 1},
    # ... more reviews
]
pd.DataFrame(reviews).to_csv('data/amazon_reviews_sample.csv', index=False)
print('Sample data created successfully!')
"
```

Usage
Running Jupyter Notebooks
Start Jupyter Notebook:

```
jupyter notebook
```
Execute notebooks in order:

1_iris_classical_ml.ipynb - Classical ML with Iris dataset
2_mnist_deep_learning.ipynb - Deep Learning with MNIST
3_spacy_nlp_analysis.ipynb - NLP with spaCy
4_tensorflow_debugging.ipynb - Debugging exercises

Running Python Scripts
Classical ML implementation:

```
python src/classical_ml.py
```
Deep Learning implementation:

```
python src/deep_learning.py
```

NLP processing:

```
python src/nlp_processing.py
```
Deployment app:

```
streamlit run src/deployment_app.py
```

Features
1. Classical Machine Learning
   
Iris species classification using Decision Trees
Data preprocessing and feature engineering
Model evaluation with accuracy, precision, and recall
Confusion matrix visualization

2. Deep Learning
   
CNN architecture for MNIST digit classification
TensorFlow implementation with Keras
Training history visualization
Sample prediction visualization

3. Natural Language Processing
   
Named Entity Recognition with spaCy
Rule-based sentiment analysis
Entity distribution visualization
Sentiment analysis comparison with ratings

5. Model Deployment
   
Streamlit web application
Interactive digit classification
Real-time predictions
Probability visualization

5. Debugging Exercises
   
Common TensorFlow error examples
Solutions and explanations
Best practices for debugging

Technical Implementation
Classical ML Implementation
```
# Example code structure
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocess_data()

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

Deep Learning Architecture
```
CNN Architecture:
- Input: 28x28x1 (MNIST images)
- Conv2D: 32 filters, 3x3 kernel, ReLU activation
- MaxPooling: 2x2 pool size
- Conv2D: 64 filters, 3x3 kernel, ReLU activation
- MaxPooling: 2x2 pool size
- Flatten layer
- Dense: 128 units, ReLU activation
- Dropout: 0.5 rate
- Output: 10 units, Softmax activation
```

NLP Processing Pipeline
```
# spaCy NLP pipeline
nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment = analyze_sentiment(text)
    return entities, sentiment
```
Results
Performance Metrics
Task	Model	Accuracy	Precision	Recall
Iris Classification	Decision Tree	96.67%	96.67%	96.67%
MNIST Classification	CNN	98.20%	98.25%	98.20%
Sentiment Analysis	Rule-based	80.00%	82.00%	80.00%


Visualizations
Training History: Accuracy and loss curves for CNN training
Confusion Matrix: Classification performance visualization
Entity Distribution: Named entity frequency analysis
Sentiment Distribution: Pie chart of sentiment analysis results
Sample Predictions: MNIST digit classification examples


📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

