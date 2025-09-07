import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from .utils import plot_training_history

def load_mnist_data():
    """
    Load and preprocess MNIST dataset
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Reshape and normalize
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (X_train, y_train), (X_test, y_test)

def build_cnn_model(input_shape=(28, 28, 1)):
    """
    Build CNN model for MNIST classification
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    return model

def compile_model(model):
    """
    Compile the model with appropriate optimizer and loss
    """
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=10, validation_split=0.1):
    """
    Train the model and return history
    """
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=128,
                        validation_split=validation_split,
                        verbose=1)
    return history

def evaluate_cnn_model(model, X_test, y_test):
    """
    Evaluate the model on test data
    """
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    return test_loss, test_acc

def visualize_predictions(model, X_test, y_test, num_samples=5):
    """
    Visualize model predictions on sample images
    """
    # Get predictions
    predictions = model.predict(X_test[:num_samples])
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test[:num_samples], axis=1)
    
    # Plot samples
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_labels[i]}\nPred: {predicted_labels[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return predicted_labels, true_labels

def run_mnist_classification():
    """
    Complete workflow for MNIST classification
    """
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    
    print("Building CNN model...")
    model = build_cnn_model()
    model = compile_model(model)
    
    print("Training model...")
    history = train_model(model, X_train, y_train, epochs=10)
    
    print("Evaluating model...")
    test_loss, test_acc = evaluate_cnn_model(model, X_test, y_test)
    
    print("Visualizing training history...")
    plot_training_history(history)
    plt.show()
    
    print("Visualizing sample predictions...")
    visualize_predictions(model, X_test, y_test)
    
    return model, test_acc, history

if __name__ == "__main__":
    run_mnist_classification()