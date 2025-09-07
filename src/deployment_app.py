import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from deep_learning import load_mnist_data, build_cnn_model, compile_model

def load_trained_model():
    """
    Load pre-trained MNIST model or train a new one if not available
    """
    try:
        model = tf.keras.models.load_model('models/mnist_cnn_model.h5')
        st.sidebar.success("Loaded pre-trained model")
    except:
        st.sidebar.info("Training new model... This may take a few minutes.")
        (X_train, y_train), (X_test, y_test) = load_mnist_data()
        model = build_cnn_model()
        model = compile_model(model)
        model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=0)
        model.save('models/mnist_cnn_model.h5')
        st.sidebar.success("Model trained and saved")
    
    return model

def preprocess_image(image):
    """
    Preprocess uploaded image for model prediction
    """
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array
    image_array = np.array(image) / 255.0
    # Reshape for model input
    image_array = image_array.reshape(1, 28, 28, 1)
    
    return image_array

def main():
    """
    Main Streamlit app function
    """
    st.title("MNIST Digit Classifier")
    st.write("Upload an image of a handwritten digit (0-9) for classification")
    
    # Load model
    model = load_trained_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Display results
        st.success(f"Prediction: **{predicted_digit}**")
        st.info(f"Confidence: **{confidence:.2%}**")
        
        # Show prediction probabilities
        st.subheader("Prediction Probabilities")
        fig, ax = plt.subplots()
        ax.bar(range(10), prediction[0])
        ax.set_xlabel('Digit')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        st.pyplot(fig)
        
        # Show sample images from dataset
        st.subheader("Sample Images from MNIST Dataset")
        (_, _), (X_test, y_test) = load_mnist_data()
        sample_indices = np.random.choice(len(X_test), 5, replace=False)
        
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, idx in enumerate(sample_indices):
            axes[i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'Label: {np.argmax(y_test[idx])}')
            axes[i].axis('off')
        st.pyplot(fig)

if __name__ == "__main__":
    main()