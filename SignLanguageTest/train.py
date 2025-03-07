import pandas as pd
import numpy as np

# Load CSV files
train_df = pd.read_csv('archive/sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('archive/sign_mnist_test/sign_mnist_test.csv')

# Separate labels and pixel data
X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values
X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Reshape the pixel data to (num_samples, 28, 28, 1) and normalize to [0, 1]
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print("Training samples:", X_train.shape, "Testing samples:", X_test.shape)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define a simple CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(25, activation='softmax')  # Adjust the number if your dataset uses a different number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Train the model (adjust epochs, batch_size as needed)
with tf.device('/device:GPU:0'):
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
# model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to disk
with open('sign_mnist_model_20epoch.tflite', 'wb') as f:
    f.write(tflite_model)
    
print("Model converted to TFLite and saved as sign_mnist_model.tflite")
