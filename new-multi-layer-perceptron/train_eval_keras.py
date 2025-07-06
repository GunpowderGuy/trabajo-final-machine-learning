# train_eval_keras.py
from keras.models import Sequential
from keras.layers import Dense, ReLU
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import Input
import numpy as np
from activation_dropout_keras import ActivationDropout

# Generate dummy data
num_samples = 1000
input_dim = 100
num_classes = 10

X_train = np.random.randn(num_samples, input_dim).astype("float32")
y_train = np.random.randint(0, num_classes, size=(num_samples,))

# Build model
model = Sequential([
    Input(shape=(input_dim,)),
    Dense(128),
    ReLU(),
    ActivationDropout(base_retain_prob=0.5),
    Dense(num_classes, activation='softmax')
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

# Evaluate
loss, acc = model.evaluate(X_train, y_train, verbose=0)
print(f"Train accuracy: {acc:.2%}")

