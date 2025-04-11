
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define a function to create a model with varying hyperparameters
def create_model(X_train, y_train, X_test, y_test, filters_1=32, filters_2=64, kernel_size=(3, 3), learning_rate=0.001,
                 batch_size=100, epochs=20):
    model = Sequential()

    model.add(Conv2D(filters=filters_1, kernel_size=kernel_size, activation='relu', kernel_initializer='he_uniform',
                     input_shape=X_train.shape[1:]))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=filters_2, kernel_size=kernel_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    # Compile model with given learning rate
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # Callbacks for early stopping and model checkpoint
    es = EarlyStopping(monitor='val_loss', patience=20, verbose=0)
    mc = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

    # Train model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                        callbacks=[es, mc], verbose=0)

    return model, history

# Define hyperparameter options
param_options = {
    'filters_1': [4, 8],
    'filters_2': [32, 64],
    'kernel_size': [(2, 2)],
    'learning_rate': [0.01, 0.02, 0.008],
    'batch_size': [16, 32],
    'epochs': [100]
}

# Number of random searches to perform
num_random_searches = 10  # You can adjust this value

# Store best model details
best_accuracy = 0
best_params = None
best_model = None

# Perform random search
for _ in range(num_random_searches):
    # Randomly select a combination of hyperparameters
    filters_1 = random.choice(param_options['filters_1'])
    filters_2 = random.choice(param_options['filters_2'])
    kernel_size = random.choice(param_options['kernel_size'])
    learning_rate = random.choice(param_options['learning_rate'])
    batch_size = random.choice(param_options['batch_size'])
    epochs = random.choice(param_options['epochs'])

    print(
        f"Testing: filters_1={filters_1}, filters_2={filters_2}, kernel_size={kernel_size}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")

    # Train model with selected parameters
    model, history = create_model(X_train, y_train, X_test, y_test, filters_1, filters_2, kernel_size, learning_rate,
                                  batch_size, epochs)

    # Evaluate model
    final_accuracy = max(history.history['val_accuracy'])  # Get the best validation accuracy

    print(f"Validation Accuracy: {final_accuracy:.4f}\n")

    # Check if it's the best model so far
    if final_accuracy > best_accuracy:
        best_accuracy = final_accuracy
        best_params = (filters_1, filters_2, kernel_size,
                       learning_rate, batch_size, epochs)
        best_model = model

# Print best model parameters
print("\nBest Model Parameters:", best_params)
print("Best Validation Accuracy:", best_accuracy)
