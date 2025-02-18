def ex1_under_performing_tanh_func_network():
    # deeper mlp with tanh for the two circles classification problem
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import MinMaxScaler
    from keras.layers import Dense
    from keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    from keras.initializers import RandomUniform
    import matplotlib.pyplot as plt

    # Generate 2d classification dataset.
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    # split into train and test
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    # Define the model.
    model = Sequential()
    init = 'he_uniform'

    model.add(Dense(5, input_dim=2, activation='sigmoid', kernel_initializer=init))
    model.add(Dense(5, activation='sigmoid', kernel_initializer=init))
    model.add(Dense(5, activation='sigmoid', kernel_initializer=init))
    model.add(Dense(5, activation='sigmoid', kernel_initializer=init))
    model.add(Dense(5, activation='sigmoid', kernel_initializer=init))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model.
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=1)

    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))

    # Plot loss learning curves.
    plt.subplot(211)
    plt.title('Loss', pad=-40)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

    # Plot accuracy learning curves.
    plt.subplot(212)
    plt.title('Accuracy', pad=-40)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()
    plt.show()
# ex1_underperforming_tanh_func_network()

# creating a deep network using reLu. Better results than tanh
def ex2_reLu_deeper_networks():
    # deeper mlp with tanh for the two circles classification problem
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import MinMaxScaler
    from keras.layers import Dense
    from keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    from keras.initializers import RandomUniform
    import matplotlib.pyplot as plt

    # Generate 2d classification dataset.
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    # split into train and test
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    # Define the model.
    model = Sequential()
    init = 'he_uniform'
    model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer=init))
    model.add(Dense(5, activation='relu', kernel_initializer=init))
    model.add(Dense(5, activation='relu', kernel_initializer=init))
    model.add(Dense(5, activation='relu', kernel_initializer=init))
    model.add(Dense(5, activation='relu', kernel_initializer=init))
    model.add(Dense(1, activation='sigmoid'))  # output layer

    # Compile model.
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=1)

    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))

    # Plot loss learning curves.
    plt.subplot(211)
    plt.title('Loss', pad=-40)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

    # Plot accuracy learning curves.
    plt.subplot(212)
    plt.title('Accuracy', pad=-40)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()
    plt.show()
# ex2_reLu_deeper_networks()

# To eliminate overall vanishing gradients. But relu performs just as well if not better
def ex3_leaky_reLu():
    # deeper mlp with tanh for the two circles classification problem
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import MinMaxScaler
    from keras.layers import Dense
    from keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    from keras.initializers import RandomUniform
    import matplotlib.pyplot as plt

    # Generate 2d classification dataset.
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    # split into train and test
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    import tensorflow as tf
    leakyReLU = tf.keras.layers.LeakyReLU(alpha=0.3)

    # Define the model.
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation=leakyReLU, kernel_initializer='he_uniform'))
    model.add(Dense(5, activation=leakyReLU, kernel_initializer='he_uniform'))
    model.add(Dense(5, activation=leakyReLU, kernel_initializer='he_uniform'))
    model.add(Dense(5, activation=leakyReLU, kernel_initializer='he_uniform'))
    model.add(Dense(5, activation=leakyReLU, kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model.
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=1)

    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))

    # Plot loss learning curves.
    plt.subplot(211)
    plt.title('Loss', pad=-40)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

    # Plot accuracy learning curves.
    plt.subplot(212)
    plt.title('Accuracy', pad=-40)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()
    plt.show()
# ex3_leaky_reLu()

#uses early stopping and model save to fix overfitting
# example 4,5
# example 1,2,3
def ex4_overfitting_fix():
    # Creating an overfit situation with the moons data set.
    from sklearn.datasets import make_moons
    from keras.layers import Dense
    from keras.models import Sequential
    import matplotlib.pyplot as plt

    # Generate 2d classification dataset.
    X, y = make_moons(n_samples=100, noise=0.2, random_state=1)

    # Split data into train and test.
    n_train = 30
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    # Define the model.
    model = Sequential()
    model.add(Dense(500, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model. THIS CAUSES overfitting.
    # history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=1)

    # Fit the model. THIS IS THE FIX ###################################
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint
    from keras.models import load_model

    # simple early stopping
    # patience:  # of epochs observed where no improvement before exiting.
    # mode:      Could be max, min, or auto.
    # min_delta: Amount of change needed to be considered an improvement.
    # monitor: val_accuracy, val_loss can monitor loss or acc
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, min_delta=0.000001, patience=0)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1,
                         save_best_only=True)

    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0,
                        callbacks=[es, mc])

    # load the saved model
    model = load_model('best_model.h5')
    ##### THIS IS THE FIX ###################################

    # Evaluate the model.
    train_loss, train_acc = model.evaluate(trainX, trainy, verbose=0)
    test_loss, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_acc, test_acc))
    print('Train loss: %.3f, Test loss: %.3f' % (train_loss, test_loss))

    # Plot loss learning curves.
    plt.subplot(211)
    plt.title('Cross-Entropy Loss', pad=-40)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

    # Plot accuracy learning curves.
    plt.subplot(212)
    plt.title('Accuracy', pad=-40)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()
# ex4_overfitting_fix()

# unscaled data
# example 6, 7, 8
def ex6_bad_exploding_gradients():
    # mlp with unscaled data for the regression problem
    from sklearn.datasets import make_regression
    from keras.layers import Dense
    from keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    import matplotlib.pyplot as plt

    # Generate the regression dataset.
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

    plt.hist(y)
    plt.title("Unscaled Input")
    plt.show()

    # Split into train and test.
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    # Define the model.
    model = Sequential()
    model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear'))

    # Compile the model. replace to test it out without scaling and scaling using clipnorm
    # model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01, momentum=0.9))

    # # example 7 added
    # opt = SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0)
    # model.compile(loss='mean_squared_error', optimizer=opt)

    # example 8 gradient clipping or norm clipping
    opt = SGD(learning_rate=0.01, momentum=0.9, clipvalue=5.0)
    model.compile(loss='mean_squared_error', optimizer=opt)

    # Fit the model.
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=1)

    # Evaluate the model.
    train_mse = model.evaluate(trainX, trainy, verbose=0)
    test_mse = model.evaluate(testX, testy, verbose=0)
    print('Train MSE: %.3f, Test MSE: %.3f' % (train_mse, test_mse))

    # Plot losses during training.
    plt.title('Losses')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
# ex6_bad_exploding_gradients()

# testing out different clipping values or ranges to see what has the best outcome?
# exercise 4,5 clip value testing as well
def exercise4_clip_norm_testing():
    # mlp with unscaled data for the regression problem
    from sklearn.datasets import make_regression
    from keras.layers import Dense
    from keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    import matplotlib.pyplot as plt

    # Generate the regression dataset.
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

    plt.hist(y)
    plt.title("Unscaled Input")
    plt.show()

    # Split into train and test.
    n_train = 500
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    clipResults = []

    # def buildModel(clipSize):
    #     # Define the model.
    #     model = Sequential()
    #     model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
    #     model.add(Dense(1, activation='linear'))
    #
    #     # Compile the model.
    #     opt = SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0)
    #     model.compile(loss='mean_squared_error', optimizer=opt)
    #
    #     # Fit the model.
    #     history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=1)
    #
    #     # Evaluate the model.
    #     train_mse = model.evaluate(trainX, trainy, verbose=0)
    #     test_mse = model.evaluate(testX, testy, verbose=0)
    #     print('Train MSE: %.3f, Test MSE: %.3f' % (train_mse, test_mse))
    #     clipResults.append({'train mse': train_mse, 'test mse': test_mse,
    #                         'clip size': clipSize})
    #     # Plot losses during training.
    #     plt.title('Losses')
    #     plt.plot(history.history['loss'], label='train')
    #     plt.plot(history.history['val_loss'], label='test')
    #     plt.legend()
    #     plt.show()

    # Clip value grid search
    def buildModel(clipValue):
        # Define the model.
        model = Sequential()
        model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='linear'))

        # Compile the model.
        opt = SGD(learning_rate=0.01, momentum=0.9, clipvalue=clipValue)
        model.compile(loss='mean_squared_error', optimizer=opt)

        # Fit the model.
        history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=1)

        # Evaluate the model.
        train_mse = model.evaluate(trainX, trainy, verbose=0)
        test_mse = model.evaluate(testX, testy, verbose=0)
        print('Train MSE: %.3f, Test MSE: %.3f' % (train_mse, test_mse))
        clipResults.append({'train mse': train_mse, 'test mse': test_mse, 'clip value: ': clipValue})
        # Plot losses during training.
        plt.title('Losses')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    clipSizes = [0.9, 1, 1.1, 2]
    clipValue = [-0.5, -0.1, 0.1, 0.5]
    for i in range(0, len(clipValue)):
        buildModel(clipValue[i])

    for clipResult in clipResults:
        print(clipResult)
# exercise4_clip_norm_testing()

# crashes due to input having nan values
def ex9_unscaled_data():
    from sklearn.datasets import make_regression
    from keras.layers import Dense
    from keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    from matplotlib import pyplot
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Generate regression set.
    X, y = make_regression(n_samples=1000, n_features=20,
                           noise=0.1, random_state=1)

    # Split data.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.60,
                                                        random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.50,
                                                    random_state=0)

    normSizeEvaluations = []

    # Define the model.
    model = Sequential()
    model.add(Dense(25, input_dim=20, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear'))

    # Compile the model.
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(learning_rate=0.01, momentum=0.9))

    # Fit the model.
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=200, verbose=1)

    predictions = model.predict(X_test)

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(predictions, y_test)
    rmse = np.sqrt(mse)
    print('Test RMSE: ' + str(rmse))

    # Plot the loss during training.
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
# ex9_unscaled_data()

# outputs a value as the data is scaled and filled?
def ex10_input_output_scaling():
    from sklearn.datasets import make_regression
    from keras.layers import Dense
    from keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    from matplotlib import pyplot
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Generate regression set.
    X, y = make_regression(n_samples=1000, n_features=20,
                           noise=0.1, random_state=1)

    # Split data.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.60,
                                                        random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.50,
                                                    random_state=0)

    normSizeEvaluations = []

    # Define the model.
    model = Sequential()
    model.add(Dense(25, input_dim=20, activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear'))

    # Compile the model.
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(learning_rate=0.01, momentum=0.9))

    from sklearn.preprocessing import StandardScaler

    # reshape 1d arrays to 2d arrays
    reshapedTrainY = y_train.reshape(len(y_train), 1)
    reshapedValY = y_val.reshape(len(y_val), 1)

    # Scale y
    yscaler = StandardScaler()
    yscaler.fit(reshapedTrainY)
    scaled_Y_train = yscaler.transform(reshapedTrainY)
    scaled_Y_val = yscaler.transform(reshapedValY)

    # Scale x
    xscaler = StandardScaler()
    xscaler.fit(X_train)
    scaled_X_Train = xscaler.transform(X_train)
    scaled_Y_val = xscaler.transform(X_val)
    scaled_X_test = xscaler.transform(X_test)

    # Fit the model.
    history = model.fit(scaled_X_Train, scaled_Y_train,
                        validation_data=(scaled_Y_val, scaled_Y_val),
                        epochs=200, verbose=1)

    scaledPredictions = model.predict(scaled_X_test)
    unscaledPredictions = yscaler.inverse_transform(np.array(scaledPredictions))

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(unscaledPredictions, y_test)
    rmse = np.sqrt(mse)
    print('Test RMSE: ' + str(rmse))

    # Plot the loss during training.
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
# ex10_input_output_scaling()

# greedy layer-wise pre training data setup
def ex11_data_setup():
    # scatter plot of blobs dataset
    from sklearn.datasets import make_blobs
    from matplotlib import pyplot
    from numpy import where
    # generate 2d classification dataset
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
    # scatter plot for each class value
    for class_value in range(3):
        # select indices of points with the class label
        row_ix = where(y == class_value)
        # scatter plot for points with a different color
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
        # show plot
    pyplot.show()
# ex11_data_setup()

def ex12_greedy_layer_wise_pretrain():
    from sklearn.datasets import make_blobs
    from keras.layers import Dense
    from keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.utils import to_categorical
    import matplotlib.pyplot as plt

    # Generate the data.
    def prepare_data():
        X, y = make_blobs(n_samples=1000, centers=3,
                          n_features=2, cluster_std=2, random_state=2)
        y = to_categorical(y)
        n_train = 500
        trainX, testX = X[:n_train, :], X[n_train:, :]
        trainy, testy = y[:n_train], y[n_train:]
        return trainX, testX, trainy, testy

    # Build the base model.
    def get_base_model(trainX, trainy):
        # Define the model.
        model = Sequential()
        model.add(Dense(10, input_dim=2, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(3, activation='softmax'))

        # Compile the model.
        opt = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=opt,
                      metrics=['accuracy'])

        # Fit the model.
        model.fit(trainX, trainy, epochs=100, verbose=0)
        return model

    stats = []

    # Evaluate the model.
    def evaluate_model(numLayers, model, trainX, testX, trainy, testy):
        train_loss, train_acc = model.evaluate(trainX, trainy, verbose=1)
        test_loss, test_acc = model.evaluate(testX, testy, verbose=1)
        stats.append({'# layers': numLayers, 'train_acc': train_acc, 'test_acc': test_acc,
                      'train_loss': train_loss, 'test_loss': test_loss})

    # Add one new layer and re-train only the new layer.
    def add_layer(model, trainX, trainy):
        # Store the output layer.
        output_layer = model.layers[-1]

        # Remove the output layer.
        model.pop()

        # Mark all remaining layers as non-trainable.
        for layer in model.layers:
            layer.trainable = False

        # Add a new hidden layer.
        model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))

        # Add the output layer back.
        model.add(output_layer)

        # fit model
        model.fit(trainX, trainy, epochs=100, verbose=1)
        return model

    # Get the data and build the base model.
    trainX, testX, trainy, testy = prepare_data()
    model = get_base_model(trainX, trainy)

    # Evaluate the base model
    scores = dict()
    evaluate_model(-1, model, trainX, testX, trainy, testy)

    # add layers and evaluate the updated model
    n_layers = 10
    for i in range(n_layers):
        model = add_layer(model, trainX, trainy)
        evaluate_model(i, model, trainX, testX, trainy, testy)

    import pandas as pd
    columns = ['# layers', 'train_acc', 'test_acc', 'train_loss', 'test_loss']

    df = pd.DataFrame(columns=columns)
    for i in range(0, len(stats)):
        df = df._append(stats[i], ignore_index=True)
    print(df)
# ex12_greedy_layer_wise_pretrain()

def exercise6_glw_pretrain():
    from sklearn.datasets import make_blobs
    from keras.layers import Dense
    from keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.utils import to_categorical
    import matplotlib.pyplot as plt

    # Generate the data.
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    import pandas as pd
    def prepare_data():
        PATH = "../Datasets/"
        # load the dataset
        df = pd.read_csv(PATH + 'diabetes.csv', sep=',')

        # split into input (X) and output (y) variables
        X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age']]
        y = df[['Outcome']]

        # Split into train and test data sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        return X_train, X_test, y_train, y_test

    # Build the base model.
    def get_base_model(trainX, trainy):
        # define the keras model
        model = Sequential()
        model.add(Dense(230, input_dim=8, activation='relu',
                        kernel_initializer='he_normal'))

        model.add(Dense(1, activation='sigmoid'))
        opitimizer = tf.keras.optimizers.SGD(
            learning_rate=0.0005, momentum=0.9, name="SGD",
        )

        # Compile the keras model.
        model.compile(loss='binary_crossentropy', optimizer=opitimizer,
                      metrics=['accuracy'])

        # Fit the keras model on the dataset.
        model.fit(trainX, trainy, epochs=200, batch_size=10)

        return model

    stats = []

    # Evaluate the model.
    def evaluate_model(numLayers, model, trainX, testX, trainy, testy):
        train_loss, train_acc = model.evaluate(trainX, trainy, verbose=1)
        test_loss, test_acc = model.evaluate(testX, testy, verbose=1)
        stats.append({'# layers': numLayers, 'train_acc': train_acc, 'test_acc': test_acc,
                      'train_loss': train_loss, 'test_loss': test_loss})

    # Add one new layer and re-train only the new layer.
    def add_layer(model, trainX, trainy):
        # Store the output layer.
        output_layer = model.layers[-1]

        # Remove the output layer.
        model.pop()

        # Mark all remaining layers as non-trainable.
        for layer in model.layers:
            layer.trainable = False

        # Add a new hidden layer.
        model.add(Dense(230, activation='relu', kernel_initializer='he_uniform'))

        # Add the output layer back.
        model.add(output_layer)

        # fit model
        model.fit(trainX, trainy, epochs=300, verbose=1)
        return model

    # Get the data and build the base model.
    trainX, testX, trainy, testy = prepare_data()
    model = get_base_model(trainX, trainy)

    # Evaluate the base model
    scores = dict()
    evaluate_model(-1, model, trainX, testX, trainy, testy)

    # add layers and evaluate the updated model
    n_layers = 14
    for i in range(n_layers):
        model = add_layer(model, trainX, trainy)
        evaluate_model(i, model, trainX, testX, trainy, testy)

    columns = ['# layers', 'train_acc', 'test_acc', 'train_loss', 'test_loss']

    df = pd.DataFrame(columns=columns)
    for i in range(0, len(stats)):
        df = df._append(stats[i], ignore_index=True)
    print(df)
exercise6_glw_pretrain()
