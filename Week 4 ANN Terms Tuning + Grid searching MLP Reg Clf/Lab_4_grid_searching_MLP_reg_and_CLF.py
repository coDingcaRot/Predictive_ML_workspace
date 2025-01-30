# example 1 visualizing loss and acc
def ex1_loss_and_acc_visual():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense

    PATH = "../Datasets/"
    df = pd.read_csv(PATH + 'iris_v2.csv')
    df.columns = ['Sepal L', 'Sepal W', 'Petal L', 'Petal W', 'Iris Type']
    print(df)

    # Convert text to numeric category.
    # 0 is setosa, 1 is versacolor and 2 is virginica
    df['y'] = LabelEncoder().fit_transform(df['Iris Type'])

    # Prepare the data.
    dfX = df.iloc[:, 0:4]  # Get X features only from columns 0 to 3
    dfY = df.iloc[:, 5:6]  # Get X features only from column 5

    ROW_DIM = 0
    COL_DIM = 1

    # Create vertical array of features.
    x_array = dfX.values
    x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
                                      x_array.shape[COL_DIM])

    y_array = dfY.values
    y_arrayReshaped = y_array.reshape(y_array.shape[ROW_DIM],
                                      y_array.shape[COL_DIM])

    # Split into train, validation and test data sets.
    X_train, X_temp, y_train, y_temp = train_test_split(
        x_arrayReshaped, y_arrayReshaped, test_size=0.33)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50)

    n_features = X_train.shape[COL_DIM]

    # Define the model.
    model = Sequential()

    # Hidden layer 1 (also receives the input layer)
    model.add(Dense(2, activation='relu', input_shape=(n_features,)))

    # Output layer
    model.add(Dense(3, activation='softmax'))

    # Compile the model.
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit the model.
    history = model.fit(X_train, y_train, epochs=1000, batch_size=28, verbose=1,
                        validation_data=(X_val, y_val))

    # Evaluate the model with unseen data.
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)

    # make a prediction
    row = [5.1, 3.5, 1.4, 0.2]
    import numpy as np
    rowArray = np.array(row).reshape(1, 4)
    yhat = model.predict(rowArray)

    import matplotlib.pyplot as plt
    def showLoss(history):
        # Get training and test loss histories
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 1)
        # Visualize loss history for training data.
        plt.plot(epoch_count, training_loss, label='Train Loss', color='red')

        # View loss on unseen data.
        plt.plot(epoch_count, validation_loss, 'r--', label='Validation Loss',
                 color='black')

        plt.xlabel('Epoch')
        plt.legend(loc="best")
        plt.title("Loss")

    def showAccuracy(history):
        # Get training and test loss histories
        training_loss = history.history['accuracy']
        validation_loss = history.history['val_accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 2)
        # Visualize loss history for training data.
        plt.plot(epoch_count, training_loss, label='Train Accuracy', color='red')

        # View loss on unseen data.
        plt.plot(epoch_count, validation_loss, 'r--',
                 label='Validation Accuracy', color='black')
        plt.xlabel('Epoch')
        plt.legend(loc="best")
        plt.title('Accuracy')

    plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    showLoss(history)
    showAccuracy(history)
    plt.show()

    from sklearn.metrics import classification_report
    # Provide detailed evaluation with unseen data.
    y_probability = model.predict(X_test)
    import numpy as np
    # Convert probability arrays to whole numbers.
    # eg. [0.0003, 0.01, 0.9807] becomes 2.
    predictions = np.argmax(y_probability, axis=-1)
    print(classification_report(y_test, predictions))
# ex1_loss_and_acc_visual()


# exericise 1
def exercise1():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    import numpy as np

    PATH = "../Datasets/"
    df = pd.read_csv(PATH + 'bill_authentication.csv')

    # Convert text to numeric category.
    # 0 is setosa, 1 is versacolor and 2 is virginica
    y = df['Class']
    X = df
    del X['Class']
    ROW_DIM = 0
    COL_DIM = 1

    # Create vertical array of features.
    x_array = X.values
    x_arrayReshaped = x_array.reshape(x_array.shape[ROW_DIM],
                                      x_array.shape[COL_DIM])

    y_array = np.array(y.values)
    y_arrayReshaped = y_array.reshape(len(y_array), 1)

    # Code from example 1 bottom half
    # Split into train, validation and test data sets.
    X_train, X_temp, y_train, y_temp = train_test_split(
        x_arrayReshaped, y_arrayReshaped, test_size=0.33)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50)

    n_features = X_train.shape[COL_DIM]

    # Define the model.
    model = Sequential()

    # Hidden layer 1 (also receives the input layer)
    model.add(Dense(2, activation='relu', input_shape=(n_features,)))
    # Output layer
    model.add(Dense(3, activation='softmax'))

    # Compile the model.
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fit the model.
    history = model.fit(X_train, y_train, epochs=1000, batch_size=28, verbose=1,
                        validation_data=(X_val, y_val))

    # Evaluate the model with unseen data.
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %.3f' % acc)

    # make a prediction
    row = [1, 1, 1, 0] #changed to work with binary outputs
    import numpy as np
    rowArray = np.array(row).reshape(1, 4)
    yhat = model.predict(rowArray)

    import matplotlib.pyplot as plt
    def showLoss(history):
        # Get training and test loss histories
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 1)
        # Visualize loss history for training data.
        plt.plot(epoch_count, training_loss, label='Train Loss', color='red')

        # View loss on unseen data.
        plt.plot(epoch_count, validation_loss, 'r--', label='Validation Loss',
                 color='black')

        plt.xlabel('Epoch')
        plt.legend(loc="best")
        plt.title("Loss")

    def showAccuracy(history):
        # Get training and test loss histories
        training_loss = history.history['accuracy']
        validation_loss = history.history['val_accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 2)
        # Visualize loss history for training data.
        plt.plot(epoch_count, training_loss, label='Train Accuracy', color='red')

        # View loss on unseen data.
        plt.plot(epoch_count, validation_loss, 'r--',
                 label='Validation Accuracy', color='black')
        plt.xlabel('Epoch')
        plt.legend(loc="best")
        plt.title('Accuracy')

    plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    showLoss(history)
    showAccuracy(history)
    plt.show()

    from sklearn.metrics import classification_report
    # Provide detailed evaluation with unseen data.
    y_probability = model.predict(X_test)
    import numpy as np
    # Convert probability arrays to whole numbers.
    # eg. [0.0003, 0.01, 0.9807] becomes 2.
    predictions = np.argmax(y_probability, axis=-1)
    print(classification_report(y_test, predictions))
# exercise1()

#example 2 predicting house price
def ex2_predicting_house_price():
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense
    import tensorflow as tf
    import matplotlib.pyplot as plt

    # Read the data.
    PATH = "../Datasets/"
    CSV_DATA = "USA_Housing.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Show all columns and data info.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.info())  # Check column types
    print(df.head())
    print(df.tail())
    print(df.describe())

    # Drop non-numeric columns if necessary (e.g., 'Address' in USA_Housing.csv)
    if 'Address' in df.columns:
        df = df.drop('Address', axis=1)

    # Convert the DataFrame to numpy arrays.
    dataset = df.values
    X = dataset[:, :-1]  # All columns except the last (features)
    y = dataset[:, -1]   # Last column (target)

    # Ensure all features are numeric.
    X = X.astype(float)
    y = y.astype(float)

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

    # Define the model.
    def baseline_model(input_dim):
        model = Sequential()
        model.add(Dense(25, input_dim=input_dim, kernel_initializer='uniform', activation='softplus'))
        model.add(Dense(10, kernel_initializer='lecun_uniform', activation='softplus'))
        model.add(Dense(1, kernel_initializer='uniform'))

        # Use Adam optimizer with the given learning rate
        opt = tf.keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer=opt, loss='mean_squared_error')
        return model

    # Build the model with the correct input dimension.
    input_dim = X_train.shape[1]
    model = baseline_model(input_dim)

    # Train the model.
    history = model.fit(X_train, y_train, epochs=100, batch_size=9, verbose=1, validation_data=(X_val, y_val))

    # Evaluate the model.
    predictions = model.predict(X_test)

    mse = metrics.mean_squared_error(y_test, predictions)
    print("Neural network MSE: " + str(mse))
    print("Neural network RMSE: " + str(np.sqrt(mse)))

    def showLoss(history):
        # Get training and test loss histories
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 1)
        # Visualize loss history for training data.
        plt.plot(epoch_count, training_loss, label='Train Loss', color='red')

        # View loss on unseen data.
        plt.plot(epoch_count, validation_loss, 'r--', label='Validation Loss',
                 color='black')

        plt.xlabel('Epoch')
        plt.legend(loc="best")
        plt.title("Loss")

    plt.plot()
    showLoss(history)
    plt.show()
# ex2_predicting_house_price()

# example 3 MLP classifier
def ex3_MLP_CLF():
    from sklearn import metrics
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    plt.style.use('ggplot')

    # Create numeric target for iris type.
    dataset = pd.read_csv('../Datasets/iris_v2.csv')
    dataset.iris_type = pd.Categorical(dataset.iris_type)

    # Prepare x and y.
    dataset['flowertype'] = dataset.iris_type.cat.codes
    del dataset['iris_type']
    y = dataset['flowertype']
    X = dataset
    del X['flowertype']

    # Split X and y.
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.30)
    # Scale X and Y.
    scX = StandardScaler()
    scalerX = scX.fit(X_train)
    trainX_scaled = scalerX.transform(X_train)
    testX_scaled = scalerX.transform(X_test)

    # Create and fit model.
    model = MLPClassifier()
    model.fit(trainX_scaled, y_train)
    print(model.get_params())  # Show model parameters.

    # Evaluate model.
    predicted_y = model.predict(testX_scaled)
    print(metrics.classification_report(y_test, predicted_y))
    print(metrics.confusion_matrix(y_test, predicted_y))

    def showLosses(model):
        plt.plot(model.loss_curve_)
        plt.title("Loss Curve")
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

    showLosses(model)
# ex3_MLP_CLF()

# example 4 GridSearchCV vs MLPClassifiers
def ex4_Gridsearching_MLPClf():
    from sklearn import metrics
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    plt.style.use('ggplot')

    # Create numeric target for iris type.
    dataset = pd.read_csv('../Datasets/iris_v2.csv')
    dataset.iris_type = pd.Categorical(dataset.iris_type)

    # Prepare x and y.
    dataset['flowertype'] = dataset.iris_type.cat.codes
    del dataset['iris_type']
    y = dataset['flowertype']
    X = dataset
    del X['flowertype']

    # Split X and y.
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.30)
    # Scale X and Y.
    scX = StandardScaler()
    scalerX = scX.fit(X_train)
    trainX_scaled = scalerX.transform(X_train)
    testX_scaled = scalerX.transform(X_test)

    # Create and fit model.
    model = MLPClassifier()
    model.fit(trainX_scaled, y_train)
    print(model.get_params())  # Show model parameters.

    # Evaluate model.
    predicted_y = model.predict(testX_scaled)
    print(metrics.classification_report(y_test, predicted_y))
    print(metrics.confusion_matrix(y_test, predicted_y))

    def showLosses(model):
        plt.plot(model.loss_curve_)
        plt.title("Loss Curve")
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

    showLosses(model)

    # Gridsearching MLP additional code
    parameters = {
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        'hidden_layer_sizes': [(200, 200), (300, 200), (150, 150)],
        'activation': ["logistic", "relu", "tanh"]
    }
    model2 = GridSearchCV(estimator=model, param_grid=parameters,
                          scoring='accuracy',  # average='macro'),
                          n_jobs=-1, cv=4, verbose=1,
                          return_train_score=False)

    model2.fit(trainX_scaled, y_train)
    print("Best parameters: ")
    print(model2.best_params_)
    y_pred = model2.predict(testX_scaled)

    print("Report with grid: ")
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))
    showLosses(model2.best_estimator_)
# ex4_Gridsearching_MLPClf()

# example 5 MLP regressors
def ex5_MLP_regressors():
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn import metrics
    import warnings
    warnings.filterwarnings(action='once')

    PATH = "../Datasets/"
    CSV_DATA = "USA_Housing.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Show all columns.
    pd.set_option('display.max_columns', None)

    # Increase number of columns that display on one line.
    pd.set_option('display.width', 1000)

    print(df.head())
    print(df.tail())
    print(df.describe())

    dataset = df.values

    # split into input (X) and output (Y) variables
    X = df.copy()
    y = X['Price']
    del X['Price']
    del X['Address']

    trainX, temp_X, trainY, temp_y = train_test_split(X, y, train_size=0.7)
    valX, testX, valY, testY = train_test_split(temp_X, temp_y, train_size=0.5)

    # Scale X and Y.
    scX = StandardScaler()
    scalerX = scX.fit(trainX)
    trainX_scaled = scalerX.transform(trainX)
    valX_scaled = scalerX.transform(valX)
    testX_scaled = scalerX.transform(testX)

    scY = StandardScaler()
    trainY_scaled = scY.fit_transform(np.array(trainY).reshape(-1, 1))
    testY_scaled = scY.transform(np.array(testY).reshape(-1, 1))
    valY_scaled = scY.transform(np.array(valY).reshape(-1, 1))

    # Build basic multilayer perceptron.
    model1 = MLPRegressor(
        # 3 hidden layers with 150 neurons, 100, and 50.
        hidden_layer_sizes=(150, 100, 50),
        max_iter=50,  # epochs
        activation='relu',
        solver='adam',  # optimizer
        verbose=1)
    model1.fit(trainX_scaled, trainY_scaled)

    def showLosses(model):
        plt.plot(model.loss_curve_)
        plt.title("Loss Curve")
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

    def evaluateModel(model, testX_scaled, testY_scaled, scY):
        showLosses(model)
        scaledPredictions = model.predict(testX_scaled)
        y_pred = scY.inverse_transform(
            np.array(scaledPredictions).reshape(-1, 1))
        mse = metrics.mean_squared_error(testY_scaled, y_pred)
        rmse = np.sqrt(mse)
        print("RMSE: " + str(rmse))

    evaluateModel(model1, valX_scaled, valY_scaled, scY)

    # here is the new part.
    param_grid = {
        'hidden_layer_sizes': [(150, 100, 50), (120, 80, 40), (100, 50, 30)],
        'max_iter': [50, 100],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
# ex5_MLP_regressors()

# example 6 grid searching MLP Regressor
def ex6_grid_searching_MLP_regs():
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn import metrics
    import warnings
    warnings.filterwarnings(action='once')

    PATH = "../Datasets/"
    CSV_DATA = "USA_Housing.csv"
    df = pd.read_csv(PATH + CSV_DATA)

    # Show all columns.
    pd.set_option('display.max_columns', None)

    # Increase number of columns that display on one line.
    pd.set_option('display.width', 1000)

    print(df.head())
    print(df.tail())
    print(df.describe())

    dataset = df.values

    # split into input (X) and output (Y) variables
    X = df.copy()
    y = X['Price']
    del X['Price']
    del X['Address']

    print(y)

    trainX, temp_X, trainY, temp_y = train_test_split(X, y, train_size=0.7)
    valX, testX, valY, testY = train_test_split(temp_X, temp_y, train_size=0.5)

    # Scale X and Y.
    scX = StandardScaler()
    scalerX = scX.fit(trainX)
    trainX_scaled = scalerX.transform(trainX)
    valX_scaled = scalerX.transform(valX)
    testX_scaled = scalerX.transform(testX)

    scY = StandardScaler()
    trainY_scaled = scY.fit_transform(np.array(trainY).reshape(-1, 1)).ravel()  # Flatten the array
    testY_scaled = scY.transform(np.array(testY).reshape(-1, 1)).ravel()  # Flatten the array
    valY_scaled = scY.transform(np.array(valY).reshape(-1, 1)).ravel()  # Flatten the array

    # Build basic multilayer perceptron.
    model1 = MLPRegressor(
        # 3 hidden layers with 150 neurons, 100, and 50.
        hidden_layer_sizes=(150, 100, 50),
        max_iter=230,  # epochs stops at this point
        activation='relu',
        solver='adam',  # optimizer
        verbose=1)
    model1.fit(trainX_scaled, trainY_scaled)

    def showLosses(model):
        plt.plot(model.loss_curve_)
        plt.title("Loss Curve")
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()
    def evaluateModel(model, testX_scaled, testY_scaled, scY):
        print("Evaluating model")
        showLosses(model)
        scaledPredictions = model.predict(testX_scaled)
        y_pred = scY.inverse_transform(
            np.array(scaledPredictions).reshape(-1, 1))
        mse = metrics.mean_squared_error(testY_scaled, y_pred)
        rmse = np.sqrt(mse)
        print("RMSE: " + str(rmse))
    evaluateModel(model1, valX_scaled, valY_scaled, scY)

    # here is the new part.
    param_grid = {
        'hidden_layer_sizes': [(150, 100, 50), (120, 80, 40), (100, 50, 30)],
        'max_iter': [200, 250, 300],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2]
    }

    from sklearn.model_selection import GridSearchCV

    # n_jobs=-1 means use all processors.
    # Run print(metrics.get_scorer_names()) for scoring choices.
    model2 = MLPRegressor()
    gridModel = GridSearchCV(model2, param_grid, n_jobs=-1, cv=10,
                             scoring='neg_mean_squared_error')
    gridModel.fit(trainX_scaled, trainY_scaled)

    print("Best parameters")
    print(gridModel.best_params_)

    evaluateModel(gridModel.best_estimator_, valX_scaled, valY, scY)

    # Evaluate both models with test (unseen) data.
    print("\n*** Base model with test data: ")
    evaluateModel(model1, testX_scaled, testY, scY)
    print(model1.get_params())
    print("\n*** Grid searched model with test data: ")
    evaluateModel(gridModel.best_estimator_, testX_scaled, testY, scY)
    print(gridModel.get_params())
# ex6_grid_searching_MLP_regs()
