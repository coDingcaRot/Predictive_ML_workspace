#Lesson 5 6 7 8 Errors replace the code with the given in .txt
#lr should be learning_rate. lr causes errors
import pandas as pd
import numpy as np
from numexpr.necompiler import evaluate_lock
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

#example 1 baseline regression model base line
#RMSE 5.2133376653973365
def ex1_building_baseline_model():
    PATH = "../Datasets/"
    CSV_DATA = "housing.data"

    df = pd.read_csv(PATH + CSV_DATA, header=None)

    # Show all columns on one line.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(df.head())
    print(df.tail())
    print(df.describe())

    dataset = df.values

    # Split data into input (X) and output (Y) variables.
    X = dataset[:, 0:13]
    y = dataset[:, 13]

    # Adding an intercept *** This is required ***. Don't forget this step.
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y, test_size=0.3, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                    y_temp, test_size=0.5, random_state=0)

    # Make predictions and evaluate with the RMSE.
    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)
    print(model.summary())
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# ex1_building_baseline_model()

# example 2 Baseline neural network
# Neural network MSE: 28.267281534825
# Neural network RMSE: 5.316698367861863
def ex2_building_baseline_ANN():
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense

    # Read the data.
    PATH = "../Datasets/"
    CSV_DATA = "housing.data"
    df = pd.read_csv(PATH + CSV_DATA, header=None)

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head())
    print(df.tail())
    print(df.describe())

    # Convert DataFrame columns to vertical columns so they can be used by the NN.
    dataset = df.values
    X = dataset[:, 0:13]  # Columns 0 to 12
    y = dataset[:, 13]  # Columns 13
    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
    y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(x_arrayReshaped,
                                                        y_arrayReshaped, test_size=0.3, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                    y_temp, test_size=0.5, random_state=0)

    # Define the model.
    def create_model():
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    model = create_model()

    # Build the model.
    model = create_model()
    history = model.fit(X_train, y_train, epochs=100,
                        batch_size=5, verbose=1,
                        validation_data=(X_val, y_val))

    # Evaluate the model.
    predictions = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, predictions)
    print("Neural network MSE: " + str(mse))
    print("Neural network RMSE: " + str(np.sqrt(mse)))
# ex2_building_baseline_ANN()

# example 3 Epoch and batch tuning
# best epochs and batch
#   RMSE     epochs   batch
# 4.821016     200     10
def ex3_manual_epoch_batch_tuning():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense

    PATH = "../Datasets/"
    CSV_DATA = "housing.data"
    df = pd.read_csv(PATH + CSV_DATA, header=None)

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(df.head())
    print(df.tail())
    print(df.describe())

    dataset = df.values

    # split into input (X) and output (Y) variables
    X = dataset[:, 0:13]
    y = dataset[:, 13]

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y, test_size=0.3, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                    y_temp, test_size=0.5, random_state=0)

    def evaluateModel(model, X_test, y_test):
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print("RMSE: " + str(rmse))
        return rmse

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=['rmse'])
        print(dfStats)

    networkStats = []

    ### Model parameters ############################
    batch_sizes = [10, 60, 100]
    epochList = [50, 100, 200]

    #################################################

    #################################################
    # Build model
    def create_model():
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    for batch_size in batch_sizes:
        for epochs in epochList:
            model = create_model()
            history = model.fit(X_train, y_train, epochs=epochs,
                                batch_size=batch_size, verbose=1,
                                validation_data=(X_val, y_val))
            rmse = evaluateModel(model, X_test, y_test)
            networkStats.append({"rmse": rmse, "epochs": epochs, "batch": batch_size})
    showResults(networkStats)
    #################################################
# ex3_manual_epoch_batch_tuning()

# example 4 searching efficient optimizers
def ex4_determine_efficient_optimizer():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense

    PATH = "../Datasets/"
    CSV_DATA = "housing.data"
    df = pd.read_csv(PATH + CSV_DATA, header=None)

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(df.head())
    print(df.tail())
    print(df.describe())

    dataset = df.values

    # split into input (X) and output (Y) variables
    X = dataset[:, 0:13]
    y = dataset[:, 13]

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y, test_size=0.3, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                    y_temp, test_size=0.5, random_state=0)

    def evaluateModel(model, X_test, y_test):
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print("RMSE: " + str(rmse))
        return rmse

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=['rmse'])
        print(dfStats)

    networkStats = []

    ### Model parameters ############################
    optimizers = ['SGD', 'RMSprop', 'Adagrad',
                  'Adadelta', 'Adam', 'Adamax', 'Nadam']
    #################################################

    #################################################
    # Build model
    def create_model(optimizer="SGD"):
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    for optimizer in optimizers:
        BATCH_SIZE = 10
        EPOCHS = 100
        model = create_model(optimizer)
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, verbose=1,
                            validation_data=(X_val, y_val))
        rmse = evaluateModel(model, X_test, y_test)
        networkStats.append({"rmse": rmse, "optimizer": optimizer})
    showResults(networkStats)
    #################################################
# ex4_determine_efficient_optimizer()

# exmaple 5 Optimizing learning rate of optimizer
def ex5_optimizing_optimizer():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense

    PATH = "../Datasets/"
    CSV_DATA = "housing.data"
    df = pd.read_csv(PATH + CSV_DATA, header=None)

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(df.head())
    print(df.tail())
    print(df.describe())

    dataset = df.values

    # split into input (X) and output (Y) variables
    X = dataset[:, 0:13]
    y = dataset[:, 13]

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y, test_size=0.3, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                    y_temp, test_size=0.5, random_state=0)

    def evaluateModel(model, X_test, y_test):
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print("RMSE: " + str(rmse))
        return rmse

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=['rmse'])
        print(dfStats)

    networkStats = []

    ### Model parameters ############################
    learningRates = [0.001, 0.005, 0.01, 0.015, 0.2]
    #################################################

    #################################################
    # Build model
    import tensorflow as tf

    def create_model(learningRate=0.001):
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))

        # Use Adam optimizer with the given learning rate
        optimizer = tf.keras.optimizers.Adam(lr=learningRate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    for learningRate in learningRates:
        BATCH_SIZE = 10
        EPOCHS = 100
        model = create_model(learningRate)
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, verbose=1,
                            validation_data=(X_val, y_val))
        rmse = evaluateModel(model, X_test, y_test)
        networkStats.append({"rmse": rmse, "learningRate": learningRate})
    showResults(networkStats)

# exercise 2,3, 4 change kernel initializer
def modified_ex2_building_baseline_ANN():
    import pandas as pd
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense
    import tensorflow as tf

    # Read the data.
    PATH = "../Datasets/"
    CSV_DATA = "housing.data"
    df = pd.read_csv(PATH + CSV_DATA, header=None)

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head())
    print(df.tail())
    print(df.describe())

    # Convert DataFrame columns to vertical columns so they can be used by the NN.
    dataset = df.values
    X = dataset[:, 0:13]  # Columns 0 to 12
    y = dataset[:, 13]  # Columns 13
    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
    y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(x_arrayReshaped,
                                                        y_arrayReshaped, test_size=0.3, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                    y_temp, test_size=0.5, random_state=0)

    # Define the model.
    def create_model():
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer='uniform',
                        activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    model = create_model()

    # Build the model.
    model = create_model()
    history = model.fit(X_train, y_train, epochs=100,
                        batch_size=10, verbose=1,
                        validation_data=(X_val, y_val))

    # Evaluate the model.
    predictions = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, predictions)
    print("Neural network MSE: " + str(mse))
    print("Neural network RMSE: " + str(np.sqrt(mse)))
# modified_ex2_building_baseline_ANN()

# example 6
def ex6_grid_search_kernel_initializer():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense

    PATH = "../Datasets/"
    CSV_DATA = "housing.data"
    df = pd.read_csv(PATH + CSV_DATA, header=None)

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(df.head())
    print(df.tail())
    print(df.describe())

    dataset = df.values

    # split into input (X) and output (Y) variables
    X = dataset[:, 0:13]
    y = dataset[:, 13]

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y, test_size=0.3, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                    y_temp, test_size=0.5, random_state=0)

    def evaluateModel(model, X_test, y_test):
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print("RMSE: " + str(rmse))
        return rmse

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=['rmse'])
        print(dfStats)

    networkStats = []

    ### Model parameters ############################
    init_modes = ['uniform', 'lecun_uniform', 'normal', 'zero',
                  'glorot_normal',
                  'glorot_uniform', 'he_normal', 'he_uniform']
    #################################################

    #################################################
    # Build model
    import tensorflow as tf
    def create_model(inialtizer='normal'):
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer=inialtizer,
                        activation='relu'))
        model.add(Dense(1, kernel_initializer=inialtizer))

        # Use Adam optimizer with the given learning rate
        LEARNING_RATE = 0.005
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    for initializer in init_modes:
        BATCH_SIZE = 10
        EPOCHS = 100
        model = create_model(initializer)
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, verbose=1,
                            validation_data=(X_val, y_val))
        rmse = evaluateModel(model, X_test, y_test)
        networkStats.append({"rmse": rmse, "initializer": initializer})
    showResults(networkStats)
    #################################################
# ex6_grid_search_kernel_initializer()

# example 7 neuron training
#        rmse  # neurons
# 2  4.664328         50
# 3  4.919369        100
# 4  4.920458        150
# 1  5.072077         25
# 0  6.994214          5
def ex7_tuning_neurons():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense

    PATH = "../Datasets/"
    CSV_DATA = "housing.data"
    df = pd.read_csv(PATH + CSV_DATA, header=None)

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(df.head())
    print(df.tail())
    print(df.describe())

    dataset = df.values

    # split into input (X) and output (Y) variables
    X = dataset[:, 0:13]
    y = dataset[:, 13]

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y, test_size=0.3, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                    y_temp, test_size=0.5, random_state=0)

    def evaluateModel(model, X_test, y_test):
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print("RMSE: " + str(rmse))
        return rmse

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=['rmse'])
        print(dfStats)

    networkStats = []

    ### Model parameters ############################
    neuronList = [5, 25, 50, 100, 150]
    #################################################

    #################################################
    # Build model
    import tensorflow as tf
    def create_model(numNeurons):
        model = Sequential()
        model.add(Dense(numNeurons,
                        input_dim=13, kernel_initializer='uniform',
                        activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform'))

        # Use Adam optimizer with the given learning rate
        LEARNING_RATE = 0.005
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    for numNeurons in neuronList:
        BATCH_SIZE = 10
        EPOCHS = 100
        model = create_model(numNeurons)
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, verbose=1,
                            validation_data=(X_val, y_val))
        rmse = evaluateModel(model, X_test, y_test)
        networkStats.append({"rmse": rmse, "# neurons": numNeurons})
    showResults(networkStats)
    #################################################
# ex7_tuning_neurons()

# example 8 adding another layer
#        rmse  # additional layers
# 1  5.254349                    1
# 2  5.313172                    2
# 0  5.319287                    0
# 3  5.467287                    3
# 4  5.506448                    4
# 5  5.935660                    5
def ex8_adding_another_layer():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from keras.models import Sequential
    from keras.layers import Dense

    PATH = "../Datasets/"
    CSV_DATA = "housing.data"
    df = pd.read_csv(PATH + CSV_DATA, header=None)

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(df.head())
    print(df.tail())
    print(df.describe())

    dataset = df.values

    # split into input (X) and output (Y) variables
    X = dataset[:, 0:13]
    y = dataset[:, 13]

    # Split the data.
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y, test_size=0.3, random_state=0)
    X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                    y_temp, test_size=0.5, random_state=0)

    def evaluateModel(model, X_test, y_test):
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print("RMSE: " + str(rmse))
        return rmse

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=['rmse'])
        print(dfStats)

    networkStats = []

    ### Model parameters ############################
    additionalLayers = [0, 1, 2, 3, 4, 5]
    #################################################

    #################################################
    # Build model
    import tensorflow as tf
    def create_model(numExtraLayers):
        NUM_NEURONS = 25
        model = Sequential()
        model.add(Dense(NUM_NEURONS,
                        input_dim=13, kernel_initializer='uniform',
                        activation='relu'))
        for i in range(0, numExtraLayers):
            # You could further grid search initializer, num_neurons
            # and activation function for each layer if desired.
            model.add(Dense(NUM_NEURONS, kernel_initializer='uniform',
                            activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform'))

        # Use Adam optimizer with the given learning rate
        LEARNING_RATE = 0.005
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    for numLayers in additionalLayers:
        BATCH_SIZE = 10
        EPOCHS = 100
        model = create_model(numLayers)
        history = model.fit(X_train, y_train, epochs=EPOCHS,
                            batch_size=BATCH_SIZE, verbose=1,
                            validation_data=(X_val, y_val))
        rmse = evaluateModel(model, X_test, y_test)
        networkStats.append({"rmse": rmse, "# additional layers": numLayers})
    showResults(networkStats)
    #################################################
# ex8_adding_another_layer()

# example 9 grid searching batches and epochs and seeing precision, f1 and epoch recall, exercise 6
#     precision    recall        f1  epochs  batch
# 18   0.909091  0.434783  0.588235      50     90
# 24   0.666667  0.608696  0.636364      50    100
# 31   0.933333  0.608696  0.736842     100    120
# 14   0.941176  0.695652  0.800000     150     60
# 19   0.941176  0.695652  0.800000     100     90
# 20   0.941176  0.695652  0.800000     150     90
# 30   0.941176  0.695652  0.800000      50    120
# 25   0.941176  0.695652  0.800000     100    100
# 32   0.941176  0.695652  0.800000     150    120
# 13   0.944444  0.739130  0.829268     100     60
# 7    0.944444  0.739130  0.829268     100     30
# 21   1.000000  0.739130  0.850000     200     90
# 26   0.947368  0.782609  0.857143     150    100
# 12   0.947368  0.782609  0.857143      50     60
# 33   0.947368  0.782609  0.857143     200    120
# 6    1.000000  0.782609  0.878049      50     30
# 0    0.950000  0.826087  0.883721      50     10
# 23   0.952381  0.869565  0.909091     300     90
# 3    0.952381  0.869565  0.909091     200     10
# 4    0.952381  0.869565  0.909091     250     10
# 11   0.952381  0.869565  0.909091     300     30
# 34   0.952381  0.869565  0.909091     250    120
# 5    0.952381  0.869565  0.909091     300     10
# 1    1.000000  0.869565  0.930233     100     10
# 35   1.000000  0.869565  0.930233     300    120
# 2    0.954545  0.913043  0.933333     150     10
# 28   0.954545  0.913043  0.933333     250    100
# 15   0.954545  0.913043  0.933333     200     60
# 9    1.000000  0.913043  0.954545     200     30
# 8    1.000000  0.913043  0.954545     150     30 <- best choice
# 10   1.000000  0.913043  0.954545     250     30
# 17   1.000000  0.913043  0.954545     300     60
# 29   1.000000  0.913043  0.954545     300    100
# 27   1.000000  0.913043  0.954545     200    100
# 16   1.000000  0.913043  0.954545     250     60
# 22   1.000000  0.913043  0.954545     250     90
def ex9_grid_searching_batches_epochs():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, f1_score, \
        accuracy_score, classification_report

    PATH = "../Datasets/"
    FILE = "Social_Network_Ads.csv"
    data = pd.read_csv(PATH + FILE)
    y = data["Purchased"]
    X = data.copy()
    del X['User ID']
    del X['Purchased']
    X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(data.head())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Stochastic gradient descent models are sensitive to differences
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_trainScaled = scaler.transform(X_train)
    X_testScaled = scaler.transform(X_test)
    X_valScaled = scaler.transform(X_val)

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=['f1'])
        print(dfStats)

    def evaluate_model(predictions, y_test):
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        print("Precision: " + str(precision) + " " + \
              "Recall: " + str(recall) + " " + \
              "F1: " + str(f1))
        return precision, recall, f1

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_trainScaled, y_train)
    predictions = clf.predict(X_testScaled)
    evaluate_model(predictions, y_test)

    COLUMN_DIMENSION = 1
    #######################################################################
    # Part 2
    from keras.models import Sequential
    from keras.layers import Dense
    # shape() obtains rows (dim=0) and columns (dim=1)
    n_features = X_trainScaled.shape[COLUMN_DIMENSION]

    def getPredictions(model, X_test):
        probabilities = model.predict(X_test)

        predictions = []
        for i in range(len(probabilities)):
            if (probabilities[i][0] > 0.5):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    ### Model parameters ############################
    batch_sizes = [10, 30, 60, 90, 100, 120]
    epochList = [50, 100, 150, 200, 250, 300]

    #################################################

    #######################################################################
    # Model building section.
    def create_model():
        model = Sequential()
        model.add(Dense(12, input_dim=n_features, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        return model

    networkStats = []
    for batch_size in batch_sizes:
        for epochs in epochList:
            model = create_model()
            history = model.fit(X_trainScaled, y_train, epochs=epochs,
                                batch_size=batch_size, verbose=1,
                                validation_data=(X_valScaled, y_val))
            predictions = getPredictions(model, X_testScaled)

            precision, recall, f1 = evaluate_model(predictions, y_test)
            networkStats.append({"precision": precision, "recall": recall,
                                 "f1": f1, "epochs": epochs,
                                 "batch": batch_size})
    showResults(networkStats)
    #######################################################################
# ex9_grid_searching_batches_epochs()

# example 10 grid searching optimzers for classifiesr
#    precision    recall        f1 optimizer
# 2   0.750000  0.130435  0.222222   Adagrad
# 3   0.750000  0.260870  0.387097  Adadelta
# 5   0.950000  0.826087  0.883721    Adamax
# 1   0.954545  0.913043  0.933333   RMSprop
# 0   0.954545  0.913043  0.933333       SGD
# 4   1.000000  0.913043  0.954545      Adam <- One of the best
# 6   1.000000  0.913043  0.954545     Nadam <- One of the best
def ex10_grid_searching_clfs_optimzers():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, f1_score, \
        accuracy_score, classification_report

    PATH = "../Datasets/"
    FILE = "Social_Network_Ads.csv"
    data = pd.read_csv(PATH + FILE)
    y = data["Purchased"]
    X = data.copy()
    del X['User ID']
    del X['Purchased']
    X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(data.head())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Stochastic gradient descent models are sensitive to differences
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_trainScaled = scaler.transform(X_train)
    X_testScaled = scaler.transform(X_test)
    X_valScaled = scaler.transform(X_val)

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=['f1'])
        print(dfStats)

    def evaluate_model(predictions, y_test):
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        print("Precision: " + str(precision) + " " + \
              "Recall: " + str(recall) + " " + \
              "F1: " + str(f1))
        return precision, recall, f1

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_trainScaled, y_train)
    predictions = clf.predict(X_testScaled)
    evaluate_model(predictions, y_test)

    COLUMN_DIMENSION = 1
    #######################################################################
    # Part 2
    from keras.models import Sequential
    from keras.layers import Dense
    # shape() obtains rows (dim=0) and columns (dim=1)
    n_features = X_trainScaled.shape[COLUMN_DIMENSION]

    def getPredictions(model, X_test):
        probabilities = model.predict(X_test)

        predictions = []
        for i in range(len(probabilities)):
            if (probabilities[i][0] > 0.5):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    ### Model parameters ############################
    optimizers = ['SGD', 'RMSprop', 'Adagrad',
                  'Adadelta', 'Adam', 'Adamax', 'Nadam']

    #################################################

    #######################################################################
    # Model building section.
    def create_model(optimizer):
        model = Sequential()
        model.add(Dense(12, input_dim=n_features, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    networkStats = []
    EPOCHS = 150
    NUM_BATCHES = 30

    for optimizer in optimizers:
        model = create_model(optimizer)
        history = model.fit(X_trainScaled, y_train, epochs=EPOCHS,
                            batch_size=NUM_BATCHES, verbose=1,
                            validation_data=(X_valScaled, y_val))
        predictions = getPredictions(model, X_testScaled)

        precision, recall, f1 = evaluate_model(predictions, y_test)
        networkStats.append({"precision": precision, "recall": recall,
                             "f1": f1, "optimizer": optimizer})
    showResults(networkStats)
    #######################################################################
# ex10_grid_searching_clfs_optimzers()

# example 11 grid searching learning rate for parameter
#    precision    recall        f1  learningRate
# 0   0.680000  0.739130  0.708333        0.0001
# 2   0.952381  0.869565  0.909091        0.0050
# 3   0.913043  0.913043  0.913043        0.0100
# 1   1.000000  0.913043  0.954545        0.0010 <- best one
def ex11_grid_searching_optimizer_learning_rate():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, f1_score, \
        accuracy_score, classification_report

    PATH = "../Datasets/"
    FILE = "Social_Network_Ads.csv"
    data = pd.read_csv(PATH + FILE)
    y = data["Purchased"]
    X = data.copy()
    del X['User ID']
    del X['Purchased']
    X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(data.head())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Stochastic gradient descent models are sensitive to differences
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_trainScaled = scaler.transform(X_train)
    X_testScaled = scaler.transform(X_test)
    X_valScaled = scaler.transform(X_val)

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=['f1'])
        print(dfStats)

    def evaluate_model(predictions, y_test):
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        print("Precision: " + str(precision) + " " + \
              "Recall: " + str(recall) + " " + \
              "F1: " + str(f1))
        return precision, recall, f1

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_trainScaled, y_train)
    predictions = clf.predict(X_testScaled)
    evaluate_model(predictions, y_test)

    COLUMN_DIMENSION = 1
    #######################################################################
    # Part 2
    from keras.models import Sequential
    from keras.layers import Dense
    # shape() obtains rows (dim=0) and columns (dim=1)
    n_features = X_trainScaled.shape[COLUMN_DIMENSION]

    def getPredictions(model, X_test):
        probabilities = model.predict(X_test)

        predictions = []
        for i in range(len(probabilities)):
            if (probabilities[i][0] > 0.5):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    ### Model parameters ############################
    learningRates = [0.0001, 0.001, 0.005, 0.01]

    #################################################

    #######################################################################
    # Model building section.
    import tensorflow as tf
    def create_model(learningRate):
        model = Sequential()
        model.add(Dense(12, input_dim=n_features, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    networkStats = []
    EPOCHS = 150
    NUM_BATCHES = 30

    for learningRate in learningRates:
        model = create_model(learningRate)
        history = model.fit(X_trainScaled, y_train, epochs=EPOCHS,
                            batch_size=NUM_BATCHES, verbose=1,
                            validation_data=(X_valScaled, y_val))
        predictions = getPredictions(model, X_testScaled)

        precision, recall, f1 = evaluate_model(predictions, y_test)
        networkStats.append({"precision": precision, "recall": recall,
                             "f1": f1, "learningRate": learningRate})
    showResults(networkStats)
    #######################################################################
# ex11_grid_searching_optimizer_learning_rate()

# exercise 7 setting optimal learning rate, add grid searching for optimal neurons in hidden layer
def modified_ex11_grid_searching_optimizer_learning_rate():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, f1_score, \
        accuracy_score, classification_report

    PATH = "../Datasets/"
    FILE = "Social_Network_Ads.csv"
    data = pd.read_csv(PATH + FILE)
    y = data["Purchased"]
    X = data.copy()
    del X['User ID']
    del X['Purchased']
    X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(data.head())

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Stochastic gradient descent models are sensitive to differences
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_trainScaled = scaler.transform(X_train)
    X_testScaled = scaler.transform(X_test)
    X_valScaled = scaler.transform(X_val)

    def showResults(networkStats):
        dfStats = pd.DataFrame.from_records(networkStats)
        dfStats = dfStats.sort_values(by=['f1'])
        print(dfStats)

    def evaluate_model(predictions, y_test):
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        print("Precision: " + str(precision) + " " + \
              "Recall: " + str(recall) + " " + \
              "F1: " + str(f1))
        return precision, recall, f1

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_trainScaled, y_train)
    predictions = clf.predict(X_testScaled)
    print("Logistic regression")
    evaluate_model(predictions, y_test)

    COLUMN_DIMENSION = 1
    #######################################################################
    # Part 2
    from keras.models import Sequential
    from keras.layers import Dense
    # shape() obtains rows (dim=0) and columns (dim=1)
    n_features = X_trainScaled.shape[COLUMN_DIMENSION]

    def getPredictions(model, X_test):
        probabilities = model.predict(X_test)

        predictions = []
        for i in range(len(probabilities)):
            if (probabilities[i][0] > 0.5):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    ### Model parameters ############################
    neuronList = [5, 25, 50, 100, 150]
    #################################################

    #######################################################################
    # Model building section.
    import tensorflow as tf
    def create_model(neuron): #changed to neuron
        model = Sequential()
        model.add(Dense(neuron, input_dim=n_features, activation='relu')) # changed 12 -> 5
        model.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0010) # change learning rate -> 0.0010, and Adam as optimizer
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    networkStats = []
    EPOCHS = 150 # change this 200 -> 150
    NUM_BATCHES = 30 # change this 60 -> 30

    # for neuron in neuronList: #iterate through neurons list
    #     model = create_model(neuron) #change this to neurons
    #     history = model.fit(X_trainScaled, y_train, epochs=EPOCHS,
    #                         batch_size=NUM_BATCHES, verbose=1,
    #                         validation_data=(X_valScaled, y_val))
    #     predictions = getPredictions(model, X_testScaled)
    #
    #     precision, recall, f1 = evaluate_model(predictions, y_test)
    #     networkStats.append({"precision": precision, "recall": recall,
    #                          "f1": f1, "Neuron": neuron}) # appended neuron instead of learning rate
    # showResults(networkStats)

    model = create_model(neuronList[0])
    history = model.fit(X_trainScaled, y_train, epochs=EPOCHS,
                        batch_size=NUM_BATCHES, verbose=1,
                        validation_data=(X_valScaled, y_val))
    predictions = getPredictions(model, X_testScaled)
    precision, recall, f1 = evaluate_model(predictions, y_test)
    networkStats.append({"precision": precision, "recall": recall,
                         "f1": f1})  # appended neuron instead of learning rate
    showResults(networkStats)

    print("history")
    print(history)
    #######################################################################
# modified_ex11_grid_searching_optimizer_learning_rate()





