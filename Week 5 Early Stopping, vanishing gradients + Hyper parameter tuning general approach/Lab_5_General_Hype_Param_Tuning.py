# example 1 basic log regression
def ex1_basic_logi_reg():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    PATH = "../Datasets/"
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    import numpy as np

    # load the dataset
    df = pd.read_csv(PATH + 'diabetes.csv', sep=',')
    # split into input (X) and output (y) variables

    X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age']]
    y = df[['Outcome']]
    # Split into train and test data sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True, random_state=0,
                                       solver='liblinear')
    logisticModel.fit(X_train, y_train)
    y_pred = logisticModel.predict(X_test)

    # Show model coefficients and intercept.
    print("\nModel Coefficients: ")
    print("\nIntercept: ")
    print(logisticModel.intercept_)

    print(logisticModel.coef_)

    # Show confusion matrix and accuracy scores.
    confusion_matrix = pd.crosstab(np.array(y_test['Outcome']), y_pred,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])

    print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(confusion_matrix)
# ex1_basic_logi_reg()

# After optimizing and grid searching for best values of learning rate, neurons, and momentum.
def ex2_configure_total_nodes():
    # first neural network with keras tutorial
    from keras.models import Sequential
    from keras.layers import Dense
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    PATH = "../Datasets/"
    import tensorflow as tf

    # load the dataset
    df = pd.read_csv(PATH + 'diabetes.csv', sep=',')

    # split into input (X) and output (y) variables
    X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age']]
    y = df[['Outcome']]

    # Split into train and test data sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    def buildModel(num_nodes):
        # define the keras model
        model = Sequential()
        model.add(Dense(num_nodes, input_dim=8, activation='relu',
                        kernel_initializer='he_normal'))
        model.add(Dense(1, activation='sigmoid'))

        opitimizer = tf.keras.optimizers.SGD(
            learning_rate=0.0005, momentum=0.9, name="SGD",
        )

        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer=opitimizer,
                      metrics=['accuracy'])

        # fit the keras model on the dataset
        history = model.fit(X, y, epochs=80, batch_size=10, validation_data=(X_test,
                                                                             y_test))
        # evaluate the keras model

        # Evaluate the model.
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Test Accuracy: ' + str(acc) + ' Num nodes: ' + str(num_nodes))
        return history

    def showLoss(history, numNodes):
        # Get training and test loss histories
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history for training data.
        actualLabel = str(numNodes) + " nodes"
        plt.subplot(1, 2, 1)
        # View loss on unseen data.
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()

    def showAccuracy(history, numNodes):
        # Get training and test loss histories
        training_loss = history.history['accuracy']
        validation_loss = history.history['val_accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 2)

        actualLabel = str(numNodes) + " nodes"
        # View loss on unseen data.
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()

    nodeCounts = [170, 200, 230]
    plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    # loss plot is left (lower loss better)
    # accuracy plot is right (higher acc better)
    for i in range(0, len(nodeCounts)):
        history = buildModel(nodeCounts[i])
        showLoss(history, nodeCounts[i])
        showAccuracy(history, nodeCounts[i])
        plt.show()

    # Grid searching option
    # nodeCounts = [170, 200, 230]
    # lrs = [0.001, 0.05, 0.1]
    # momentums = [0.1, 0.33, 0.5]
    # plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    #
    # dfMetrics = pd.DataFrame()
    # for i in range(0, len(nodeCounts)):
    #     for j in range(0, len(lrs)):
    #         for k in range(0, len(momentums)):
    #             history, metrics = buildModel(nodeCounts[i], lrs[j], momentums[k])
    #             dfMetrics = dfMetrics._append(metrics, ignore_index=True)
    #             showLoss(history, nodeCounts[i])
    #             showAccuracy(history, nodeCounts[i])
    #             plt.show()
    # dfMetrics = dfMetrics.sort_values(by=['acc'])
    # print(dfMetrics)
# ex2_configuring_total_nodes()

# After determining nodes
def ex3_configure_total_layer():
    # first neural network with keras tutorial
    from keras.models import Sequential
    from keras.layers import Dense
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    PATH = "../Datasets/"
    import tensorflow as tf

    # load the dataset
    df = pd.read_csv(PATH + 'diabetes.csv', sep=',')

    # split into input (X) and output (y) variables
    X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age']]
    y = df[['Outcome']]

    # Split into train and test data sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    def buildModel(numLayers):
        # define the keras model
        model = Sequential()
        model.add(Dense(230, input_dim=8, activation='relu',
                        kernel_initializer='he_normal'))

        for i in range(0, numLayers - 1):
            model.add(Dense(230, activation='relu',
                            kernel_initializer='he_normal'))
        model.add(Dense(1, activation='sigmoid'))
        opitimizer = tf.keras.optimizers.SGD(
            learning_rate=0.0005, momentum=0.9, name="SGD",
        )

        # Compile the keras model.
        model.compile(loss='binary_crossentropy', optimizer=opitimizer,
                      metrics=['accuracy'])

        # Fit the keras model on the dataset.
        history = model.fit(X, y, epochs=200, batch_size=10,
                            validation_data=(X_test, y_test))

        # Evaluate the model.
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Test Accuracy: %.3f' % acc)
        return history

    def showLoss(history, numNodes):
        # Get training and test loss histories
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history for training data.
        actualLabel = str(numNodes) + " layers"
        plt.subplot(1, 2, 1)

        # View loss on unseen data.
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()

    def showAccuracy(history, numNodes):
        # Get training and test loss histories
        training_loss = history.history['accuracy']
        validation_loss = history.history['val_accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 2)

        actualLabel = str(numNodes) + " layers"
        # View loss on unseen data.
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()

    numLayers = [5, 6, 7, 8]
    plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    for i in range(0, len(numLayers)):
        history = buildModel(numLayers[i])
        showLoss(history, numLayers[i])
        showAccuracy(history, numLayers[i])
        plt.show()
# ex3_configure_total_layer()

# After determining nodes and layers
def ex4_configure_batch_size():
    # first neural network with keras tutorial
    from keras.models import Sequential
    from keras.layers import Dense
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    PATH = "../Datasets/"
    import tensorflow as tf

    # load the dataset
    df = pd.read_csv(PATH + 'diabetes.csv', sep=',')
    # split into input (X) and output (y) variables

    X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age']]
    y = df[['Outcome']]

    # Split into train and test data sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    accuracy = []

    def buildModel(batchSize):
        NUM_LAYERS = 7
        # define the keras model
        model = Sequential()
        model.add(Dense(230, input_dim=8, activation='relu', kernel_initializer='he_normal'))
        for i in range(0, NUM_LAYERS - 1):
            model.add(Dense(230, activation='relu', kernel_initializer='he_normal'))

        model.add(Dense(1, activation='sigmoid'))

        opitimizer = tf.keras.optimizers.SGD(
            learning_rate=0.0005, momentum=0.9, name="SGD",
        )

        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer=opitimizer, metrics=['accuracy'])

        # fit the keras model on the dataset
        history = model.fit(X, y, epochs=500, batch_size=batchSize,
                            validation_data=(X_test, y_test))
        # evaluate the keras model

        # Evaluate the model.
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Test Accuracy: %.3f' % acc)
        accuracy.append(acc)
        print(accuracy)
        return history

    def showLoss(history, batchSize):
        # Get training and test loss histories
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history for training data.
        actualLabel = str(batchSize) + " batch"
        plt.subplot(1, 2, 1)
        # View loss on unseen data.
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()

    def showAccuracy(history, batchSize):
        # Get training and test loss histories
        training_loss = history.history['accuracy']
        validation_loss = history.history['val_accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 2)

        actualLabel = str(batchSize) + " batch"
        # View loss on unseen data.
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()

    plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    batchSizes = [32, len(y_train)]
    for i in range(0, len(batchSizes)):
        history = buildModel(batchSizes[i])
        showLoss(history, batchSizes[i])
        showAccuracy(history, batchSizes[i])

    plt.show()
# ex4_configure_batch_size()

# The rest are exercises here 1 - 11
def exercise1_fluDiagnosis_base_code():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    PATH = "../Datasets/"
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    import numpy as np

    # load the dataset
    df = pd.read_csv(PATH + 'fluDiagnosis.csv')
    # split into input (X) and output (y) variables
    print(df)

    X = df[['A', 'B']]
    y = df[['Diagnosed']]
    # Split into train and test data sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True, random_state=0,
                                       solver='liblinear')
    logisticModel.fit(X_train, y_train)
    y_pred = logisticModel.predict(X_test)

    # Show model coefficients and intercept.
    print("\nModel Coefficients: ")
    print("\nIntercept: ")
    print(logisticModel.intercept_)

    print(logisticModel.coef_)

    # Show confusion matrix and accuracy scores.
    confusion_matrix = pd.crosstab(np.array(y_test['Diagnosed']), y_pred,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])

    print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(confusion_matrix)
# exercise1_fluDiagnosis_base_code()

def exercise2_configure_lr_momentum_ttlnodes():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    PATH = "../Datasets/"
    from keras.models import Sequential
    from keras.layers import Dense
    import matplotlib.pyplot as plt
    import tensorflow as tf
    # load the dataset
    df = pd.read_csv(PATH + 'fluDiagnosis.csv')

    # split into input (X) and output (y) variables
    X = df[['A', 'B']]
    # print(X) keeps the headers and sends it in as a dataframe
    # print(X.values) removes column headers and turns it into an numpy array
    # print("X shape", X.shape) #.shape tells the amount of [rows, cols]
    y = df[['Diagnosed']]
    # Split into train and test data sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    def buildModel(num_nodes, learning_rate, momentum):
        # define the keras model
        model = Sequential()
        model.add(Dense(num_nodes, input_dim=2, activation='relu',
                        kernel_initializer='he_normal'))
        model.add(Dense(1, activation='sigmoid'))

        opitimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, name="SGD",)

        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer=opitimizer, metrics=['accuracy'])

        # fit the keras model on the dataset
        history = model.fit(X, y, epochs=80, batch_size=10, validation_data=(X_test, y_test))
        # evaluate the keras model

        # Evaluate the model.
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Test Accuracy: ' + str(acc) + ' Num nodes: ' + str(num_nodes)
              + ' Learning rate: ' + str(learning_rate) + ' Momentum: ' + str(momentum))
        return history, {
            "accuracy": acc,
            "num_nodes": num_nodes,
            "learning_rate": learning_rate,
            "momentum": momentum
        }
    def showLoss(history, numNodes):
        # Get training and test loss histories
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history for training data.
        actualLabel = str(numNodes) + " nodes"
        plt.subplot(1, 2, 1)
        # View loss on unseen data.
        plt.title("Loss Graph")
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()
    def showAccuracy(history, numNodes):
        # Get training and test loss histories
        training_loss = history.history['accuracy']
        validation_loss = history.history['val_accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 2)

        actualLabel = str(numNodes) + " nodes"
        # View loss on unseen data.
        plt.title("Accuracy Graph")
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()

    nodeCounts = [75, 90, 105]
    lr = [0.0001]
    momentum = [0.05]
    plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    results = []
    for i in range(0, len(nodeCounts)):
        for j in range(0, len(lr)):
            for k in range(0, len(momentum)):
                history, summary = buildModel(nodeCounts[i], lr[j], momentum[k])
                results.append(summary)
                showLoss(history, nodeCounts[i])
                showAccuracy(history, nodeCounts[i])
    plt.show()

    # Convert to Pandas DataFrame for better visualization
    df_results = pd.DataFrame(results)

    print("\nSummary of Results:")
    print(df_results)
# exercise2_configure_lr_momentum_ttlnodes()

def exercise3_configure_layers():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    PATH = "../Datasets/"
    from keras.models import Sequential
    from keras.layers import Dense
    import matplotlib.pyplot as plt
    import tensorflow as tf
    # load the dataset
    df = pd.read_csv(PATH + 'fluDiagnosis.csv')
    # split into input (X) and output (y) variables
    X = df[['A', 'B']]
    y = df[['Diagnosed']]

    # Split into train and test data sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    def buildModel(numLayers):
        # define the keras model
        model = Sequential()

        # Change to test out
        neurons = 90 #old 170 -> new best 230
        lr = 0.0001 #old 0.05 -> new best 0.001
        momentum = 0.05 #old 0.1 -> 0.9

        # Layer testing
        model.add(Dense(neurons, input_dim=2, activation='relu',kernel_initializer='he_normal')) #initial layer
        for i in range(0, numLayers - 1):
            model.add(Dense(neurons, activation='relu',kernel_initializer='he_normal')) #added layers
        model.add(Dense(1, activation='sigmoid'))

        opitimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, name="SGD",)

        # Compile the keras model.
        model.compile(loss='binary_crossentropy', optimizer=opitimizer,
                      metrics=['accuracy'])

        # Fit the keras model on the dataset.
        history = model.fit(X, y, epochs=80, batch_size=10,
                            validation_data=(X_test, y_test))

        # Evaluate the model.
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Test Accuracy: %.3f' % acc)
        return history, {
            "accuracy": acc,
            "loss": loss,
            "numLayers": numLayers,
        }
    def showLoss(history, numNodes):
        # Get training and test loss histories
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history for training data.
        actualLabel = str(numNodes) + " layers"
        plt.subplot(1, 2, 1)
        # View loss on unseen data.
        plt.title("Loss Graph")
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()
    def showAccuracy(history, numNodes):
        # Get training and test loss histories
        training_loss = history.history['accuracy']
        validation_loss = history.history['val_accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 2)

        actualLabel = str(numNodes) + " layers"
        # View loss on unseen data.
        plt.title("Accuracy Graph")
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()

    layers = [1, 3, 7]
    plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    results = []
    for i in range(0, len(layers)):
        history, summary = buildModel(layers[i])
        showLoss(history, layers[i])
        showAccuracy(history, layers[i])
        results.append(summary)
    plt.show()

    df_results = pd.DataFrame(results)
    print("\nSummary of Results:\n")
    print(df_results)
# exercise3_configure_layers()

def exercise4_configure_batches():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    PATH = "../Datasets/"
    from keras.models import Sequential
    from keras.layers import Dense
    import matplotlib.pyplot as plt
    import tensorflow as tf
    # load the dataset
    df = pd.read_csv(PATH + 'fluDiagnosis.csv')
    # split into input (X) and output (y) variables
    X = df[['A', 'B']]
    y = df[['Diagnosed']]

    # Split into train and test data sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    def buildModel(batch):
        # define the keras model
        model = Sequential()

        # Change to test out
        layers = 3
        neurons = 90 #old 170 -> new best 230
        lr = 0.0001 #old 0.05 -> new best 0.001
        momentum = 0.05 #old 0.1 -> 0.9

        # Layer testing
        model.add(Dense(neurons, input_dim=2, activation='relu',kernel_initializer='he_normal')) #initial layer
        for i in range(0, layers - 1):
            model.add(Dense(neurons, activation='relu',kernel_initializer='he_normal')) #added layers
        model.add(Dense(1, activation='sigmoid'))

        opitimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, name="SGD",)

        # Compile the keras model.
        model.compile(loss='binary_crossentropy', optimizer=opitimizer,
                      metrics=['accuracy'])

        # Fit the keras model on the dataset.
        history = model.fit(X, y, epochs=80, batch_size=batch,
                            validation_data=(X_test, y_test))

        # Evaluate the model.
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Test Accuracy: %.3f' % acc)
        return history, {
            "accuracy": acc,
            "loss": loss,
            "batches": batch,
        }
    def showLoss(history, numNodes):
        # Get training and test loss histories
        training_loss = history.history['loss']
        validation_loss = history.history['val_loss']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history for training data.
        actualLabel = str(numNodes) + " layers"
        plt.subplot(1, 2, 1)
        # View loss on unseen data.
        plt.title("Loss Graph")
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()
    def showAccuracy(history, numNodes):
        # Get training and test loss histories
        training_loss = history.history['accuracy']
        validation_loss = history.history['val_accuracy']

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)
        plt.subplot(1, 2, 2)

        actualLabel = str(numNodes) + " layers"
        # View loss on unseen data.
        plt.title("Accuracy Graph")
        plt.plot(epoch_count, validation_loss, label=actualLabel)
        plt.legend()

    batch = [15]
    plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    results = []
    for i in range(0, len(batch)):
        history, summary = buildModel(batch[i])
        showLoss(history, batch[i])
        showAccuracy(history, batch[i])
        results.append(summary)
    plt.show()

    df_results = pd.DataFrame(results)
    print("\nSummary of Results:\n")
    print(df_results)
# exercise4_configure_batches()