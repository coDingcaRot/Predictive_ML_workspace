# example 1
# exercise 1, 2
def ex1_bin_clfs():
    from sklearn.datasets import make_classification
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(4, num_neurons)
            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # # EXERCISE 1 Third layer
            # self.dense2 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    def buildModel(x, y):
        # Trains the Neural Network with fixed hyperparameters
        # The Neural Net is initialized with fixed hyperparameters
        myNetwork = MyNeuralNet(num_neurons=10, dropout=0.1)
        nn = NeuralNetClassifier(myNetwork, max_epochs=10,
                                 lr=0.01, batch_size=12,
                                 optimizer=optim.RMSprop)
        model = nn.fit(x, y)
        return model

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    # # Example 1
    # # Prep the data.
    # X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    # X = X.astype(np.float32)
    # y = y.astype(np.int64)
    # X_train, X_test, y_train, y_test = \
    #     train_test_split(X, y, test_size=0.2)

    # Exercise 2
    import torch
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('../Datasets/bill_authentication.csv')
    X = df.copy()
    del X['Class']
    y = df['Class']

    print(df.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    scaledXTrain = scalerX.fit_transform(X_train)
    scaledXTest = scalerX.transform(X_test)

    # Must convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(scaledXTrain, dtype=torch.float32)
    X_test_tensor = torch.tensor(scaledXTest, dtype=torch.float32)
    y_train_tensor = torch.tensor(list(y_train), dtype=torch.long)
    y_test_tensor = torch.tensor(list(y_test), dtype=torch.long)

    # Build the model.
    model = buildModel(X_train_tensor, y_train_tensor)

    # Evaluate the model.
    evaluateModel(model, X_test_tensor, y_test_tensor)
# ex1_bin_clfs()

# example 2 loss & acc plot
# exercise 3, 4
def ex2_pytorch_visual_acc_loss_info():
    from sklearn.datasets import make_classification
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(4, num_neurons)
            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # # EXERCISE 1 Third layer
            # self.dense2 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    # Replaced for example 2
    from skorch.callbacks import EpochScoring
    def buildModel(X_train, y_train):
        num_neurons = 25  # hidden layers
        net = NeuralNetClassifier(MyNeuralNet(num_neurons), max_epochs=200,
                                  lr=0.001, batch_size=100, optimizer=optim.RMSprop,
                                  callbacks=[EpochScoring(scoring='accuracy',
                                                          name='Jonathaniel_acc', on_train=True)])
        # Pipeline execution
        model = net.fit(X_train, y_train)
        return model, net

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    import torch
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv('../Datasets/bill_authentication.csv')
    X = df.copy()
    del X['Class']
    y = df['Class']

    print(df.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from sklearn.preprocessing import StandardScaler
    scalerX = StandardScaler()
    scaledXTrain = scalerX.fit_transform(X_train)
    scaledXTest = scalerX.transform(X_test)

    # Must convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(scaledXTrain, dtype=torch.float32)
    X_test_tensor = torch.tensor(scaledXTest, dtype=torch.float32)
    y_train_tensor = torch.tensor(list(y_train), dtype=torch.long)
    y_test_tensor = torch.tensor(list(y_test), dtype=torch.long)

    # Build the model.
    model, net = buildModel(X_train_tensor, y_train_tensor)

    print("Breakpoint ")
    # Evaluate the model.
    evaluateModel(model, X_test_tensor, y_test_tensor)

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 30})

    def drawLossPlot(net):
        plt.plot(net.history[:, 'train_loss'], color='blue', label='train')
        plt.plot(net.history[:, 'valid_loss'], color='orange', label='val')
        plt.legend()
        plt.show()

    def drawAccuracyPlot(net):
        plt.plot(net.history[:, 'Jonathaniel_acc'], color='blue', label='train')
        plt.plot(net.history[:, 'valid_acc'], color='orange', label='val')
        plt.legend()
        plt.show()

    drawLossPlot(net)
    drawAccuracyPlot(net)
# ex2_pytorch_visual_acc_loss_info()


# if given error proceed with example 4 which is an error fix to mat1 mat2 dtype error
def ex3_grid_search():
    from sklearn.datasets import make_classification
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(20, num_neurons)
            print("Dense layer type:")
            print(self.dense0.weight.dtype)

            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            print("X type: ")
            print(x.dtype)

            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    def buildModel(x, y):
        # Through a grid search, the optimal hyperparameters are found
        # A pipeline is used in order to scale and train the neural net
        # The grid search module from scikit-learn wraps the pipeline

        # The Neural Net is instantiated, none hyperparameter is provided
        nn = NeuralNetClassifier(MyNeuralNet, verbose=0, train_split=False)
        # The pipeline is instantiated, it wraps scaling and training phase
        pipeline = Pipeline([('scale', StandardScaler()), ('nn', nn)])

        # The parameters for the grid search are defined
        # Must use prefix "nn__" when setting hyperparamters for the training phase
        # Must use prefix "nn__module__" when setting hyperparameters for the Neural Net
        params = {
            'nn__max_epochs': [10, 20],
            'nn__lr': [0.1, 0.01],
            'nn__module__num_neurons': [5, 10],
            'nn__module__dropout': [0.1, 0.5],
            'nn__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}

        # The grid search module is instantiated
        gs = GridSearchCV(pipeline, params, refit=True, cv=3,
                          scoring='balanced_accuracy', verbose=1)

        return gs.fit(x, y)

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    # Prep the data.
    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2)

    # Build the model.
    model = buildModel(X_train, y_train)

    print("Best parameters:")
    print(model.best_params_)

    # Evaluate the model.
    evaluateModel(model.best_estimator_, X_test, y_test)
# ex3_grid_search()

# fixes it
def ex4_mat1_mat2_handling():
    from sklearn.datasets import make_classification
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(3, num_neurons)
            print("Dense layer type:")
            print(self.dense0.weight.dtype)

            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            print("X type: ")
            print(x.dtype)
            x = x.to(torch.float32)
            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    def buildModel(x, y):
        # Through a grid search, the optimal hyperparameters are found
        # A pipeline is used in order to scale and train the neural net
        # The grid search module from scikit-learn wraps the pipeline

        # The Neural Net is instantiated, none hyperparameter is provided
        nn = NeuralNetClassifier(MyNeuralNet, verbose=0, train_split=False)
        # The pipeline is instantiated, it wraps scaling and training phase
        pipeline = Pipeline([('nn', nn)])

        # The parameters for the grid search are defined
        # Must use prefix "nn__" when setting hyperparamters for the training phase
        # Must use prefix "nn__module__" when setting hyperparameters for the Neural Net
        params = {
            'nn__max_epochs': [10, 20],
            'nn__lr': [0.1, 0.01],
            'nn__module__num_neurons': [5, 10],
            'nn__module__dropout': [0.1, 0.5],
            'nn__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}

        # The grid search module is instantiated
        gs = GridSearchCV(pipeline, params, refit=True, cv=3,
                          scoring='balanced_accuracy', verbose=1)

        return gs.fit(x, y)

    def evaluateModel(model, X_test, y_test):
        print(model.best_estimator_)
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    ######################### Data Prep Start
    # Setup data.
    import pandas as pd
    import numpy as np
    import torch

    candidates = {'gmat': [780, 750, 690, 710, 680, 730, 690, 720,
                           740, 690, 610, 690, 710, 680, 770, 610, 580, 650, 540, 590, 620,
                           600, 550, 550, 570, 670, 660, 580, 650, 660, 640, 620, 660, 660,
                           680, 650, 670, 580, 590, 690],
                  'gpa': [4, 3.9, 3.3, 3.7, 3.9, 3.7, 2.3, 3.3,
                          3.3, 1.7, 2.7, 3.7, 3.7, 3.3, 3.3, 3, 2.7, 3.7, 2.7, 2.3,
                          3.3, 2, 2.3, 2.7, 3, 3.3, 3.7, 2.3, 3.7, 3.3, 3, 2.7, 4,
                          3.3, 3.3, 2.3, 2.7, 3.3, 1.7, 3.7],
                  'work_experience': [3, 4, 3, 5, 4, 6, 1, 4, 5,
                                      1, 3, 5, 6, 4, 3, 1, 4, 6, 2, 3, 2, 1, 4, 1, 2, 6, 4, 2, 6, 5, 1, 2, 4, 6,
                                      5, 1, 2, 1, 4, 5],
                  'admitted': [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1,
                               1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0,
                               0, 0, 1]}

    df = pd.DataFrame(candidates, columns=['gmat', 'gpa',
                                           'work_experience', 'admitted'])
    y = np.array(df['admitted'])
    X = df.copy()
    del X['admitted']
    X = X

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # define standard scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # transform data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    # transform data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Tensors are not being used here because the data is being
    # passed to a sci-kit learn pipeline.
    ######################### End

    # Build the model.
    model = buildModel(X_train_scaled, y_train)

    print("Best parameters:")
    print(model.best_params_)
    print("Jonathaniel's Network model for college admissions")

    # Evaluate the model.
    evaluateModel(model, X_test_scaled, y_test)
# ex4_mat1_mat2_handling()

# fixes it (Unfinished here)
def ex5_flu_Diagnosis():
    from sklearn.datasets import make_classification
    from torch import optim
    from skorch import NeuralNetClassifier
    import torch.nn as nn
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            self.dense0 = nn.Linear(3, num_neurons)
            print("Dense layer type:")
            print(self.dense0.weight.dtype)

            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1 = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            print("X type: ")
            print(x.dtype)
            x = x.to(torch.float32)
            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    def buildModel(x, y):
        # Through a grid search, the optimal hyperparameters are found
        # A pipeline is used in order to scale and train the neural net
        # The grid search module from scikit-learn wraps the pipeline

        # The Neural Net is instantiated, none hyperparameter is provided
        nn = NeuralNetClassifier(MyNeuralNet, verbose=0, train_split=False)
        # The pipeline is instantiated, it wraps scaling and training phase
        pipeline = Pipeline([('nn', nn)])

        # The parameters for the grid search are defined
        # Must use prefix "nn__" when setting hyperparamters for the training phase
        # Must use prefix "nn__module__" when setting hyperparameters for the Neural Net
        params = {
            'nn__max_epochs': [10, 20],
            'nn__lr': [0.1, 0.01],
            'nn__module__num_neurons': [5, 10],
            'nn__module__dropout': [0.1, 0.5],
            'nn__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}

        # The grid search module is instantiated
        gs = GridSearchCV(pipeline, params, refit=True, cv=3,
                          scoring='balanced_accuracy', verbose=1)

        return gs.fit(x, y)

    def evaluateModel(model, X_test, y_test):
        print(model.best_estimator_)
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    ######################### Data Prep Start
    # Setup data.
    import pandas as pd
    import numpy as np
    import torch

    candidates = {'gmat': [780, 750, 690, 710, 680, 730, 690, 720,
                           740, 690, 610, 690, 710, 680, 770, 610, 580, 650, 540, 590, 620,
                           600, 550, 550, 570, 670, 660, 580, 650, 660, 640, 620, 660, 660,
                           680, 650, 670, 580, 590, 690],
                  'gpa': [4, 3.9, 3.3, 3.7, 3.9, 3.7, 2.3, 3.3,
                          3.3, 1.7, 2.7, 3.7, 3.7, 3.3, 3.3, 3, 2.7, 3.7, 2.7, 2.3,
                          3.3, 2, 2.3, 2.7, 3, 3.3, 3.7, 2.3, 3.7, 3.3, 3, 2.7, 4,
                          3.3, 3.3, 2.3, 2.7, 3.3, 1.7, 3.7],
                  'work_experience': [3, 4, 3, 5, 4, 6, 1, 4, 5,
                                      1, 3, 5, 6, 4, 3, 1, 4, 6, 2, 3, 2, 1, 4, 1, 2, 6, 4, 2, 6, 5, 1, 2, 4, 6,
                                      5, 1, 2, 1, 4, 5],
                  'admitted': [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1,
                               1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0,
                               0, 0, 1]}

    df = pd.DataFrame(candidates, columns=['gmat', 'gpa',
                                           'work_experience', 'admitted'])
    y = np.array(df['admitted'])
    X = df.copy()
    del X['admitted']
    X = X

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # define standard scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # transform data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    # transform data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Tensors are not being used here because the data is being
    # passed to a sci-kit learn pipeline.
    ######################### End

    # Build the model.
    model = buildModel(X_train_scaled, y_train)

    print("Best parameters:")
    print(model.best_params_)
    print("Jonathaniel's Network model for college admissions")

    # Evaluate the model.
    evaluateModel(model, X_test_scaled, y_test)
# ex5_flu_Diagnosis()

# Nathans working code.
# Grid Search
def e6_ex3_auto():
    from sklearn.datasets           import make_classification
    from   torch                      import optim
    from   skorch                     import NeuralNetClassifier
    import torch.nn as nn
    import numpy    as np
    from   sklearn.model_selection  import train_test_split
    from   sklearn.metrics          import classification_report

    # This class could be any name.
    # nn.Module is needed to enable grid searching of parameters
    # with skorch later.
    class MyNeuralNet(nn.Module):
        # Define network objects.
        # Defaults are set for number of neurons and the
        # dropout rate.
        def __init__(self, num_neurons=10, dropout=0.1):
            super(MyNeuralNet, self).__init__()
            # 1st hidden layer.
            # nn. Linear(n,m) is a module that creates single layer
            # feed forward network with n inputs and m output.
            # self.dense0         = nn.Linear(20, num_neurons)
            self.dense0 = nn.Linear(X.shape[1], num_neurons)
            print("Dense layer type:")
            print(self.dense0.weight.dtype)

            self.activationFunc = nn.ReLU()

            # Drop samples to help prevent overfitting.
            self.dropout        = nn.Dropout(dropout)

            # 2nd hidden layer.
            self.dense1         = nn.Linear(num_neurons, num_neurons)

            # Output layer.
            self.output         = nn.Linear(num_neurons, 2)

            # Softmax activation function allows for multiclass predictions.
            # In this case the prediction is binary.
            self.softmax        = nn.Softmax(dim=-1)

        # Move data through the different network objects.
        def forward(self, x):
            print("X type: ")
            print(x.dtype)

            # Pass data from 1st hidden layer to activation function
            # before sending to next layer.
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    from sklearn.pipeline           import Pipeline
    from sklearn.preprocessing      import StandardScaler
    from sklearn.model_selection    import GridSearchCV
    def buildModel(x, y):
        # Through a grid search, the optimal hyperparameters are found
        # A pipeline is used in order to scale and train the neural net
        # The grid search module from scikit-learn wraps the pipeline

        # The Neural Net is instantiated, none hyperparameter is provided
        nn = NeuralNetClassifier(MyNeuralNet, verbose=0, train_split=False)
        # The pipeline is instantiated, it wraps scaling and training phase
        pipeline = Pipeline([('scale', StandardScaler()), ('nn', nn)])

        # The parameters for the grid search are defined
        # Must use prefix "nn__" when setting hyperparamters for the training phase
        # Must use prefix "nn__module__" when setting hyperparameters for the Neural Net
        params = {
            'nn__max_epochs': [10, 20],
            'nn__lr': [0.1, 0.01],
            'nn__module__num_neurons': [5, 10],
            'nn__module__dropout': [0.1, 0.5],
            'nn__optimizer': [optim.Adam, optim.SGD, optim.RMSprop]}

        # The grid search module is instantiated
        gs = GridSearchCV(pipeline, params, refit=True, cv=3,
                          scoring='balanced_accuracy', verbose=1)

        return gs.fit(x, y)

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    # Load the fluDiagnosis.csv file
    import pandas as pd
    data = pd.read_csv('../Datasets/fluDiagnosis.csv')
    X = data.drop('Diagnosed', axis=1).values.astype(np.float32)
    y = data['Diagnosed'].values.astype(np.int64)

    # Prep the data.
    # X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    # X    = X.astype(np.float32)
    # y    = y.astype(np.int64)

    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.2)

    # Build the model.
    model  = buildModel(X_train, y_train)

    print("Jonathaniels's flu diagnosis network best parameters.")
    print("Best parameters:")
    print(model.best_params_)

    # Evaluate the model.
    evaluateModel(model.best_estimator_, X_test, y_test)
# e6_ex3_auto()

def e7():
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    from torch import optim
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    from skorch import NeuralNetClassifier
    from skorch.callbacks import EpochScoring

    df = pd.read_csv('../Datasets/fluDiagnosis.csv')
    X = df.drop(columns=['Diagnosed'])  # Features
    y = df['Diagnosed'].values         # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class MyNeuralNet(nn.Module):
        def __init__(self, num_neurons=5, dropout=0.2):
            super(MyNeuralNet, self).__init__()
            self.dense0 = nn.Linear(2, num_neurons)
            self.activationFunc = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.dense1 = nn.Linear(num_neurons, num_neurons)
            self.output = nn.Linear(num_neurons, 2)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            x = x.to(torch.float32)
            x = self.activationFunc(self.dense0(x))
            x = self.dropout(x)
            x = self.activationFunc(self.dense1(x))
            x = self.softmax(self.output(x))
            return x

    net = NeuralNetClassifier(
        MyNeuralNet(num_neurons=5, dropout=0.1),
        max_epochs=10,
        lr=0.1,
        optimizer=torch.optim.RMSprop,
        callbacks=[EpochScoring(scoring='accuracy', on_train=True, name='train_acc')]
    )

    pipeline = Pipeline([('nn', net)])
    pipeline.fit(X_train_scaled, y_train)
    y_pred = pipeline.predict(X_test_scaled)

    print(classification_report(y_test, y_pred))

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 30})

    def drawLossPlot(net):
        plt.title("Loss Plot")
        plt.plot(net.history[:, 'train_loss'], color='blue', label='train')
        plt.plot(net.history[:, 'valid_loss'], color='orange', label='val')
        plt.legend()
        plt.show()

    def drawAccuracyPlot(net):
        plt.title("Accuracy Plot")
        plt.plot(net.history[:, 'train_acc'], color='blue', label='train')
        plt.plot(net.history[:, 'valid_acc'], color='orange', label='val')
        plt.legend()
        plt.show()

    drawLossPlot(net)
    drawAccuracyPlot(net)
# e7()

# e8: ex1 vs e8, e9: breakpoint debug
def e8_e9():
    import sklearn
    import torch.nn as nn
    import torch
    from sklearn.datasets import load_iris
    from torch import optim
    from skorch import NeuralNetClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    from skorch.callbacks import EpochScoring
    from sklearn.model_selection import train_test_split

    # Get iris data.
    iris = load_iris()
    X = iris.data
    Y = iris.target

    # Split and scale the data.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)  # Features
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)    # Features
    y_train = torch.tensor(y_train, dtype=torch.long)            # Targets
    y_test = torch.tensor(y_test, dtype=torch.long)              # Targets

    class Net(nn.Module):
        def __init__(self, num_features, num_neurons, output_dim):
            super(Net, self).__init__()
            self.dense0 = nn.Linear(num_features, num_neurons)
            self.activationFunc = nn.ReLU()
            DROPOUT = 0.1
            self.dropout = nn.Dropout(DROPOUT)
            self.dense1 = nn.Linear(num_neurons, output_dim)
            self.output = nn.Linear(output_dim, 3)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            x = x.to(torch.float32)
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test.numpy(), y_pred)  # Convert y_test to numpy for evaluation
        print(report)

    def buildModel(X_train, y_train):
        input_dim = 4
        num_neurons = 25
        output_dim = 3
        net = NeuralNetClassifier(Net(
            input_dim, num_neurons, output_dim), max_epochs=300,
            lr=0.001, batch_size=100, optimizer=optim.RMSprop,
            callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True)],
                                 )
        model = net.fit(X_train, y_train)
        return model, net

    model, net = buildModel(X_train, y_train)
    evaluateModel(model, X_test, y_test)
    print("Done")

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 30})
    def drawLossPlot(net):
        # Breakpoint below e9
        print("breakpoint (e9)")
        plt.plot(net.history[:, 'train_loss'], color='blue', label='train')
        plt.plot(net.history[:, 'valid_loss'], color='orange', label='val')
        plt.legend()
        plt.show()

    def drawAccuracyPlot(net):
        plt.plot(net.history[:, 'train_acc'], color='blue', label='train')
        plt.plot(net.history[:, 'valid_acc'], color='orange', label='val')
        plt.legend()
        plt.show()

    drawLossPlot(net)
    drawAccuracyPlot(net)
# e8_e9()

# Set from 200 (e5? is 300?) to 1000 epochs
def e10_eary_stop():
    import sklearn
    import torch.nn as nn
    import torch
    from sklearn.datasets import load_iris
    from torch import optim
    from skorch import NeuralNetClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    from skorch.callbacks import EpochScoring
    from skorch.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split

    # Get iris data.
    iris = load_iris()
    X = iris.data
    Y = iris.target

    # Split and scale the data.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)  # Features
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)    # Features
    y_train = torch.tensor(y_train, dtype=torch.long)            # Targets
    y_test = torch.tensor(y_test, dtype=torch.long)              # Targets

    class Net(nn.Module):
        def __init__(self, num_features, num_neurons, output_dim):
            super(Net, self).__init__()
            self.dense0 = nn.Linear(num_features, num_neurons)
            self.activationFunc = nn.ReLU()
            DROPOUT = 0.1
            self.dropout = nn.Dropout(DROPOUT)
            self.dense1 = nn.Linear(num_neurons, output_dim)
            self.output = nn.Linear(output_dim, 3)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            x = x.to(torch.float32)
            X = self.activationFunc(self.dense0(x))
            X = self.dropout(X)
            X = self.activationFunc(self.dense1(X))
            X = self.softmax(self.output(X))
            return X

    def evaluateModel(model, X_test, y_test):
        print(model)
        y_pred = model.predict(X_test)
        report = classification_report(y_test.numpy(), y_pred)  # Convert y_test to numpy for evaluation
        print(report)

    def buildModel(X_train, y_train):
        input_dim = 4
        num_neurons = 25
        output_dim = 3
        net = NeuralNetClassifier(Net(
            input_dim, num_neurons, output_dim), max_epochs=1000,
            lr=0.001, batch_size=100, optimizer=optim.RMSprop,
            # callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True)],
            callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True),
                       EarlyStopping(patience=100)]

        )
        model = net.fit(X_train, y_train)
        return model, net

    model, net = buildModel(X_train, y_train)
    evaluateModel(model, X_test, y_test)
    print("Done")

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 30})
    def drawLossPlot(net):
        plt.title("Loss Plot")
        plt.plot(net.history[:, 'train_loss'], color='blue', label='train')
        plt.plot(net.history[:, 'valid_loss'], color='orange', label='val')
        plt.legend()
        plt.show()

    def drawAccuracyPlot(net):
        plt.title("Accuracy Plot")
        plt.plot(net.history[:, 'train_acc'], color='blue', label='train')
        plt.plot(net.history[:, 'valid_acc'], color='orange', label='val')
        plt.legend()
        plt.show()

    drawLossPlot(net)
    drawAccuracyPlot(net)
# e10_eary_stop()

# no grid search, early stopping
def e11():
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch import optim
    from skorch import NeuralNetClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    from skorch.callbacks import EpochScoring, EarlyStopping
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Show all columns.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    def getCustomerSegmentationData():
        df = pd.read_csv('../Datasets/CustomerSegmentation.csv')
        df = pd.get_dummies(df, columns=['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1'])
        df['Segmentation'] = df['Segmentation'].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3})
        print(df['Segmentation'].value_counts())
        X = df.copy()
        del X['Segmentation']
        y = df['Segmentation']
        return X, y

    X, y = getCustomerSegmentationData()

    # Split and scale the data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    # Define the Neural Network class
    class Net(nn.Module):
        def __init__(self, num_features, num_neurons, output_dim):
            super(Net, self).__init__()
            self.dense0 = nn.Linear(num_features, num_neurons)
            self.activationFunc = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.dense1 = nn.Linear(num_neurons, output_dim)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            x = x.to(torch.float32)
            x = self.activationFunc(self.dense0(x))
            x = self.dropout(x)
            x = self.dense1(x)
            return self.softmax(x)

    # Define model parameters
    input_dim = X_train.shape[1]
    num_neurons = 10
    output_dim = 4

    net = NeuralNetClassifier(
        Net,
        module__num_features=input_dim,
        module__num_neurons=num_neurons,
        module__output_dim=output_dim,
        max_epochs=1000,
        optimizer=optim.Adam,
        lr=0.01,
        criterion=nn.CrossEntropyLoss,
        callbacks=[EpochScoring(scoring='accuracy', name='train_acc', on_train=True), EarlyStopping(patience=100)],
        verbose=1
    )

    # Train the model
    net.fit(X_train, y_train)

    # Evaluate the model
    y_pred = net.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Plot functions
    def drawLossPlot(net):
        plt.title("Loss Plot")
        plt.plot(net.history[:, 'train_loss'], color='blue', label='train')
        plt.plot(net.history[:, 'valid_loss'], color='orange', label='val')
        plt.legend()
        plt.show()

    def drawAccuracyPlot(net):
        plt.title("Accuracy Plot")
        plt.plot(net.history[:, 'train_acc'], color='blue', label='train')
        plt.plot(net.history[:, 'valid_acc'], color='orange', label='val')
        plt.legend()
        plt.show()

    drawLossPlot(net)
    drawAccuracyPlot(net)
# e11()