def example1_SDG():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm

    PATH = "../Datasets/"
    CSV_DATA = "winequality.csv"

    dataset = pd.read_csv(PATH + CSV_DATA,
                          skiprows=1,  # Don't include header row as part of data.
                          encoding="ISO-8859-1", sep=',',
                          names=('fixed acidity', 'volatile acidity', 'citric acid',
                                 'residual sugar', 'chlorides', 'free sulfur dioxide',
                                 'total sulfur dioxide', 'density', 'pH', 'sulphates',
                                 'alcohol', 'quality'))
    # Show all columns.
    pd.set_option('display.max_columns', None)

    # Increase number of columns that display on one line.
    pd.set_option('display.width', 1000)
    print(dataset.head())
    print(dataset.describe())
    X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']]

    # Adding an intercept *** This is required ***. Don't forget this step.
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)
    y = dataset['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ###########################################################
    print("\nStochastic Gradient Descent")
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import mean_squared_error

    # Stochastic gradient descent models are sensitive to differences
    # in scale so a StandardScaler is usually used.
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_trainScaled = scaler.transform(X_train)
    X_testScaled = scaler.transform(X_test)

    # SkLearn SGD classifier
    sgd = SGDRegressor(verbose=1)
    sgd.fit(X_trainScaled, y_train)
    predictions = sgd.predict(X_testScaled)
    print('Root Mean Squared Error:',
          np.sqrt(mean_squared_error(y_test, predictions)))
# example1_SDG()

def example2_logisticRegression():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics

    # Setup data.
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
    print(df)

    # Separate into x and y values.
    X = df[['gmat', 'gpa', 'work_experience']]
    y = df['admitted']

    # Import the necessary libraries first
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    # Show chi-square scores for each feature.
    # There is 1-degree freedom since 1 predictor during feature evaluation.
    # Generally, >=3.8 is good)
    test = SelectKBest(score_func=chi2, k=3)
    chiScores = test.fit(X, y)  # Summarize scores
    np.set_printoptions(precision=3)
    print("\nPredictor Chi-Square Scores: " + str(chiScores.scores_))

    # Re-assign X with significant columns only after chi-square test.
    X = df[['gmat', 'work_experience']]

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Perform logistic regression.
    logisticModel = LogisticRegression(fit_intercept=True, random_state=0,
                                       solver='liblinear')

    # Stochastic gradient descent models are sensitive to differences
    # in scale so a StandardScaler is usually used.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_trainScaled = scaler.transform(X_train)
    X_testScaled = scaler.transform(X_test)

    logisticModel.fit(X_trainScaled, y_train)
    y_pred = logisticModel.predict(X_testScaled)

    # Show model coefficients and intercept.
    print("\nModel Coefficients: ")
    print("\nIntercept: ")
    print(logisticModel.intercept_)

    print(logisticModel.coef_)

    # Show confusion matrix and accuracy scores.
    confusion_matrix = pd.crosstab(y_test, y_pred,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])

    print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(confusion_matrix)
# example2_logisticRegression()

def example3_SDGClassifer():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm

    PATH = "../Datasets/"
    CSV_DATA = "winequality.csv"

    dataset = pd.read_csv(PATH + CSV_DATA,
                          skiprows=1,  # Don't include header row as part of data.
                          encoding="ISO-8859-1", sep=',',
                          names=('fixed acidity', 'volatile acidity', 'citric acid',
                                 'residual sugar', 'chlorides', 'free sulfur dioxide',
                                 'total sulfur dioxide', 'density', 'pH', 'sulphates',
                                 'alcohol', 'quality'))
    # Show all columns.
    pd.set_option('display.max_columns', None)

    # Increase number of columns that display on one line.
    pd.set_option('display.width', 1000)
    print(dataset.head())
    print(dataset.describe())
    X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']]

    # Adding an intercept *** This is required ***. Don't forget this step.
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)
    y = dataset['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    ###########################################################
    print("\nStochastic Gradient Descent")
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier()

    # Stochastic gradient descent models are sensitive to differences
    # in scale so a StandardScaler is usually used.
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics

    scaler = StandardScaler()
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_trainScaled = scaler.transform(X_train)
    X_testScaled = scaler.transform(X_test)

    clf.fit(X_trainScaled, y_train)

    y_pred = clf.predict(X_testScaled)

    # Show confusion matrix and accuracy scores.
    confusion_matrix = pd.crosstab(y_test, y_pred,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])

    print('\nAccuracy: ', metrics.accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix")
    print(confusion_matrix)
# example1_SDG()


weights = [0.5, 2.3, 2.9]
heights = [1.4, 1.9, 3.2]

def getSlopeOfLossFunction(weights, heights, intercept):
    sum = 0
    BETA = 0.64
    for i in range(0, len(weights)):
        sum += -2 * (heights[i] - intercept - BETA * weights[i])

    print("Intercept: " + str(intercept) + " Res: " + str(round(sum, 2)))

intercept = 0.95
# getSlopeOfLossFunction(weights, heights, intercept)
