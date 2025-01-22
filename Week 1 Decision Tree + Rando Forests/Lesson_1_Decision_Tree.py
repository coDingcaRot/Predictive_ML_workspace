##############
# EXERCISE 4 #
##############
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from sklearn.metrics import confusion_matrix


def getBrowser():
    options = Options()

    # this parameter tells Chrome that
    # it should be run without UI (Headless)
    # Uncommment this line if you want to hide the browser.
    # options.add_argument('--headless=new')

    try:
        # initializing webdriver for Chrome with our options
        browser = webdriver.Chrome(options=options)
        print("Success.")
    except:
        print("It failed.")
    return browser
browser = getBrowser()

import time
from bs4 import BeautifulSoup
import re

URL = "https://vpl.bibliocommons.com/events/search/index"
browser.get(URL)

# Give the browser time to load all content.
time.sleep(3)

data = browser.find_elements(By.CSS_SELECTOR, ".cp-events-search-item")

def getText(content):
    innerHtml = content.get_attribute('innerHTML')

    # Beautiful soup allows us to remove HTML tags from our content.
    soup = BeautifulSoup(innerHtml, features="lxml")
    rawString = soup.get_text()

    # Remove hidden carriage returns and tabs.
    textOnly = re.sub(r"[\n\t]*", "", rawString)
    # Replace two or more consecutive empty spaces with '*'
    textOnly = re.sub('[ ]{2,}', ' ', textOnly)

    return textOnly
import sklearn.datasets as datasets
from   sklearn.tree import DecisionTreeClassifier
from   sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# check for the sklearn version, it has to be 0.21
import sklearn
print(sklearn.__version__)
breast_cancer = datasets.load_breast_cancer()

import pandas as pd
dfX =  pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
dfy = breast_cancer.target
print(dfX.head())
print(dfy)

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    dfX, dfy, test_size=0.20,random_state=0)

#Create predictions
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

def showAccuracyScores(y_test, y_pred):
    print("\nModel Evaluation")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("")
    tn = cm[0][0]
    fp = cm[0][1]
    tp = cm[1][1]
    fn = cm[1][0]
    accuracy  = (tp + tn)/(tn + fp + tp + fn)
    precision = tp/(tp + fp)
    recall    = tp/(tp + fn)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))

showAccuracyScores(y_test, y_pred)

from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(20, 10))

plot_tree(classifier.fit(X_train, y_train), max_depth=2, fontsize=4)
a = plot_tree(classifier,
              feature_names=breast_cancer.feature_names,
              class_names=breast_cancer.target_names,
              filled=True,
              rounded=True,
              fontsize=14)
plt.show()
