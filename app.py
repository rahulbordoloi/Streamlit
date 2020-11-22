# Importing Libraries
import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Functions

## Function for Selecting Dataset Names
def getDataset(name):

    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif name == "Wine Dataset":
        data = datasets.load_wine()

    x, y = data.data, data.target
    return x, y


## Function for Adding Parameters to our Models
def addParameter(clf_name):

    params = dict()
    if clf_name == 'Logistic Regression':
        pass
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params


## Function to Return Selected Classifier with Hyperparamters
def getClassifier(clf_name, params, y):

    yy = np.array(y)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(yy.reshape(-1, )), yy.reshape(-1, ))
    im_weight = dict(enumerate(class_weights))
    classifier = None
    if clf_name == 'Logistic Regression':
        classifier = LogisticRegression(random_state = 0, class_weight = im_weight, n_jobs = -1)
    elif clf_name == 'SVM':
        classifier = SVC(C = params['C'], class_weight = im_weight, random_state = 0)
    else:
        classifier = RandomForestClassifier(n_estimators = params['n_estimators'], max_depth = params['max_depth'],
                                           random_state = 0, n_jobs = -1, class_weight = im_weight)
    return classifier


# Main Method
if __name__ == '__main__':

    # Title
    st.title("Streamlit ML Demo Deployment")
    st.write("""
# Exploring Different ML Classifiers.
Which one is the Best?
""")

    # Selecting off Dataset & Classifiers from USER
    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
    classifier_name = st.sidebar.selectbox("Select Classifier", ("Logistic Regression", "SVM", "Random Forest"))

    # Getting Information
    x, y = getDataset(dataset_name)
    st.write(f'Shape of Dataset: {x.shape}')
    st.write(f'Number of Classes: {len(np.unique(y))}')

    # Getting Parameters of the Selected Model
    params = addParameter(classifier_name)

    # Getting our Desired Classifier
    model = getClassifier(classifier_name, params, y)

    # ------ Classification ------

    ## Feature Scaling [Standard Scaling our Features]
    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)

    ## Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size = 0.2, random_state = 0, stratify = y,
                                                        shuffle = True)

    ## Training our AI Model
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    ## Giving out the Accuracy Score
    acc = accuracy_score(y_test, y_pred)
    st.write(f'Classifier : {classifier_name}')
    st.write(f'Accuracy : {acc}')

    # ------ Plotting Dataset ------

    ## Projecting the Data onto the 2 Primary Principal Components
    pca = PCA(n_components = 2, random_state = 0)
    x_projected = pca.fit_transform(x)

    ## Taking out the Principal Components
    x1 = x_projected[:, 0]
    x2 = x_projected[:, 1]

    ## Plotting the Same
    fig = plt.figure()
    plt.scatter(x1, x2, c = y, alpha = 0.8, cmap = 'viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    st.pyplot(fig)


