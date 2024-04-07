#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets, metrics
import time

# Define the Streamlit app
def app():

    st.subheader('Supervised Learning, Classification, and KNN with Wine Dataset')
    text = """**Supervised Learning:**
    \nSupervised learning is a branch of machine learning where algorithms learn from labeled data. 
    This data consists of input features (X) and corresponding outputs or labels (y). The algorithm learns a 
    mapping function from the input features to the outputs, allowing it to predict the labels for 
    unseen data points.
    \n**Classification:**
    Classification is a specific task within supervised learning where the labels belong to discrete 
    categories. The goal is to build a model that can predict the category label of a new data 
    point based on its features.
    \n**K-Nearest Neighbors (KNN):**
    KNN is a simple yet powerful algorithm for both classification and regression tasks. 
    \n**The Wine Dataset:**
    The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators. 
    There are thirteen different measurements taken for different constituents found in the three types of wine.
    https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset
    1.Alcohol
    2.Malic acid
    3.Ash
    4.Alcalinity of ash
    5.Magnesium
    6.Total phenols
    7.Flavanoids
    8.Nonflavanoid phenols
    0.Proanthocyanins
    10.Color intensity
    11.Hue
    12.OD280/OD315 of diluted wines
    13.Proline
    \n**KNN Classification with Wine:**
    \n1. **Training:**
    * The KNN algorithm stores the entire Wine dataset (features and labels) as its training data.
    \n2. **Prediction:**
    * When presented with a new wine, KNN calculates the distance (often Euclidean distance) 
    between this wine's features and all the flowers in the training data.
    * The user defines the value of 'k' (number of nearest neighbors). KNN identifies the 'k' closest 
    data points (flowers) in the training set to the new flower.
    * KNN predicts the class label (species) for the new wine based on the majority vote among its 
    'k' nearest neighbors. For example, if three out of the five nearest neighbors belong to class_1, 
    the new wine is classified as class_1.
    **Choosing 'k':**
    The value of 'k' significantly impacts KNN performance. A small 'k' value might lead to overfitting, where the 
    model performs well on the training data but poorly on unseen data. Conversely, a large 'k' value might not 
    capture the local patterns in the data and lead to underfitting. The optimal 'k' value is often determined 
    through experimentation.
    \n**Advantages of KNN:**
    * Simple to understand and implement.
    * No complex model training required.
    * Effective for datasets with well-defined clusters."""
    st.write(text)
    k = st.sidebar.slider(
        label="Select the value of k:",
        min_value= 2,
        max_value= 10,
        value=5,  # Initial value
    )

    if st.button("Begin"):
        wine = datasets.load_wine()
        X = wine.data 
        y = wine.target  
        
        # KNN for supervised classification (reference for comparison)
        
        # Define the KNN classifier with k=5 neighbors
        knn = KNeighborsClassifier(n_neighbors=k)
        
        # Train the KNN model
        knn.fit(X, y)
        
        # Predict the cluster labels for the data
        y_pred = knn.predict(X)
        #st.write('Confusion Matrix')
        cm = confusion_matrix(y, y_pred)
        #st.text(cm)
        #st.subheader('Performance Metrics')
        #st.text(classification_report(y, y_pred))
        
        # Get unique class labels and color map
        unique_labels = list(set(y_pred))
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for label, color in zip(unique_labels, colors):
            indices = y_pred == label
            # Use ax.scatter for consistent plotting on the created axis
            ax.scatter(X[indices, 0], X[indices, 1], label=wine.target_names[label], c=color)
        
        # Add labels and title using ax methods
        ax.set_xlabel('Alcohol')
        ax.set_ylabel('Malic acid')
        ax.set_title('Alcohol vs Malic Acid Colored by Predicted Wine Class')
        
        # Add legend and grid using ax methods
        ax.legend()
        ax.grid(True)
        

# Display the plot


        # Add legend and grid using ax methods
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', 
                    xticklabels=wine.target_names, 
                    yticklabels=wine.target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        st.pyplot(plt)
#run the app
if __name__ == "__main__":
    app()
