import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.title("📚 Learn Machine Learning Models")

model_choice = st.selectbox(
    "Choose a model to learn:",
    ["Logistic Regression","SVM","KNN","Decision Tree"]
)

st.write("---")

# Dataset for simulation
X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# 🎓 THEORY SECTION
if model_choice == "Logistic Regression":
    st.header("What is Logistic Regression?")
    st.write("Used for classification. Creates a linear decision boundary.")

    C = st.slider("Regularization Strength (C)",0.01,10.0,1.0)

    model = LogisticRegression(C=C)

elif model_choice == "SVM":
    st.header("What is SVM?")
    st.write("Finds best boundary with maximum margin.")

    C = st.slider("C (Regularization)",0.01,10.0,1.0)
    gamma = st.slider("Gamma",0.01,5.0,1.0)

    model = SVC(C=C,gamma=gamma)

elif model_choice == "KNN":
    st.header("What is KNN?")
    st.write("Classifies based on nearest neighbors.")

    k = st.slider("Number of Neighbors",1,20,5)
    model = KNeighborsClassifier(n_neighbors=k)

elif model_choice == "Decision Tree":
    st.header("What is Decision Tree?")
    st.write("Splits data using tree structure.")

    depth = st.slider("Max Depth",1,10,3)
    model = DecisionTreeClassifier(max_depth=depth)


# 🎛 TRAIN MODEL
model.fit(X_train,y_train)

# 📊 PLOT FUNCTION
def plot_boundary(model,X,y):
    h=0.02
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx,yy,Z,alpha=0.3)
    plt.scatter(X[:,0],X[:,1],c=y)
    st.pyplot(plt)

st.subheader("🔍 Decision Boundary Simulation")
plot_boundary(model,X,y)