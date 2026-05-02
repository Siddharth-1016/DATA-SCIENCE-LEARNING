import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("⚔️ Model Battle Arena")

# ⭐ Page config (browser tab + layout)
st.set_page_config(
    page_title="ML Model Hub",
    page_icon="🤖",
    layout="wide"
)

from sklearn.datasets import (
    make_moons, make_circles, make_blobs,
    make_classification, make_gaussian_quantiles,
    load_breast_cancer, load_wine, load_iris, load_digits
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# ================= SIDEBAR =================
st.sidebar.title("ML Model Hub – Compare Models on Multiple Datasets")

dataset_name = st.sidebar.selectbox(
    "Dataset",
    (
        "Moons", "Circles", "Blobs", "Linearly Separable",
        "Breast Cancer", "Wine", "Iris", "Digits",
        "Noisy Moons", "Noisy Circles", "XOR", "Gaussian Quantiles"
    )
)

models_selected = st.sidebar.multiselect(
    "Estimators",
    [
        "Logistic Regression","SVM","KNN","Decision Tree",
        "Random Forest","Gradient Boosting","AdaBoost",
        "Naive Bayes","SGD Classifier","Neural Network (MLP)"
    ],
    default=["Logistic Regression", "SVM", "Random Forest"]
)

voting_type = st.sidebar.radio("Voting Type", ("hard", "soft"))


# ================= DATASETS =================
def get_dataset(name):

    if name == "Moons":
        X, y = make_moons(n_samples=400, noise=0.25, random_state=42)

    elif name == "Noisy Moons":
        X, y = make_moons(n_samples=400, noise=0.35, random_state=42)

    elif name == "Circles":
        X, y = make_circles(n_samples=400, noise=0.2, factor=0.5, random_state=42)

    elif name == "Noisy Circles":
        X, y = make_circles(n_samples=400, noise=0.3, factor=0.4, random_state=42)

    elif name == "Blobs":
        X, y = make_blobs(n_samples=400, centers=2, random_state=42)

    elif name == "Linearly Separable":
        X, y = make_classification(
            n_samples=400, n_features=2, n_redundant=0,
            n_informative=2, n_clusters_per_class=1, random_state=42
        )

    elif name == "Gaussian Quantiles":
        X, y = make_gaussian_quantiles(n_samples=400, n_features=2, random_state=42)

    elif name == "XOR":
        X = np.random.randn(400, 2)
        y = np.logical_xor(X[:,0] > 0, X[:,1] > 0)

    elif name == "Breast Cancer":
        data = load_breast_cancer()
        X, y = data.data[:, :2], data.target

    elif name == "Wine":
        data = load_wine()
        X, y = data.data[:, :2], data.target

    elif name == "Iris":
        data = load_iris()
        X, y = data.data[:, :2], data.target

    elif name == "Digits":
        data = load_digits()
        X, y = data.data[:, :2], data.target

    return X, y


X, y = get_dataset(dataset_name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ================= MODELS =================
estimators = []

if "Logistic Regression" in models_selected:
    estimators.append(("lr", LogisticRegression()))

if "SVM" in models_selected:
    estimators.append(("svm", SVC(probability=True)))

if "KNN" in models_selected:
    estimators.append(("knn", KNeighborsClassifier()))

if "Decision Tree" in models_selected:
    estimators.append(("dt", DecisionTreeClassifier()))

if "Random Forest" in models_selected:
    estimators.append(("rf", RandomForestClassifier()))

if "Gradient Boosting" in models_selected:
    estimators.append(("gb", GradientBoostingClassifier()))

if "AdaBoost" in models_selected:
    estimators.append(("ada", AdaBoostClassifier()))

if "Naive Bayes" in models_selected:
    estimators.append(("nb", GaussianNB()))

if "SGD Classifier" in models_selected:
    estimators.append(("sgd", SGDClassifier(loss="log_loss")))

if "Neural Network (MLP)" in models_selected:
    estimators.append(("mlp", MLPClassifier(max_iter=1000)))


# ================= MAIN UI =================
if len(estimators) > 0:

    voting_clf = VotingClassifier(estimators=estimators, voting=voting_type)
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # ⭐ New Professional Title Section
    st.title("ML Model Hub – Decision Boundary Explorer")

    st.markdown(
        f"""
        ### Dataset: **{dataset_name}**
        Compare how multiple machine learning models work together using a **Voting Classifier**.
        """
    )

    st.success(f"Voting Accuracy: {round(acc,3)}")

    st.write("### Selected Models:")
    st.write(", ".join(models_selected))


    # ================= PLOT =================
    st.subheader("Decision Boundary Visualization")

    def plot_decision_boundary(model, X, y):
        h = 0.02
        x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
        y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:,0], X[:,1], c=y)
        st.pyplot(plt)

    plot_decision_boundary(voting_clf, X, y)

else:
    st.warning("Please select at least one model from the sidebar.")