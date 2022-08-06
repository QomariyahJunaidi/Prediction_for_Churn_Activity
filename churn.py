# import streamlit as st 
# import numpy as np 
# import pandas as pd

# #import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler

# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import classification_report



# st.title('Machine Learning - CLASSIFICATION')

# st.sidebar.write("""
# This is a web app demo using python libraries such as Streamlit, Sklearn etc
# """)

# st.sidebar.write ("For more info, please contact:")

# st.sidebar.write("<a href='https://www.linkedin.com/in/yong-poh-yu/'>Dr. Yong Poh Yu </a>", unsafe_allow_html=True)


# choice = st.sidebar.radio(
#     "Choose a dataset",   
#     ('Default', 'User-defined '),
#     index = 0
    
# )

# st.write(f"## You Have Selected <font color='Aquamarine'>{choice}</font> Dataset", unsafe_allow_html=True)

# def get_default_dataset(name):
#     data = None
#     if name == 'Iris':
#         data = datasets.load_iris()
#     elif name == 'Wine':
#         data = datasets.load_wine()
#     else:
#         data = datasets.load_breast_cancer()
#     X = data.data
#     y = data.target
#     return X, y

# def add_dataset_ui(choice_name):
#     X=[]
#     y=[]
#     X_names = []
#     X1 = []
#     if choice_name == 'Default':
#        dataset_name = st.sidebar.selectbox(
#             'Select Dataset',
#             ('Iris', 'Breast Cancer', 'Wine')
#         )
#        X, y = get_default_dataset (dataset_name)
#        X_names = X
#     else:
#         uploaded_file = st.sidebar.file_uploader(
#             "Upload a CSV",
#             type='csv'    )
        

#         if uploaded_file!=None:
           
#            st.write(uploaded_file)
#            data = pd.read_csv(uploaded_file)
  
        
#            y_name = st.sidebar.selectbox(
#                     'Select Label @ y variable',
#                     sorted(data)
#                     )

#            X_names = st.sidebar.multiselect(
#                      'Select Predictors @ X variables.',
#                      sorted(data),
#                      default = sorted(data)[1],
#                      help = "You may select more than one predictor"
#                      )

#            y = data.loc[:,y_name]
#            X = data.loc[:,X_names]
#            X1 = X.select_dtypes(include=['object'])
        
#            X2 = X.select_dtypes(exclude=['object'])

#            if sorted(X1) != []:
#               X1 = X1.apply(LabelEncoder().fit_transform)
#               X = pd.concat([X2,X1],axis=1)

#            y = LabelEncoder().fit_transform(y)
#         else:
#            st.write(f"## <font color='Aquamarine'>Note: Please upload a CSV file to analyze the data.</font>", unsafe_allow_html=True)

#     return X,y, X_names, X1

# X, y , X_names, cat_var= add_dataset_ui (choice)




# classifier_name = st.sidebar.selectbox(
#     'Select classifier',
#     ('KNN', 'SVM', 'Random Forest')
# )

# test_data_ratio = st.sidebar.slider('Select testing size or ratio', 
#                                     min_value= 0.10, 
#                                     max_value = 0.50,
#                                     value=0.2)
# random_state = st.sidebar.slider('Select random state', 1, 9999,value=1234)

# st.write("## 1: Summary (X variables)")


# if len(X)==0:
#    st.write("<font color='Aquamarine'>Note: Predictors @ X variables have not been selected.</font>", unsafe_allow_html=True)
# else:
#    st.write('Shape of predictors @ X variables :', X.shape)
#    st.write('Summary of predictors @ X variables:', pd.DataFrame(X).describe())

# st.write("## 2: Summary (y variable)")

# if len(y)==0:
#    st.write("<font color='Aquamarine'>Note: Label @ y variable has not been selected.</font>", unsafe_allow_html=True)
# elif len(np.unique(y)) <5:
#      st.write('Number of classes:', len(np.unique(y)))

# else: 
#    st.write("<font color='red'>Warning: System detects an unusual number of unique classes. Please make sure that the label @ y is a categorical variable. Ignore this warning message if you are sure that the y is a categorical variable.</font>", unsafe_allow_html=True)
#    st.write('Number of classes:', len(np.unique(y)))


# def add_parameter_ui(clf_name):
#     params = dict()
#     if clf_name == 'SVM':
#         C = st.sidebar.slider('C', 0.01, 10.0,value=1.0)
#         params['C'] = C
#     elif clf_name == 'KNN':
#         K = st.sidebar.slider('K', 1, 15,value=5)
#         params['K'] = K
#     else:
#         max_depth = st.sidebar.slider('max_depth', 2, 15,value=5)
#         params['max_depth'] = max_depth
#         n_estimators = st.sidebar.slider('n_estimators', 1, 100,value=10)
#         params['n_estimators'] = n_estimators
#     return params

# params = add_parameter_ui(classifier_name)

# def get_classifier(clf_name, params):
#     clf = None
#     if clf_name == 'SVM':
#         clf = SVC(C=params['C'])
#     elif clf_name == 'KNN':
#         clf = KNeighborsClassifier(n_neighbors=params['K'])
#     else:
#         clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
#             max_depth=params['max_depth'], random_state=random_state)
#     return clf

# clf = get_classifier(classifier_name, params)


# st.write("## 3: Classification Report")

# if len(X)!=0 and len(y)!=0: 


#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_ratio, random_state=random_state)

#   scaler = StandardScaler()
#   X_train_scaled = scaler.fit_transform(X_train)
#   X_test_scaled = scaler.transform(X_test)    

#   clf.fit(X_train_scaled, y_train)
#   y_pred = clf.predict(X_test_scaled)


#   st.write('Classifier:',classifier_name)
#   st.write('Classification report:')
#   report = classification_report(y_test, y_pred,output_dict=True)
#   df = pd.DataFrame(report).transpose()
#   st.write(df)

# else: 
#    st.write("<font color='Aquamarine'>Note: No classification report generated.</font>", unsafe_allow_html=True)


# st.write("## 4: Principal Component Analysis Plot")
# suitable = 1
# if len(X_names) <2:
#   st.write("<font color='Aquamarine'>Note: No PCA plot as it requires at least two predictors.</font>", unsafe_allow_html=True)
#   suitable = 0
# else:
#     for names in X_names:
#         if names in cat_var:
#            st.write("<font color='Aquamarine'>Note: No PCA plot as it only supports numerical predictors.</font>", unsafe_allow_html=True)
#            suitable = 0
#            break

# if suitable == 1:
#    pca = PCA(2)
#    X_projected = pca.fit_transform(X)

#    x1 = X_projected[:, 0]
#    x2 = X_projected[:, 1]

#    fig = plt.figure()
#    plt.scatter(x1, x2,
#                c=y, alpha=0.8,
#                cmap='viridis')

#    plt.xlabel('Principal Component 1')
#    plt.ylabel('Principal Component 2')
#    plt.colorbar()
#    st.pyplot(fig)
            
            
# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
