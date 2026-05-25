import pandas as pd
import numpy as np
from data_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV



df = pd.read_csv("Titanic-Dataset.csv")
print(df.head())

print(df.isnull().sum())
print(df.columns)

profile = ProfileReport(df , title="report")
profile.to_file("report.html")
df["with_family"] = df["SibSp"] + df["Parch"]
df = df.drop(["PassengerId","Name","SibSp","Parch","Ticket"],axis=1)





X = df.drop(["Survived"],axis=1)
y = df["Survived"]

X["Cabin"] = X["Cabin"].str[0]


X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)


pipe1 = Pipeline([
    ("impute",SimpleImputer(strategy='most_frequent',add_indicator=True)),
    ("encod",OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
    
])

##### filing empty age
pipe2 = Pipeline([
    ("encod",OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
    
])



pipe3 = Pipeline([
    ("int", SimpleImputer(strategy='median')),
    ("fare",StandardScaler())
    
])




trf1 = ColumnTransformer([
     ("trf2",pipe1,["Cabin"]),
    ("trf3",pipe2,["Pclass","Embarked","Sex"]),
    ("trf4",pipe3,["Age","Fare"]),
],remainder='passthrough')




pip_logistic = make_pipeline(
    trf1,
    LogisticRegression(max_iter=1000),
   
)

pip_random = make_pipeline(
    trf1,
    RandomForestClassifier(random_state=42)

)

pip_knn = make_pipeline(
    trf1,
    KNeighborsClassifier()

)

pip_knn.fit(X_train,y_train)
prediction3 = pip_knn.predict(X_test)
accu3 = accuracy_score(y_test,prediction3)



pip_logistic.fit(X_train,y_train)

prediction = pip_logistic.predict(X_test)
accu = accuracy_score(y_test,prediction)

print("Accuracy for Logistic Regression:->",accu)


pip_random.fit(X_train,y_train)
prediction2 = pip_random.predict(X_test)
accu2 = accuracy_score(y_test,prediction2)
print("Accuracy for randomforest:->",accu2)


print("Accuracy for KNN:->",accu3)

###### Confusion Matrix
def ConfusionMatrix(y_test,pred):
    cm_fun = confusion_matrix(y_test,pred)
    return cm_fun

def precision(y_test,pred):
    prec_fun = precision_score(y_test,pred)
    return prec_fun

def recall(y_test,pred):
    recall_fun = recall_score(y_test,pred)
    return recall_fun

def f1(y_test,pred):
    f1_value = f1_score(y_test,pred)
    return f1_value

print("Confusion Matrix for Logistic Regression: ",ConfusionMatrix(y_test,prediction))
print("Recall for Logistic Regression: ", recall(y_test,prediction))
print("Precision for Logistic Regression: ",precision(y_test,prediction))
print("f1 of logistic: ",f1(y_test,prediction))


print("Confusion Matrix for KNN: ",ConfusionMatrix(y_test,prediction3))
print("Recall for KNN: ", recall(y_test,prediction3))
print("Precision for KNN: ",precision(y_test,prediction3))
print("f1 of KNN: ",f1(y_test,prediction3))



print("Confusion Matrix for Random_Forest: ",ConfusionMatrix(y_test,prediction2))
print("Recall for Random_Forest: ", recall(y_test,prediction2))
print("Precision for Random_Forest: ",precision(y_test,prediction2))
print("f1 of Random_Forest: ",f1(y_test,prediction2))





print("-------------------Applyling Gridsearch------------------")

 #print(pip_random.get_params().keys())
param_grid = {
     'randomforestclassifier__n_estimators':[100,200,300,500],
     'randomforestclassifier__max_depth':[3,5,7,9],
     'randomforestclassifier__min_samples_split':[2,5,10,12],
    'randomforestclassifier__min_samples_leaf':[1,4,6,8]

}

grid = GridSearchCV(
     estimator=pip_random,
     param_grid = param_grid,
     cv = 5,
     scoring = 'accuracy',
     n_jobs = -1,
 )

grid.fit(X_train,y_train)
print("BEST Hyperprameter tunning for random forest: ",grid.best_params_)

best_model_random_forest = grid.best_estimator_
y_pred_random = best_model_random_forest.predict(X_test)
print(y_pred_random)

print(accuracy_score(y_test,y_pred_random))

print("Grid search for KNN")
 

param_grid_knn = {
     'kneighborsclassifier__n_neighbors':[3,5,7,9,11],
     'kneighborsclassifier__weights': ['uniform', 'distance'],
     'kneighborsclassifier__metric': ['euclidean', 'manhattan']

 }

grid_knn = GridSearchCV(
     estimator=pip_knn,
     param_grid=param_grid_knn,
     cv = 5,
     scoring='accuracy',
     n_jobs=-1
 )

grid_knn.fit(X_train,y_train)
print("KNN best Parameter:-", grid_knn.best_params_)
best_model_knn = grid_knn.best_estimator_
y_pred_knn = best_model_knn.predict(X_test)
print("Best accuracy for KNN",accuracy_score(y_test,y_pred_knn))




















































