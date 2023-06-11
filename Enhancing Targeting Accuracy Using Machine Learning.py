

#KNN for Classification- ABC Grocery task

############################


#import required package
#############################

import pandas as pd
import pickle
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import numpy as np


from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV


#import sample data
#############################


data_for_model = pickle.load(open("data/abc_classification_modelling.p", "rb"))

#drop unecessary columns
###############################
data_for_model.drop("customer_id", axis = 1, inplace= True)

############################
#shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)


#class balance
data_for_model["signup_flag"].value_counts(normalize = True)

#############################
#deal with missing values
##############################
#data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)

#deal with outliers

outlier_investigation = data_for_model.describe()
outlier_columns= ["distance_from_store","total_sales", "total_items"]

# boxplot approach

for column in outlier_columns: 
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr *2
    
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers= data_for_model[(data_for_model[column]< min_border)|(data_for_model[column]> max_border)].index
    print(f"{len(outliers)} outliers detected in column{column}")
    data_for_model.drop(outliers, inplace= True)
    
#########################
#Split input variables and output variable
########################

X = data_for_model.drop(["signup_flag"], axis =1)
y = data_for_model["signup_flag"]

# Split out training & test sets
###################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

#deal with categorical variables
##########################################

categorical_vars= ["gender"]
one_hot_encoder = OneHotEncoder(sparse = False, drop = "first")


X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])


encoder_feature_names_out = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded= pd.DataFrame(X_train_encoded, columns = encoder_feature_names_out)
X_train = pd.concat([X_train.reset_index(drop =True), X_train_encoded.reset_index(drop=True)], axis = 1)                 
X_train.drop(categorical_vars, axis = 1, inplace = True)


X_test_encoded= pd.DataFrame(X_test_encoded, columns = encoder_feature_names_out)
X_test = pd.concat([X_test.reset_index(drop =True), X_test_encoded.reset_index(drop=True)],axis = 1)                 
X_test.drop(categorical_vars, axis = 1, inplace = True)


########################
# Feature scaling
######################

scale_norm = MinMaxScaler()                          
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scale_norm.transform(X_test), columns = X_test.columns)

########################
# Feature selection
######################
from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(random_state = 42)
feature_selector = RFECV(clf)

fit = feature_selector.fit(X_train, y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features:{optimal_feature_count}")

X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

plt.plot(range(1, len(fit.grid_scores_) +1), fit.grid_scores_, marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Numer of features")
plt.title(f"Feature selection using RFE \n Optimal number of features is {optimal_feature_count}")
plt.tight_layout()
plt.show()

################################
#Model training
##############################
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
################################
#Model Assesment
######################################
y_pred_class=  clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]

#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

#Accuracy (the number of correct classification out of all attempted classifications)

accuracy_score(y_test, y_pred_class)


# Precision(of all observation that were as positive, how many were actually positive)

precision_score(y_test, y_pred_class)

#Recall (of all positive observations, how many did we predict as positive)
recall_score(y_test, y_pred_class)

#F1- score (the harmonic mean of precision and recall)
f1_score(y_test, y_pred_class)
##########################
#Finding the optimal value of k
###########################

k_list= list(range(2,25))
accuracy_scores = []

for k in k_list:
    
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_k_value = k_list[max_accuracy_idx]

#plot of max_depth

plt.plot(k_list, accuracy_scores)
plt.scatter(optimal_k_value, max_accuracy, marker ="x", color ="red")
plt.title(f"Accuracy(F1 Score) by k\n Optimal value for k:{(optimal_k_value)}(Accuracy:{round(max_accuracy,4)})")
plt.xlabel("k")
plt.ylabel("Accuracy(F1 Score)")
plt.tight_layout()
plt.show()


