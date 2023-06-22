#PCA
#Principal Component Analysis (PCA) is a "Dimensionality Reduction" technique

# rules to apply PCA
#always apply scaling to your variables, use standarisation instead of normalisation
#you will lose some of the information/ variance contained in the original data
#it wll be harder to interpret the outputs based on component values versus the original varaiables

##############################
#PCA code template
##############################


# Random Forest Classification

############################


#import required package
#############################

import pandas as pd
import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



#import sample data
#############################

data_for_model = pd.read_csv("data/sample_data_pca.csv")

#drop unecessary columns
###############################
data_for_model.drop("user_id", axis = 1, inplace= True)

############################
#shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

#class balance
data_for_model["purchased_album"].value_counts(normalize = True)

#############################
#deal with missing values
##############################
data_for_model.isna().sum().sum()
data_for_model.dropna(how = "any", inplace = True)

#########################
#Split input variables and output variable
########################

X = data_for_model.drop(["purchased_album"], axis =1)
y = data_for_model["purchased_album"]

# Split out training & test sets
###################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)



###############################
#feature scaling
###############################

scale_standard = StandardScaler()

X_train = scale_standard.fit_transform(X_train)
X_test = scale_standard.fit_transform(X_test)


##############################
#Apply PCA
##############################
#Instantiate & fit 

pca = PCA(n_components = None, random_state = 42)

pca.fit(X_train)

#extract the explained variance across components

explained_variance = pca.explained_variance_ratio_
explained_variance_cumulative = pca.explained_variance_ratio_.cumsum()

################################
#Plot the explained variance
#################################

#create list for number of components

num_vars_list = list(range(1,101))
plt.figure(figsize = (15,10))

#plot the variance explained by each component

plt.subplot(2,1,1)
plt.bar(num_vars_list,explained_variance)
plt.title("Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("% Variance")
plt.tight_layout()


#plot the cumulative variance
plt.subplot(2,1,2)
plt.plot(num_vars_list,explained_variance_cumulative)
plt.title("Cumulative Variance across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative % Variance")
plt.tight_layout()
plt.show()

###########################
#Apply PCA with selected number of components
###########################
pca = PCA(n_components = 0.75, random_state = 42)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

pca.n_components_


#################
#Apply PCA with selected number of components
#################

clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train, y_train)


########################
#Access Model Accuracy
########################

y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)


#PCA 