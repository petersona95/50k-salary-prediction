import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

og_train_df = pd.read_csv('Projects/income census data/adult.data',
                       names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 
                                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50k'])
og_test_df = pd.read_csv('Projects/income census data/adult.test',
                       names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 
                                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50k'])

actual = og_test_df['50k']
actual = pd.get_dummies(actual, columns=['50k'], drop_first=True)
actual = actual.rename(index=str, columns={'>50K.': '>50k'})



train_df = og_train_df
test_df = og_test_df


all_data = pd.concat([train_df, test_df])
all_data['50k'].value_counts()

#there are a bunch of them with .'s at the end. gotta remove this trash
all_data['50k'] = all_data['50k'].str.replace('50K.', '50K')


#drop education, its redundant
all_data = all_data.drop('education', axis=1)

#find and remove missing values
all_data = all_data.replace(' ?', np.NaN)
def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    return ms

missingdata(all_data)

#going to have to remove blanks from the train set

#dummy vals
all_data = pd.get_dummies(all_data, columns=['workclass', 'marital-status', 'occupation', 'relationship', 
                                'race', 'sex', '50k'], drop_first=True)

#making native-country US or Non US

all_data['native-country'].value_counts()
all_data['US_Born'] = np.where(all_data['native-country'].str.contains('United-States'), 1, 0)



all_data = all_data.drop('native-country', axis=1)


#rename stupid 50k name
all_data = all_data.rename(index=str, columns={"50k_ >50K": ">50k"})
 
all_data['>50k'].value_counts()
##figure out if theres still missing values, then get rid of them.



#NEED TO SPLIT TRAIN AND TEST!

train_df = all_data.iloc[:32561,:]
test_df = all_data.iloc[32561:,:]

#now remove blanks from the train set
train_df = train_df.dropna()



# =============================================================================
# ALL THE SHIT BELOW IS LOGISTIC REGRESSION COPYPASTE
# =============================================================================
#setting x/y for train/test
from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix

##defining X and Y
X_train = train_df.drop('>50k', axis=1)
y_train = train_df['>50k']
X_test = test_df.drop('>50k', axis=1)
y_test = test_df['>50k']

from sklearn.linear_model import LogisticRegression # Logistic Regression

model = LogisticRegression()
model.fit(X_train,y_train)
prediction_lr=model.predict(X_test)

print('The accuracy of the Logistic Regression is',round(accuracy_score(prediction_lr,y_test)*100,2))
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
result_lr=cross_val_score(model,X_train,y_train,cv=10,scoring='accuracy')
print('The cross validated score for Logistic REgression is:',round(result_lr.mean()*100,2))
y_pred = cross_val_predict(model,X_train,y_train,cv=10)


#this is kinda cool. Shows performance of a model. top left and bottom right are correct guesses, others are incorrect
sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix', y=1.05, size=15)

#predicting test-set
predictions = model.predict(X_test)

#final submission
submission = pd.DataFrame({
        "True Value": y_test,
        "Predicted Value": predictions})

submission.to_csv('outputs/census_prediction.csv')