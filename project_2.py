import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split    #import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics #metrics module for accuracy calculation


existing_emp = pd.ExcelFile('TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx')

df2 = existing_emp.parse('Existing employees')
print df2

#Building a Prediction Model

#creating LabelEncoder
lab_en = preprocessing.LabelEncoder()
#converting string labels into number
df2['salary'] = lab_en.fit_transform(df2['salary'])
df2['dept'] = lab_en.fit_transform(df2['dept'])


#split into feature and target variables
X = df2[['satisfaction_level','last_evaluation','number_project',
         'average_montly_hours', 'time_spend_company', 'Work_accident',
         'promotion_last_5years','dept']]
y = df2['salary']

#Split dataset into the training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#Prediction Model Building using Gradient Boosting Classifier

#create Gradient Boosting Classifier
gb = GradientBoostingClassifier()

#Train the model using the training sets
gb = gb.fit(X_train, y_train)

#predict the response for test dataset
y_pred = gb.predict(X_test)

#Model Accuracy
print 'Accuracy:',metrics.accuracy_score(y_test, y_pred)

#Model Precision
#print('Precision:',metrics.precision_score(y_test,y_pred))
