import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

employees = pd.ExcelFile('TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx')

df1 = employees.parse('Employees who have left')
#print df1

print df1.info()

salary = df1.groupby('salary')
print salary.mean()

print df1.describe()

#TRYING TO DETERMINE WHAT TYPES OF EMPLOYEES ARE LEAVING

#using matplotlib to plot bar graph to show discrete variable count

#Considering Salary
salary_count = df1.groupby('salary').count()
plt.bar(salary_count.index.values, salary_count['satisfaction_level'])
plt.xlabel('Salary Scale of Employees')
plt.ylabel('Employees Left Company')
#plt.show()
plt.savefig('Salary.png')

print df1.salary.value_counts()

#Considering number of projects
num_projects = df1.groupby('number_project').count()
plt.bar(num_projects.index.values, num_projects['satisfaction_level'])
plt.xlabel('Number of projects')
plt.ylabel('Employees Left Company')
#plt.show()
plt.savefig('Number_of_projects.png')

# #Considering time spent in the company
time_spent = df1.groupby('time_spend_company').count()
plt.bar(time_spent.index.values, time_spent['satisfaction_level'])
plt.xlabel('Time spent at the company')
plt.ylabel('Employees Left Company')
#plt.show()
plt.savefig('Time_spent.png')

#subplots using seaborn
features = ['number_project','time_spend_company','Work_accident', 'promotion_last_5years','dept','salary']
fig = plt.subplots(figsize=(10,15))
for i,j in enumerate(features):
    plt.subplot(6, 2, i+1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j,data=df1)
    plt.xticks(rotation=90)
    plt.title('Employees that left')
    plt.savefig('subplots.png')


# #using cluster analysis to find the groups of employees that left
# #Filter data
# emp_left = df1[['satisfaction_level', 'last_evaluation']]
# #create groups using K-Means clustering
# kmeans = KMeans(n_clusters= 2, random_state= 0).fit(emp_left)
# #Add new column 'label' and assign cluster labels
# emp_left['label'] = kmeans.labels_
# #draw scatter plot
# plt.scatter(emp_left['satisfaction_level'], emp_left['last_evaluation'], c=emp_left['label'],cmap='Accent')
# plt.xlabel('Satisfaction Level')
# plt.ylabel('Last Evaluation')
# plt.title('3 Clusters of Employees who left')
# #plt.show()
# plt.savefig('cluster.png')

