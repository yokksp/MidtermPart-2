from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

File_path = '/Users/sirirat/Downloads/'
File_name =  'car_data.csv'

df = pd.read_csv(File_path + File_name)

df.drop(columns=['User ID'],inplace=True)
encoders = []
for i in range(0, len(df.columns) - 1):
    enc = LabelEncoder()
    df.iloc[:,i] = enc.fit_transform(df.iloc[:, i])
    encoders.append(enc)
 
x = df[['Gender','Age','AnnualSalary']]
y = df['Purchased']
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)


model = DecisionTreeClassifier(criterion='entropy') #'gini'
model.fit(x, y)

x_pred = ['Male',50,50000]
for i in range(0, len(df.columns) - 1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
    
x_pred_adj = np.array(x_pred).reshape(-1, 3 )

y_pred = model.predict(x_pred_adj)
print('Prodiction:' , y_pred[0])
score = model.score(x, y)
print('Accuracy:' , '{: .2f}' .format(score))

feature = x.columns.tolist()
Data_class = y.tolist()

plt.figure(figsize=(25,20))
_= plot_tree(model,
             feature_names = feature,
             class_names = Data_class,
             label = 'all',
             impurity = True,
             precision = 3,
             filled = True,
             fontsize = 16)

plt.show()

import seaborn as sns
feature_importances = model.feature_importances_
feature_names = ['Gender' ,'Age','AnnualSalary']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x = feature_importances, y =feature_names)

print(feature_importances)

