# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset containing tumor features and their corresponding labels .
2. Split the dataset into training data and testing data.
3. Train the Decision Tree model using the training dataset.
4. Test the model and evaluate accuracy by predicting tumor types on the testing dataset.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data=pd.read_csv('tumor.csv')
print(data.head())
print(data.columns)
X=data.drop(columns=['Class'])
y=data['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Name: KRITHIKAA P")
print("Register Number:212225040193")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))

conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

## Output:
![alt text](<Screenshot 2026-03-15 113715.png>)
![alt text](<Screenshot 2026-03-15 113739.png>)
![alt text](<Screenshot 2026-03-15 113822.png>)

## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
