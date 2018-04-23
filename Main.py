import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, recall_score
from sklearn.metrics import r2_score,f1_score, classification_report

# LOADING THE DATASET

names=['State','Education','Gender','Cause','Age','Prone']
data = pd.read_csv("/Volumes/Krunal/downloads/Suicidedatasetcleaned.csv",names=names)
ds=data[:10000]
print(data.shape)
print(data.head(5))

#ENCODING THE LABELS TO NUMERIC VALUES 
le=LabelEncoder()
for col in ds.columns:
    ds[col]=le.fit_transform(ds[col])
sns.heatmap(ds.corr(),annot=True)

#SETTING FEATURES AND TARGET COLUMNS 
input_cols=['State','Education','Gender','Cause','Age']
output_cols=['Prone']
X=ds[input_cols]
Y=ds[output_cols]
ds.head()

#SPLITTING DATA INTO TRAIN AND TEST
split=int(0.6*ds.shape[0])
X_train=X[:split]
y_train=Y[:split]
X_test=X[split:]
y_test=Y[split:]
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)



#RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=1, criterion="gini", min_samples_split=50, random_state=5)
clf.fit(X_train,y_train)
clf.score(X_train,y_train)
algo1=clf.predict(X_test)

#QUALITY METRIC SCORES

print("Precision score:")  
print(precision_score(algo1,y_test))
print("R2 Score")  
print(r2_score(algo1,y_test))
print("Accuracy score")
print(accuracy_score(algo1, y_test))
print("F1 score")
print(f1_score(algo1, y_test))
print("Recall Score")
recall_score(algo1, y_test)
print("Overall Summary")
print(classification_report(algo1,y_test))

'''
INTERPRETATOINS
'''


#ROC CURVE

plt.title('Receiver Operating Characteristic')
fpr,tpr, treshold=roc_curve(y_test,algo1)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


'''
actual=algo1[:100]
pred=y_test[:100]
display(actual)
display(pred)
'''

#Precision-Recall Curve

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, algo1)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(y_test, algo1)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))




#CONFUSION MATRIX

import itertools
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(y_test, algo1)
np.set_printoptions(precision=2)
# Non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['True Positive','False Positive'],
                      title='Confusion matrix, without normalization')
# Normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['True Positive','False Positive'], normalize=True,
                      title='Normalized confusion matrix')
plt.show()




#DECISION TREE CLASSIFIER.. (used just as a reference)


from sklearn.tree import DecisionTreeClassifier
clf2=DecisionTreeClassifier()
clf2.fit(X_train,y_train)
clf2.score(X_train,y_train)
algo2=clf2.predict(X_test)
precision_score(algo2,y_test)
roc_curve(algo2,y_test)
r2_score(algo2,y_test)
plt.title('Receiver Operating Characteristic')
fpr,tpr, treshold=roc_curve(y_test,algo2)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print("Decision Tree Classifier:")
print("Decision Tree Classifier:")
print(accuracy_score(algo2, y_test))