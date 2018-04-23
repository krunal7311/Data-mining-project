The goal of the data mining is to predict if a given state, age and gender is prone to suicide by a specific cause and the educational status. The data mining will also allow us to predict if the victim is prone to suicide by a specific cause or the educational status individually.
Two classification algorithms were selected and analysed for the accuracy score. These are Decision Tree Classifier and Random Forest Classifier. Decision Tree Classifier gave slightly higher accuracy as compared to Decision Tree classifier.
However, the Random Forest Classifier is a better prediction model in terms of generalisability and robustness1 and hence it was selected.
Model selected : Random Forest Classifier
The parameters selected for the algorithm are as below:
n_estimators:
Set to 1. The number of trees in the forest. Higher the value, better is the prediction. However, there is a chance of overfitting the data and also makes the algorithm slower hence, it was set to 1.
criterion:
Set to “gini” for gini impurity. The accuracy of the algorithm is not affected by setting it to Gini or Entropy. However, calculating gini takes less computation time and hence the algorithm performance is improved slightly when criterion is set to gini.
max_depth
Set to default=None. All the nodes will be expand until all leaves contain less than min_samples_split samples.
min_samples_split
Set to 50. Minimum number of samples required to split a node. Smaller the number of samples, higher is the chance that the tree will capture the noise in the training data. However, setting it very high will result in reduced accuracy of prediction. An optimal value was set as 50. This is because the dataset used here was balanced using SMOTE technique from the previous iteration which had generated extra samples. Setting the value to 50 reduced the accuracy of the model from 90% to 85%. However, the model is now less prone to noise and is more generalised thus making it a better model. max_features
Set to default. Increasing the value will have higher accuracy but slower performance. The value selected should be such that there is a balance between speed and accuracy. For my project, since the dataset is relatively small as compared to real world problems, changing this value would have no noticeable impact on the performance and hence it was set to default. This means that all the features are selected that makes sense in every tree. random_state
Set to 5. It can be set to any integer value. This is a random seed which helps to generate same solution when the input data and parameters remain the same.

  Only the most important ones and the modified parameters are described above. Rest of the features given below were unchanged.
min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease, min_impurity_split, class_weight, pre_sort.
7. Data Mining
Test design
The quality metrics defined by class and function reference of scikit-learn were used to measure the model’s quality of predictions. 2
The quality measures used to the models are Precision score, R2 score, Accuracy score, ROC-Curve, Precision-Recall curve and Confusion Matrix.
Model execution
The model was executed in Python 3.6. The Anaconda Cloud package management was used for the entire project. The IDE used was Scientific Python Development Environment , (SPYDER). The entire model is implemented in a single python file. However, the whole code is divided into following segments as follows:
• Loading the Dataset
• Encoding the labels to numeric values
• Selecting the Feature and Target Columns
• Splitting data into Training and Test sets
• Implementing classification algorithm (Random Forest Classifier)
• Quality metric scores
• Interpretations
Dataset Link: https://drive.google.com/file/d/0B0DLgKrv2WN_QVVmWF9FWDJ1LW8/view? usp=sharing
