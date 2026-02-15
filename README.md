# ML_Assignment
Implement multiple classifi cation models - Build an interactive Streamlit web application to demonstrate your models - Deploy the app on Streamlit Community Cloud (FREE)


Implement the following classification models using the dataset chosen above. All the 6 ML models are implemented on the same dataset.
        1. Logistic Regression
        2. Decision Tree Classifi er
        3. K-Nearest Neighbor Classifi er
        4. Naive Bayes Classifi er - Gaussian or Multinomial
        5. Ensemble Model - Random Forest
        6. Ensemble Model - XGBoost
For each of the models above, calculate the following evaluation metrics:
        1. Accuracy
        2. AUC Score
        3. Precision
        4. Recall
        5. F1 Score
        6. Matthews Correlation Coeffi cient (MCC Score)

Dataset description:

The dataset is downloaded from Kaggle repository. The dataset contains 1548 total entries. 
The dataset is for the various type of people having information on their age, credit history, employment status, annual income, children, mobile phone, type of occupation, family memmbers etc. 
The dataset has 18 features. The models have to predict if the person can be given credit card of not. Binary classification problem

Model Comparison:

Model Name                Accuracy        AUC      Precision      Recall        F1          MCC

Logistics Regression      0.8903        0.6528          1          0.286      0.556        0.1595

Decision Tree             0.8806        0.7071        0.4706      0.4571      0.4638       0.3967

KNN                       0.8871        0.7310        0.5000      0.1143      0.1860       0.1991

Naive Bayes               0.1355        0.6112        0.1155      1.0000      0.2071       0.0542

Random Forest            0.9323         0.8664        0.8889      0.4571      0.6038       0.6088
(Ensemble)    

XGBoost                  0.9129        0.8160         0.6667      0.4571      0.5424       0.5069
(Ensemble)



Observations about model performance



Model Name	                          Key Points

Logistic Regression		    High accuracy, but recall is low → model predicts positives very conservatively.

Decision Tree           	Balanced precision/recall, moderate MCC → decent generalization but prone to     overfitting.

KNN	                       Accuracy looks fine, but recall is very poor → misses many positives.

Naive Bayes              	Recall is perfect but precision extremely low → predicts almost everything as positive.

Random Forest	          	Best accuracy and MCC → strong ensemble performance, balanced but recall could improve.

XGBoost	                	High accuracy and AUC, solid balance between metrics → strong ensemble


Random Forest is the top performer overall (highest accuracy and MCC).

XGBoost is close behind, with strong AUC and balanced metrics.

Logistic Regression achieves high accuracy but sacrifices recall.

Decision Tree is balanced but less robust than ensembles.

KNN has decent accuracy but fails on recall → not reliable for imbalanced datasets.

Naive Bayes shows extreme behavior: perfect recall but terrible precision, leading to very low accuracy.
