# Adult_Income_Prediction
The Project aims to predict the income level of adult based on various factors using machine learning techniques. The objective is to build the model that can accurately predict whether the adult earn above or below certain income threshold. Importance of adult income prediction: predicting income is crucial for understanding socioeconomic patterns and making informed decision in areas such as policy making, resource allocation, and financial planning. By analysing factors such as age, occupation and work hour,we can get insights into the factor that contribute to higher income levels. This prediction task help individuals, organizations and governments make better inform decision and allocation resources effectively.
Dataset and machine learning techniques: the datasets used for adult income prediction contains information about individuals, including their age, education, occupation, marital status, and more. Machine learning techniques, such as decision tree, random forest, or logistic regression, can be utilize to use model that learn pattern from this data and predict income level based on given features. theses techniques involve the training the model on subset of the dataset, evaluating its performance, and fine tunning the model to improve accuracy. By applying these techniques, we can develop a predictive model that can assist in identifying the factors influencing adult income levels.
Data Description:
The adult income dataset is the collection of information about adults, designed for predicting their income levels. It consists of various features or attributes that provide insights into an individual background and circumstances. The dataset structure typically includes rows representing different individuals and column representing different characteristics. 
Target variable and attributes: the target variable in adult income dataset is “income”, which indicate whether an individual earn above or below a specific income threshold. This variable is crucial as it serves as the prediction goal. The dataset contains several attributes that can used to predict income level, such as age, work class, education, marital status, occupation relationship status and more
Significance of the dataset: 
The adult income dataset holds significant importance for income prediction task. By analysing the available attribute we can gain insights into the factors that contribute to higher or lower income levels. for instance, education level, occupation and work hour can be strong indicator of earing potential. This dataset allows us to explore patterns and relationships between various factor and income, enabling us to develop predictive models.
Data Exploration and Visualization:
To explore the adult income dataset and gain insights into the relationship between different to feature and income levels, we can perform data exploration and visualization.
1.	Load the dataset: start by loading the adult income dataset into your programming environment. this dataset contains information about adults and their income levels.
2.	Analysis feature distribution: look at each feature in the dataset into your programming environment this dataset contains information about adults and their income levels.

3.	Visualize feature income relationship: 

In the data exploration and visualization, when I plotted histogram using the “age” variable to analyse the relation between age and income, I found that age group between30 to 40 had high frequency of individuals. this means that there where many adults within this age range in dataset. By looking at the histogram I suggest that individual in this age group might be more prevalent in dataset compare to other age group.





 


After plotting the box plot to visualize the relationship between ‘income’ and ‘hours per week’ the result can be observed from that plot. The box plot shows the distribution of hours worked per week for different income levels.
 
Data pre-processing:
Data pre-processing is a crucial step in preparing the dataset for analysis and modelling. It involve missing handling values, and other data quality issues to ensure reliable and accurate results. Here some important steps in data pre-processing:
•	Handling Missing Values:  handling missing values can occur when certain data is not available for some instance. The first step is to check the missing values in the data set using df.isnull().sum(). This counts the number of missing values in each column. Afterward, the df.dropna() method is to remove any rows containing missing values. This ensure that dataset dose not have any missing values, as missing data can cause issue during analysis and modelling.
•	Splitting the Dataset: Next, the dataset split into two parts- the feature (X) and the target variable (y). the df.drop(‘income, axis=1) operation removes the ‘income’ columns from the dataset storing the remaining columns in X. the ‘income, columns assigned to (y). this step seperates the data into input features and the corresponding target variable that we want to predict.
•	Splitting into training and testing sets: the data is further divided into training and testing sets using train_test_split(). This function splits the data into random subsets, with 80% allocated for training and 20% for testing this division allow us to train our models on a portion of the data and evaluate their performance on unseen data.
•	Feature Scaling: The next step is feature scaling, which ensure that all features are on similar scale and have similar range. Here, the SnandardScaler class from sklearn.preprocessing is used to standardize the numerical features. The fit.transform() method scale the feature in X_train and X_test, separately. This step is crucial because some machine learning algorithms perform better when the features are on similar scale. 
Model Selection and Training:
In the adult income prediction task, you have experimented with several machine learning algorithms and evaluate their performance. Let”s discuss the process and results:
•	Selection of Machine Learning algorithms: I have selected three algorithms for this task: Logistic Regression, Decision Tree, and Support Vector Machine (SVM). Each algorithm has its own characteristics and suitability for different types of problems.
•	Splitting the Dataset: the dataset was split into training and testing. the training was used to train the models, while the testing set was used to evaluate their performance.

Training and Evaluating the Models:
1.	Logistic Regression:
 
•	The Logistic Regression model was chosen for its suitability in binary classification task like predicting income levels
•	the Logistic Regression was trained using model.fit() on the training set after scaling the feature.
•	Prediction were made on the testing data using model.predict().
•	The accuracy of logistic regression was found 0.8142.
•	The classification report provides additional evaluation metrics such as precision, recall, and F1-score for each class. It shows that model  has higher precision and recall for the ‘<=50K’ class compared to the ‘>50K’ class.

2.	Decision Tree:

•	The Decision Tree Classifier was selected as another algorithm for income prediction.
•	Similarly, the dataset was split, the model was trained, and prediction were made.
•	The accuracy of Decision Tree model was found 0.8142.
•	The classification report shows that the model achieved higher precision, recall and F1-score for the ‘<=50K’ class, while the ‘>50K’ class had lower values.


3.	Support Vector Machine (SVM):

•	The SVM classifier was chosen as third algorithm for income prediction.
•	Again, the dataset was split the model was trained and prediction were made.
•	The accuracy score for the SVM model was also 0.8142.
•	The classification report revels that the model had higher precision and recall for the ‘<=50k’ class compared to the ‘>50K’ class. 

4.	Model Architectures and Hyperparameters:

•	After performing hyperparameter tuning using grid search for both the decision tree and logistic regression model, the performance din not improve significantly. The classification report for both models show similar results to the previous ones, with accuracy score of 0.81 for both models.



