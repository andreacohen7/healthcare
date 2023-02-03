# Stroke Predictions

## 11 clinical features for predicting stroke events

**Andrea Cohen**

### Task:
- To predict whether a patient is likely to have a stroke based on the input parameters like gender, age, various diseases, and smoking status.

### Data:
Data Source https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download

For this data set there were 5110 rows and 11 columns.

### Data Dictionary:
Variable Name	| Description
---| ---
id |	unique identifier
gender |	"Male", "Female" or "Other"
age |	age of the patient
hypertension |	0 if the patient doesn't have hypertension, 1 if the patient has hypertension
heart_disease |	0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
ever_married |	"No" or "Yes"
work_type |	"children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
Residence_type |	"Rural" or "Urban"
avg_glucose_level |	average glucose level in blood
bmi |	body mass index
smoking_status |	"formerly smoked", "never smoked", "smokes" or "Unknown"*
stroke |	1 if the patient had a stroke or 0 if not

 *Note: "Unknown" in smoking_status means that the information is unavailable for this patient

### Methods:

The data was first inspected.
- For exploratory and explanatory analysis: 
  - missing data was noted but not imputed 
  - no duplicates were noted
  - no inconsistencies in categorical data were noted
  - the imbalance of the target class was noted

The data was then prepared for machine learning by cleaning it.
- For machine learning with the Random Forest Classifier: 
  - a preprocessor was used for mean imputation for float columns and one-hot encoding of categorical columns
  - a model pipeline was used for dealing with the target class imbalance through SMOTE and for the model
  - GridSearchCV was used to tune the model's hyperparameters
- For machine learning with the Logistic Regression Model:
  - a preprocessor was used for mean imputation for float columns and one-hot encoding of categorical columns
  - a model pipeline was created for scaling the numerical data, dealing with the target class imbalance through SMOTE, and for the model
  - GridSearchCV was used to tune the model's hyperparameters
- For machine learning with LightGBM:
  - a preprocessor was used for mean imputation for float columns and one-hot encoding of categorical columns
  - a model pipeline was created for scaling the numerical data, dealing with the target class imbalance through SMOTE, and for the model
  - GridSearchCV was used to tune the model's hyperparameters
- For machine learning with Principal Component Analysis:
  - a preprocessor was used for mean imputation for float columns and one-hot encoding of categorical columns
  - a model pipeline was created for scaling the numerical data, dealing with the target class imbalance through SMOTE, applying PCA while retaining 95% of the variance, and for the model
  - GridSearchCV was used to tune the model's hyperparameters
- For machine learning with Feature Engineering:
  - columns without predictive information were deleted
    - Multivariate visualizations showed that there was no relationship between the gender of the patient and strokes. They also showed that there is no difference in urban vs rural residences and strokes. 
  - columns with limited predictive information were combined
    - Multivariate visualizations showed that both of these features increased strokes. However, there are other reasons, besides the risk of stroke, that affect patients' choices for marriage and work. 
  - columns with numerical medical data were binned into categories 
    - because some of the medical characteristics are already categorical.)
  - a preprocessor was used for mean imputation for float columns and one-hot encoding of categorical columns
  - a model pipeline was created for scaling the numerical data, dealing with the target class imbalance through SMOTE, and for the model
  - GridSearchCV was used to tune the model's hyperparameters


### Univariate Visualizations:

#### Stroke Distribution

![Stroke Distribution.png](https://github.com/andreacohen7/healthcare/blob/main/Stroke%20Distribution.png)

  - The target classes are severely imbalanced (95.13% of the samples did not have a stroke and 4.87% of the samples had a stroke).

#### Graphs to view medical characteristics

![Medical Characteristics.png](https://github.com/andreacohen7/healthcare/blob/main/Medical%20Characteristics.png)

  - The distribution of hypertension is significantly unbalanced--the majority of patients did not have hypertension.
  - The distribution of heart disease is significantly unbalanced--the majority of patients did not have heart disease.
  - The distribution of glucose levels is right-skewed, with the majority of patients having lower average glucose levels.
  - The distribution of BMI is right skewed, with the majority of patients having lower BMI's.

### Multivariate Exploratory Visualizations:

#### Do behavioral characteristics increase strokes?

![The Effect of Behavioral Characteristics on Strokes.png](https://github.com/andreacohen7/healthcare/blob/main/The%20Effect%20of%20Behavioral%20Characteristics%20on%20Strokes.png)

- These barplots show that most behavioral characteristics increase strokes.
  - Mariage (even previous marriages) increases strokes.
  - Work type (especially self-employed, government employees, and private employees) increases strokes.
  - Smoking status (especially former smokers) increases strokes.
  - There is no difference, though, in urban vs rural residences and strokes.

#### Do medical characteristics increase strokes?

![The Effect of Hypertension and Heart Disease on Strokes.png](https://github.com/andreacohen7/healthcare/blob/main/The%20Effect%20of%20Hypertension%20and%20Heart%20Disease%20on%20Strokes.png)

![The Effect of Average Glucose Level and BMI on Strokes.png](https://github.com/andreacohen7/healthcare/blob/main/The%20Effect%20of%20Average%20Glucose%20Level%20and%20BMI%20on%20Strokes.png)

- These plots show that medical characteristics increase strokes.
  - The barplots show that hypertension and heart disease both increase strokes.
  - The scatterplot shows that the average glucose level increases strokes, but that BMI does not.

### Machine Learning:

#### Models Evaluated and Results:

- **Random Forest Classifier with SMOTE:**
- using decision trees to predict strokes
  - Test Scores
    - accuracy for making correct predictions on the test set:  95% 
    - precision for predicting the stroke class:  33% 
    - recall for predicting the stroke class:  5% 
    - type 2 (false negative) errors:  95%
  - Using SMOTE and tuning the hyperparameters helped the Random Forest Classifier predict the stroke class.

- **Logistic Regression Model with SMOTE:**
- using logistic regression to predict strokes
  - Test Scores
    - accuracy for making correct predictions on the test set:  95% 
    - precision for predicting the stroke class:  0% 
    - recall for predicting the stroke class:  0% 
    - type 2 (false negative) errors:  100%
  - Using SMOTE did not help the Logistic Regression Model predict the stroke class.

- **LightGBM Model with SMOTE:**
- using gradient boosting to predict strokes
  - Test Scores
    - accuracy for making correct predictions on the test set:  93% 
    - precision for predicting the stroke class:  11% 
    - recall for predicting the stroke class:  6% 
    - type 2 (false negative) errors:  94%
  - Using SMOTE helped the LightGBM predict the stroke class.
  - Because of the class imbalance, the final model must predict well for the stroke class--the tuned LightGBM with SMOTE was best so far at predicting the stroke class.
  - Because of the high cost of false negatives (predicting no stroke when, in fact, there will be a stroke), these errors must be reduced--the tuned LightGBM with SMOTE had the lowest false negative rate so far.

- **LightGBM with SMOTE and Principal Component Analysis:**
- using gradient boosting with dimensionality reduction to predict strokes
  - Test Scores
    - accuracy for making correct predictions on the test set:  93% 
    - precision for predicting the stroke class:  11% 
    - recall for predicting the stroke class:  6% 
    - type 2 (false negative) errors:  94%
  - Using PCA did not help the LightGBM with SMOTE predict the stroke class.
  
- **LightGBM with SMOTE and Feature Engineering:**
- using gradient boosting with feature engineering to predict strokes
  - Test Scores
    - accuracy for making correct predictions on the test set:  93% 
    - precision for predicting the stroke class:  19% 
    - recall for predicting the stroke class:  11% 
    - type 2 (false negative) errors:  89%
  - Using feature engineering helped the LightGBM with SMOTE predict the stroke class.
  - Because of the class imbalance, the final model must predict well for the stroke class--feature engineering helped LightGBM with SMOTE predict the stroke class. The improvement was 8% for precision and 5% for recall.
  - Because of the high cost of false negatives (predicting no stroke when, in fact, there will be a stroke), these errors must be reduced--feature engineering did help LightGBM with SMOTE reduce the false negative rate by 5%. It also helped LightGBM with SMOTE reduce the false positive rate by 0.2%.

The LightGBM with SMOTE and feature engineering had the highest precision and recall for predicting the stroke class. The LightGBM with SMOTE and feature engineering also had the lowest false negative rate.
**The LightGBM with SMOTE and feature engineering was the best model, based on the classification metrics for the testing data.**
A tuned LightGBM with SMOTE made the best predictions for the stroke class while limiting the false negative rate. Feature engineering improved those predictions even further.  However, the tuned LightGBM with SMOTE is not ready for real-world data with real-world consequences.

### Recommendations:

#### Understanding the patient, behavioral, and medical characteristics of patients that increase strokes:

The medical stakeholder should know that older patients have more strokes, regardless of their gender; that mariage (even previous marriages), work type (especially self-employed, government employees, and private employees), and smoking status (especially former smokers) increases strokes; and that hypertension, heart disease, and average glucose level increase strokes.  The medical stakeholder should focus on those features (smoking status, hypertension, heart disease, and average glucose level) which can be changed for predicting and preventing strokes.

#### Predicting future strokes based on the data provided:
Overall, the LightGBM with SMOTE was the best model, compared to the Random Forest Classifier and the Logistic Regression model.  Feature engineering further improved the predictions.  However, the best model was not even very good at predicting the stroke class--it only achieved 19% precision and 11% recall for the stroke class. Plus, it had an 89% rate of false negatives--that means that 89% of people who will have a stroke were actually predicted to not have a stroke!

### Limitations & Next Steps
The information collected provided some insight into the patient characteristics that increase strokes, but did not predict strokes well.  The medical stakeholder should collect different data--there are clearly other medical features that were not captured in this data set that can predict future strokes better.

### For further information

For any additional questions, please contact *andrearcohen7@gmail.com*.
