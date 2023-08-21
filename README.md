# Customer_Churn_Prediction_ML
# A Customer Churn Analysis for Zanzibar Telecommunication

![Project Image](project_image.png) <!-- Replace with an image relevant to your project -->

Welcome to the Customer Churn Analysis project for Zanzibar Telecommunication! In this project, we dive into the world of data to help the telecommunication company enhance their profit and revenue margins by refining their customer retention strategies. We'll employ machine learning classification models to perform churn analysis on their customer datasets.

## Project Overview
- Understand Classification Models and their applications.
- Assist the telecommunication company in comprehending their customer data.
- Calculate the lifetime value of each customer.
- Identify factors influencing customer churn rates.
- Predict if a customer is likely to churn.

## Skills Developed
1. Data Exploration and Analysis.
2. Handling Missing Data.
3. Feature Engineering.
4. Implementing Machine Learning Algorithms: Logistic Regression, Decision Trees, Support Vector Machine, Random Forest, etc.
5. Model Evaluation and Interpretation using LIME and SHAP techniques.
6. Model Optimization and Hyperparameter Tuning.

## Key Libraries Used
- Pandas, NumPy for data manipulation and cleaning.
- Matplotlib, Seaborn, Plotly for data visualization.
- Statsmodels, SciPy for statistical analysis.
- Scikit-learn for feature engineering and machine learning.
- Imbalanced-learn (SMOTE) for class imbalance.
- SHAP for interpreting model predictions.
- Pyodbc, dotenv for database access.

## Project Steps
1. **Load Datasets (Collect Initial Data):**
   - Import necessary libraries.
   - Retrieve login credentials from .env and connect to the database.
   - Load data into pandas DataFrame for manipulation and cleaning.

2. **Exploratory Data Analysis (EDA):**
   - Check the first 5, last 5, and 10 random rows of the datasets.
   - Examine the shape and information of the datasets.
   - Check for duplicates and missing values.
   - Perform descriptive analysis.

3. **Data Quality Assessment:**
   - Identify key data quality issues.


### Machine Learning Models Explored

- KNeighborsClassifier
- LinearRegression
- LogisticRegression
- AdaBoostClassifier
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- Support Vector Machine (SVC)
- GaussianNB

### Data Preprocessing & Feature Engineering

- Load and examine the dataset.
- Handle missing values through imputation and removing rows.
- Create or select relevant features that could impact customer churn.
- Scale or normalize numerical features.
- Encode categorical variables using techniques like one-hot encoding.
- Split the dataset into features (X) and target variable (y).

### Machine Learning Models Explored
We consider eight (8) models for churn prediction:
1. AdaBoostClassifier
2. Logistic Regression
3. Gaussian Naive Bayes
4. Random Forest Classifier
5. KNeighbors Classifier
6. Decision Tree Classifier
7. Gradient Boosting Classifier
8. Support Vector Classifier (SVC)

### Model Evaluation
We evaluate each model's performance using cross-validation to ensure reliable metrics. The key evaluation metrics used are accuracy for balanced datasets and F1-score for the unbalanced dataset.

### Confusion Matrix
Visualizing the confusion matrix for each model to gain insights into the performance and areas for improvement.

### Hyperparameter Tuning and Cross-validation using k-fold
Fine-tuning the models' hyperparameters to achieve optimal performance. We utilize k-fold cross-validation to validate the model's generalizability.

### Future Predictions
Based on the results obtained from our evaluation, we predict future customer churn using the top-performing models. This insight can guide the business in taking proactive measures to retain valuable customers.

### Let's Collaborate!
Your contributions are most welcome! Whether you're a data wizard or a curious soul, join our enchanting journey by contributing insights, optimizations, or enhancements.


## License
This project is licensed under the [GNU.2.0 License](LICENSE).

🔗 [Link to Article] [(https://www.linkedin.com/pulse/unveiling-power-collaboration-predicting-customer-churn-kofi-bamfo]

🔗 [Link to PowerBI Dashboard] [https://app.powerbi.com/groups/me/reports/655df298-36e9-4537-953c-37e410844b55/ReportSection?experience=power-bi] 

## Contributors: 
| Edinam Doe Abla | https://github.com/doeabla |

| Enoch Taylor-Nketiah |https://github.com/akbamfo |

| Timothy Morenikeji Akinremi | https://github.com/timothyakinremi
