# Twitter-Tweets-Sentimental-Analysis-ML-Project

#
##  <h>  **Analysis of Twitter's User's Sentiments regarding Twitter's Services with Machine Learning and Python (Pandas, NLTK, RE, Seaborn, Matplotlib, NumPy etc).**

**The Project is aimed to provide an analysis of the Sentiments of the Users of Twitter Social Media Platform using Machine Learning Models like Logistic Regression, Decision Tree and Random Forest based upon their Ratings and Reviews. The Project's objectives were to build a ML Model that can accurately predict the Sentiments (Positive/Negative) of the users on the basis of the Ratings and the content of the Reviews provided by it's users. The aim was to assist the Social media platforms with the information that they can use to formulate their strategy in order to enhance the User's satisfaction and expand their user base. Additionally, different ML Models have been compared on the basis of their Accuracy Score and other Key Metrics, to find out the best fit model for the task. Moreover, it aimed to reveal some of the dataset's hidden insights. The Data Cleaning was done with Pandas. The ML Models were built with NLTK, RE and Pandas, whereas the analysis of the data has been done with Pandas and visualisation library, Seaborn. This project was completed under the supervision of respected Mr. Tejas Natani, Instructional Associate at Masai School within 2 days.**




<br>
<br>
<p align="center"><a><img src="https://forthebadge.com/images/badges/built-with-love.svg"><img src="https://forthebadge.com/images/badges/made-with-python.svg"></a></p>

#  <img src="https://user-images.githubusercontent.com/106439762/181935629-b3c47bd3-77fb-4431-a11c-ff8ba0942b63.gif" width="48" height="48"> **User's Manual**

| Files| Description |
| ------------- | ------------- |
| **ML-ModelCreation_and_Analysis_EPML.ipynb** | This file contains the Python codes of the ML model creation and their Analysis parts. |
| **DataCleaning_and_Analysis_EPML.ipynb** | This file contains the Python codes of the Data Cleaning and Exploratory Analysis parts. |
| **Placement_Data.csv**  | This file provides the raw data for the project .  |
| **Employment_Prediction_ML_Project.csv**  | This file contains the final data drawn out after the Data Cleaning.  |
<br>


#  <img src=https://user-images.githubusercontent.com/106439762/178428775-03d67679-9aa4-4b08-91e9-6eb6ed8faf66.gif  width="48" height="48"> Analysis
   
    
    o       Analysed the relationship between scores at different level of education and the placement status using scatter plot.
    
    o	Analysed the relationship between all the numerical columns of the Data using Pairplot.
     
    o	Analysed the relationship between Percentage score in Graduation and the Placement Status using Boxplot.
  
    o	Analysed the correlation between the columns by using correlation heat map.
    
    o	Analysed the distribution of Salary amongst the people who are placed or are not placed.
    
    o	Analysed the suitability of different ML models for predicting placement status and salary of the people.

<br>

# <img src="https://user-images.githubusercontent.com/106439762/181937125-2a4b22a3-f8a9-4226-bbd3-df972f9dbbc4.gif" width="48" height="48" > Quick Start

    1. Started with cleaning the Data using Pandas, NumPy and Python's programming features 'for loop', replace, fillna, datatype conversion etc.
    
    2. Did Outlier analysis of the numerical columns and removed the Outliers.
 
    3. Did Exploratory analysis of the data, drawn out some important hidden insights using Pandas, Seaborn, Matplotlib etc and exported the cleaned data to a new CSV file.
    
    4. Imported the Cleaned data in another ipynb file and did Scaling of Numerical columns using Standard Scaler and Encoding of categorical ones using OneHot Encoder. 
    
    5. Built Linear Regression model, predicting the Salary and tested it by finidng R2 score and other accuracy metrics.
    
    6. Built Logistic Regression, Decision Tree and Random Forest models to predict Placement Status and did comparison analysis with the help of accuracy score, confusion matrix, Classification report etc.
   
<br>

# <img src="https://user-images.githubusercontent.com/108053296/185756908-fbb62168-d923-48f2-992f-b8e2fde848fe.gif" width="48" height="48" > Findings
   
   1. The Data shows high Correlation between High School Percentage and Placement Status. 
   
   2. The Data shows that the People having Lower Percentage in High School are mostly not placed and others are mostly placed.
   
   3. There is also a high positive correlation between scores in Graduation and Placement Status.
   
   4. The Data shows that around 25% of the people are not placed, the reason is mostly the lower scores at different levels of education.
   
   5. The Count of People decreases with Increasing salary, showing less placements at levels with higher salary.
   
   6. Linear Regression Model is suitable for Predicting Salary as the R2 Score is found around 96%, when tested, which is suitable.
   
   7. For Predicting the Employement Status, Logistic Regression Model is better as it shows the highest accuracy, around 84%. Total Positive Rate is also higher (22), compared to Decision Tree and Random Forest models (having 17).
   
   <br>
   
   #  <img src=https://user-images.githubusercontent.com/106439762/178803205-47a08ce7-2187-4f96-b301-a2b68690619a.gif width="48" height="48" > Metrics
![pyhton-pandas](https://user-images.githubusercontent.com/106439762/177094844-d74edfa1-823d-4f17-8d94-3600e058cf1e.svg)
   
   1. The R2 Score of the Linear Regression Model built for predicting Salary is found 96%. 
   
   2. The Accuracy Score of the Logistic Regression Model built for predicting Placement Status is found around 84% and the True Positive Rate (TPR) is found 22.
   
   3. The Accuracy Score of the Decision Tree Model built for predicting Placement Status is found around 72% and the True Positive Rate (TPR) is found 17.
   
   4. The Accuracy Score of the Random Forest Model built for predicting Placement Status is found around 80% and the True Positive Rate (TPR) is found 17.
   
   <br>
   
   
   #  <img src=https://user-images.githubusercontent.com/106439762/178803205-47a08ce7-2187-4f96-b301-a2b68690619a.gif width="48" height="48" > Future Scope
   
   <B> The aim of the Project is to: </B>
   
   1. Assist the students in making themselves capable of getting placed and to get higher salaries.
   
   2. Help the organisations like Colleges and Educational Institutes in predicting the number of Students who can get placements, studying there.
   
   3. To provide an idea of the employement situation in the Economy.
   
   4. Helping out the freshers and the people lacking in information, in finding the easiest path to secure placements.
   
   5. Helping the New Generation to plan their future with the informations like the required scores at different levels of eduction, most in demand subjects etc.
   
   6. Empowerment of common people with the relevant information that can help them to secure Placements and aid to raising the Employment rate of the Economy.
   
   
   
    

![image](https://user-images.githubusercontent.com/108053296/189940016-b2f9ffd2-ff3c-46a7-90a0-ac2929953469.png)
