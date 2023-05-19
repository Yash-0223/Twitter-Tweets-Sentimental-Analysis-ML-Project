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
| **Twitter-Tweets-Sentimental-Analysis.ipynb** | This file contains the Python codes of the Suitable structuring of the Data, ML model creation and their Analysis parts. |
| **TTSAP - Data - Cleaning.ipynb** | This file contains the Python codes of the Data Cleaning and Exploratory Analysis parts. |
| **twitter_tweets.csv**  | This file provides the raw data for the project .  |
| **TTSAP_Final_Data.csv**  | This file contains the final data drawn out after the Data Cleaning.  |
<br>


#  <img src=https://user-images.githubusercontent.com/106439762/178428775-03d67679-9aa4-4b08-91e9-6eb6ed8faf66.gif  width="48" height="48"> Analysis
   
    
    o       Analysed the Accuracy Scores of Different ML Models for predicting the Sentiments.
    
    o	Also analysed the F1 Scores, Precision Scores and Recall Scores of Different ML Models by creating their Classification Reports.
     
    o	Performed comparison analysis to find out the Best Fit Machine Learning Model suitable for this task.
  
    o	Analysed the relationship between the sentiments and ratings of the users.

<br>

# <img src="https://user-images.githubusercontent.com/106439762/181937125-2a4b22a3-f8a9-4226-bbd3-df972f9dbbc4.gif" width="48" height="48" > Quick Start

    1. Started with cleaning the Data using Pandas by Dropping the Duplicates and Null values.
    
    2. Created a new Data Frame with only relevant columns out of the initial Data Frame.
 
    3. Also Replaced values in Ratings columns to Numbers based on their Intensity and exported the Cleaned data as CSV to use it further for Sentimental Analysis.
    
    4. Imported the Cleaned data in another ipynb file and did Scaling of Numerical columns using Standard Scaler and Encoding of categorical ones using OneHot Encoder. 
    
    5. Performed Sentiment Intensity Analysis and based on it's score, categorized it as Positive or Negative and made a column of it in the Data Frame and performed analysis on it.
    
    6. Removing special charcters and stop words present in the text and performed Vectorization (feature extraction) on it.
   
    7. Built different ML models like Logistic Regression, Decision Tree and Random Forest and compared them on the basis of their Accuracy Score and other Key Metrics, to find out the best fit model for the task.
   
<br>

# <img src="https://user-images.githubusercontent.com/108053296/185756908-fbb62168-d923-48f2-992f-b8e2fde848fe.gif" width="48" height="48" > Findings
   
   1. Around 38% of the users have Negative Sentiments, whereas rest 62% have Positive Sentiments for Twitter's Services. This shows healthy User Satisfaction.
   
   2. Users providing Ratings as 1 are having highest Negative Sentiment bearing User Population.
   
   3. Whereas, those Users who provided Ratings as 3 are having highest Positive Sentiment bearing User Population.
   
   4. The Random Forest ML Model with about 90% Accuracy Score is found the Best Fit Model for this task.
   
   <br>
   
   #  <img src=https://user-images.githubusercontent.com/106439762/178803205-47a08ce7-2187-4f96-b301-a2b68690619a.gif width="48" height="48" > Metrics
![pyhton-pandas](https://user-images.githubusercontent.com/106439762/177094844-d74edfa1-823d-4f17-8d94-3600e058cf1e.svg)
   
   1. The Accuracy Score of the Logistic Regression Model built for predicting Sentiments of the Users is found around 76% and the F1 Score is found 75%.
   
   2. The Accuracy Score of the Decision Tree Model built for predicting Sentiments of the Users is found around 73% and also the F1 Score is found 73%.
   
   3. The Accuracy Score of the Random Forest Model built for predicting Sentiments of the Users is found around 90% and also the F1 Score is found 90%.
   
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
