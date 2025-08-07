# Ai-project
This project analyzes employee emails to classify sentiment as positive, negative, or neutral. It performs monthly sentiment scoring, ranks employees by sentiment, identifies flight risk based on sentiment trends, and models overall sentiment changes over time using linear regression—helping improve employee engagement insights.
Employee Sentiment Analysis Project

This project analyzes employee email data to perform sentiment labeling, exploratory data analysis (EDA), monthly sentiment scoring, employee ranking, flight risk identification, and sentiment trend modeling using linear regression.

 Dataset Description

The dataset contains 2,191 employee emails with the following columns:

Subject: Email subject line.
body: Email body text.
date: Timestamp of the email.
from: Sender’s email address.

The data was sourced from provided SharePoint Excel files.


 Environment Setup

This project was developed with Python 3 and uses the following libraries:

 pandas  
 numpy  
 matplotlib  
 seaborn  
 scikit-learn  
 textblob  
 nltk  
 transformers (optional)

You can install the dependencies via:
pip install pandas numpy matplotlib seaborn scikit-learn textblob nltk transformers


 Usage Instructions

1. Upload the dataset file to your working environment (e.g., Google Colab).  
2. Run the notebook or script to perform Sentiment Labeling on combined email subject and body text.  
3. Execute exploratory data analysis cells/scripts and generate visualizations to understand sentiment distributions and trends.  
4. Calculate monthly sentiment scoring and ranking of employees based on average sentiment polarity.  
5. Execute the flight risk identification script, which flags employees with low or declining sentiment trends.  
6. Run the linear regression model to analyze overall sentiment trend over time.  
7. Review generated plots, tables, and summaries for insights.



 Methodology

 Sentiment Labeling
 Combined the email Subject and body text for comprehensive analysis.  
 Used TextBlob for polarity scoring of text.  
 Classified messages with polarity > 0 as Positive, polarity < 0 as Negative, and polarity = 0 as Neutral.

 Exploratory Data Analysis
 Generated plots for sentiment label distributions and trends over months.  
 Analyzed monthly counts of each sentiment label and average sentiment polarity.

 Monthly Sentiment Scoring
 Aggregated polarity scores grouped by month to assess temporal sentiment trends.

 Employee Ranking
 Calculated average sentiment polarity per employee.  
 Ranked employees based on their average polarity score, signaling more positive or negative overall sentiment.

 Flight Risk Identification
 Identified employees with average polarity below a threshold (e.g., 0.1).  
 Detected employees exhibiting declining sentiment trends using linear regression slope analysis on their monthly polarity scores.  
 Combined these factors to flag potential flight risk employees.

 Sentiment Trend Modeling
 Built a linear regression model to evaluate overall sentiment polarity changes over time across the company’s email data.

 Results Summary

 Majority of employee emails were classified as neutral or positive sentiment.  
 Monthly sentiment trends fluctuate periodically, indicating varying sentiment over time.  
 Employee ranking highlighted highly positive employees and those potentially at risk.  
 Flight risk identification successfully flagged employees with consistently low or deteriorating sentiment scores for attention.


 Contact and Support

For any questions or clarifications about this project, please contact:  
souradeep datta - souradeepdatta222@gmail.com.com

Thank you for reviewing this project!

