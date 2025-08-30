 📊 Employee Sentiment Analysis Project

This project analyzes employee email data to perform sentiment labeling, exploratory data analysis (EDA), monthly sentiment scoring, employee ranking, flight risk identification, and sentiment trend modeling using linear regression.



  Dataset Description

The dataset contains 2,191 employee emails with the following columns:

Subject: Email subject line
body: Email body text
date: Timestamp of the email
from: Sender’s email address

The data was sourced from provided SharePoint Excel files.



 Environment Setup

Developed with Python 3.10+, using the following libraries:

pandas
numpy
matplotlib
seaborn
scikit-learn
textblob
nltk

Install dependencies:

bash
pip install pandas numpy matplotlib seaborn scikit-learn textblob nltk transformers




 Usage Instructions

1. Upload the dataset file (`test.xlsx`) to your environment (e.g., local machine or Google Colab).
2. Run the notebook or script to:

   * Perform sentiment labeling (Positive, Neutral, Negative).
   * Execute EDA and generate sentiment distribution & trend plots.
   * Calculate monthly sentiment scores (+1, 0, -1 per message).
   * Rank employees (Top 3 Positive & Top 3 Negative per month).
   * Detect flight risk employees (≥4 Negative emails in a rolling 30-day window).
   * Fit a linear regression model on monthly sentiment trends.
3. Review the generated plots, tables, and printed summaries for insights.



Methodology

Sentiment Labeling

Combined Subject + Body for text analysis.
Used TextBlob for polarity scoring.
Classification rules:

   Polarity > +0.05 → Positive
   Polarity < -0.05 → Negative
   Otherwise → Neutral

 Exploratory Data Analysis (EDA)

 Distribution of polarity values.
 Count of sentiment labels.
 Monthly trends of sentiment counts & average polarity.

 Monthly Sentiment Scoring

 Each email assigned a score: Positive = +1, Negative = -1, Neutral = 0.
 Aggregated scores per employee per month.

 Employee Ranking

 For each month:

   Top  Positive employees → highest scores.
   Top  Negative employees → lowest scores.

 Flight Risk Identification

 An employee is flagged if they send ≥4 Negative emails in any rolling 30-day window**.
 This strictly follows the problem statement requirement.

 Sentiment Trend Modeling

 Calculated monthly average polarity.
 Fitted a linear regression model:

   Outputs slope, intercept, and R² score.
   Interprets overall sentiment trend over time.



 Results Summary

 Sentiment Distribution → Majority emails are Positive or Neutral.
 Monthly Trends → Sentiment fluctuates; dips in mid-2011 reflect stress periods.
 Employee Ranking → Top 3 positive & negative employees identified for each month.
 Flight Risk → 9 employees flagged under the ≥4 negative emails in 30 days rule.
 Regression Analysis → Slight upward trend detected, but weak fit (R² = 0.27) → sentiment is noisy.


 Contact & Support

For any questions or clarifications, please contact:

Souradeep Datta
 [souradeepdatta222@gmail.com](mailto:souradeepdatta222@gmail.com)

