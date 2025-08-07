# 1. Import necessary libraries
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 2. Load dataset
df = pd.read_excel('data/test.xlsx')

#  Preview data
print(df.info())
print(df.describe())
print(df.head())

# 3. Combine Subject and body into one text column for sentiment analysis
df['text'] = df['Subject'].fillna('').astype(str) + ' ' + df['body'].fillna('').astype(str)

# 4. Define sentiment classification function
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment labeling
df['Sentiment'] = df['text'].apply(get_sentiment)

# 5. Extract month for temporal analysis
df['month'] = df['date'].dt.to_period('M')

# Monthly sentiment counts
monthly_sentiment = df.groupby(['month', 'Sentiment']).size().unstack(fill_value=0)
print(monthly_sentiment.head())

# Plot monthly sentiment trends
monthly_sentiment.plot(kind='line', figsize=(12, 6))
plt.title('Monthly Sentiment Trends')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.grid(True)
plt.show()

# 6. Calculate polarity scores for each text
df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Average monthly polarity trend
monthly_polarity = df.groupby('month')['polarity'].mean()
print(monthly_polarity.head())

# Plot average monthly polarity
monthly_polarity.plot(kind='line', figsize=(14, 6))
plt.title('Average Monthly Sentiment Polarity')
plt.xlabel('Month')
plt.ylabel('Average Polarity Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# 7. Employee-level average polarity and message count
employee_sentiment = df.groupby('from')['polarity'].mean().reset_index()
employee_sentiment.columns = ['Employee', 'Avg_Polarity']

employee_message_count = df.groupby('from').size().reset_index(name='Message_Count')

# Merge and rank employees by polarity
employee_stats = employee_sentiment.merge(employee_message_count, left_on='Employee', right_on='from').drop(columns=['from'])
employee_stats['Rank'] = employee_stats['Avg_Polarity'].rank(ascending=False, method='min').astype(int)
employee_stats = employee_stats.sort_values('Rank')
print(employee_stats.head(10))

# Visualize top 20 employees by sentiment
plt.figure(figsize=(12, 6))
sns.barplot(x='Avg_Polarity', y='Employee', data=employee_stats.head(20), palette="viridis")
plt.title('Top 20 Employees by Average Sentiment Polarity')
plt.xlabel('Average Polarity')
plt.ylabel('Employee')
plt.show()

# 8. Flight risk identification

# Monthly average polarity per employee
monthly_employee_sentiment = df.groupby(['from', 'month'])['polarity'].mean().reset_index()
employee_avg_polarity = monthly_employee_sentiment.groupby('from')['polarity'].mean()

# Threshold for flight risk
risk_threshold = 0.1
at_risk_employees = employee_avg_polarity[employee_avg_polarity < risk_threshold].index.tolist()

# Define function to calculate sentiment trend slope per employee
def sentiment_trend(employee_df):
    employee_df = employee_df.sort_values('month')
    X = np.arange(len(employee_df)).reshape(-1, 1)
    y = employee_df['polarity'].values
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0]

# Identify employees with negative sentiment trends
declining_employees = []
for emp in monthly_employee_sentiment['from'].unique():
    emp_df = monthly_employee_sentiment[monthly_employee_sentiment['from'] == emp]
    if len(emp_df) > 3:
        trend = sentiment_trend(emp_df)
        if trend < 0:
            declining_employees.append(emp)

# Combine risk lists
flight_risk_employees = set(at_risk_employees).union(set(declining_employees))

print(f"Identified {len(flight_risk_employees)} employees at flight risk.")

# Flight risk summary DataFrame
flight_risk_summary = []
for emp in flight_risk_employees:
    avg_polarity = employee_avg_polarity.loc[emp]
    emp_monthly = monthly_employee_sentiment[monthly_employee_sentiment['from'] == emp]
    trend = sentiment_trend(emp_monthly) if len(emp_monthly) > 3 else np.nan
    flight_risk_summary.append({'Employee': emp, 'Avg_Polarity': avg_polarity, 'Sentiment_Trend': trend})

flight_risk_df = pd.DataFrame(flight_risk_summary).sort_values(by=['Avg_Polarity', 'Sentiment_Trend'])
print(flight_risk_df.head(10))

# 9. Overall Sentiment Trend Using Linear Regression

# Prepare data
monthly_polarity_df = monthly_polarity.reset_index()
monthly_polarity_df['month_start'] = monthly_polarity_df['month'].dt.to_timestamp()
monthly_polarity_df['month_ordinal'] = monthly_polarity_df['month_start'].apply(lambda x: x.toordinal())

X = monthly_polarity_df['month_ordinal'].values.reshape(-1, 1)
y = monthly_polarity_df['polarity'].values

# Fit model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Print model equation
slope = model.coef_[0]
intercept = model.intercept_
print(f"Linear Regression model: polarity = {slope:.6f} * month_ordinal + {intercept:.6f}")

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(monthly_polarity_df['month_start'], y, marker='o', label='Actual Average Polarity')
plt.plot(monthly_polarity_df['month_start'], y_pred, color='red', label='Linear Regression Fit')
plt.xlabel('Month')
plt.ylabel('Average Sentiment Polarity')
plt.title('Overall Sentiment Trend with Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
