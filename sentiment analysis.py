import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import timedelta


# 1. Load dataset

df = pd.read_excel('test.xlsx')
df['date'] = pd.to_datetime(df['date'])
print(df.info())
print(df.describe())
print(df.head())


# 2. Sentiment labeling (with neutral band)

def get_sentiment(text, neutral_band=0.05):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > neutral_band:
        return 'Positive', polarity
    elif polarity < -neutral_band:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

df[['Sentiment', 'polarity']] = df['Subject'].fillna('').astype(str).add(" " + df['body'].fillna('').astype(str)).apply(
    lambda x: pd.Series(get_sentiment(x))
)


# 3. EDA

plt.figure(figsize=(8,5))
sns.histplot(df['polarity'], bins=30, kde=True)
plt.title("Distribution of Sentiment Polarity")
plt.xlabel("Polarity")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

sns.countplot(x='Sentiment', data=df, palette="Set2")
plt.title('Sentiment Label Distribution')
plt.show()


# 4. Monthly sentiment trends

df['month'] = df['date'].dt.to_period('M')
monthly_sentiment = df.groupby(['month', 'Sentiment']).size().unstack(fill_value=0)

monthly_sentiment.plot(kind='line', figsize=(14,7))
plt.title('Monthly Sentiment Trends')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.grid(True)
plt.show()


# 5. Monthly sentiment scoring (+1/-1/0)

score_map = {"Positive": 1, "Negative": -1, "Neutral": 0}
df['score'] = df['Sentiment'].map(score_map)

monthly_scores = df.groupby(['from','month'])['score'].sum().reset_index()
monthly_scores.rename(columns={'from':'Employee','month':'Month','score':'MonthlyScore'}, inplace=True)

# Top 3 positive & negative employees per month
top3_positive, top3_negative = [], []
for m, sub in monthly_scores.groupby('Month'):
    sub_pos = sub.sort_values(['MonthlyScore','Employee'], ascending=[False, True]).head(3)
    sub_neg = sub.sort_values(['MonthlyScore','Employee'], ascending=[True, True]).head(3)
    top3_positive.append(sub_pos.assign(Month=m))
    top3_negative.append(sub_neg.assign(Month=m))

top3_positive = pd.concat(top3_positive).reset_index(drop=True)
top3_negative = pd.concat(top3_negative).reset_index(drop=True)

print("\n--- Top 3 Positive Employees per Month ---")
print(top3_positive)
print("\n--- Top 3 Negative Employees per Month ---")
print(top3_negative)


# 6. Flight risk detection (≥4 negatives in rolling 30-day window)

neg = df[df['Sentiment']=="Negative"].copy()
flight_risk_employees = set()

for emp, sub in neg.groupby('from'):
    sub = sub.sort_values('date')
    dates = sub['date'].tolist()
    i = 0
    for j in range(len(dates)):
        while i <= j and (dates[j] - dates[i]) > timedelta(days=30):
            i += 1
        if (j - i + 1) >= 4:
            flight_risk_employees.add(emp)
            break

print(f"\n--- Flight Risk Employees (≥4 negatives in rolling 30 days): {sorted(flight_risk_employees)}")


# 7. Sentiment trend analysis (Linear Regression)

monthly_polarity = df.groupby('month')['polarity'].mean().reset_index()
monthly_polarity['month_start'] = monthly_polarity['month'].dt.to_timestamp()
monthly_polarity['month_ordinal'] = monthly_polarity['month_start'].apply(lambda x: x.toordinal())

X = monthly_polarity[['month_ordinal']].values
y = monthly_polarity['polarity'].values

if len(monthly_polarity) >= 2:
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)

    print("\n--- Linear Regression Model ---")
    print(f"Equation: polarity = {slope:.6f} * month_ordinal + {intercept:.6f}")
    print(f"R² score: {r2:.3f}")

    plt.figure(figsize=(12,6))
    plt.plot(monthly_polarity['month_start'], y, marker='o', label='Actual Average Polarity')
    plt.plot(monthly_polarity['month_start'], y_pred, color='red', label=f'Linear Regression Fit (R²={r2:.2f})')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Polarity')
    plt.title('Overall Sentiment Trend with Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("\nNot enough data points for regression.")
