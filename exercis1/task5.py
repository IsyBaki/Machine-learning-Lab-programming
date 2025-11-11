import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_task5():
    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    df = pd.read_csv('olympics.txt', sep=',', quotechar='"')

    # Clean column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower()  # everything lowercase

    print("First 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())

    # ---------------------------------------------------------
    # 95th percentile of height for female competitors in Athletics
    # ---------------------------------------------------------
    athletics_female = df[(df['sport'] == 'Athletics') & (df['sex'] == 'F')]
    percentile_95 = np.percentile(athletics_female['height'].dropna(), 95)
    print(f"\n95th percentile of height (female athletes in Athletics): {percentile_95:.2f} cm")

    # ---------------------------------------------------------
    # Event with highest height variability (standard deviation)
    # ---------------------------------------------------------
    std_by_event = athletics_female.groupby('event')['height'].std().sort_values(ascending=False)
    most_variable_event = std_by_event.index[0]
    max_std = std_by_event.iloc[0]
    print(f"Event with highest height variability (female): {most_variable_event} (std={max_std:.2f})")

    # ---------------------------------------------------------
    # Average age of swimmers over time, by sex
    # ---------------------------------------------------------
    swimmers = df[df['sport'] == 'Swimming']
    avg_age = swimmers.groupby(['year', 'sex'])['age'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='year', y='age', hue='sex', data=avg_age, marker='o')
    plt.title('Average Age of Olympic Swimmers Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Age')
    plt.grid(True)
    plt.legend(title='Sex')
    plt.show()
