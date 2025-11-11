import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_task4():
    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    df = pd.read_csv('flights.txt', sep=',', quotechar='"')

    print("First 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

    # ---------------------------------------------------------
    # Example exploration 1: Best time of day to fly
    # ---------------------------------------------------------
    df['Hour'] = df['DepTime'] // 100
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Hour', y='DepDelay', data=df)
    plt.title('Departure Delay by Hour of the Day (Austin Flights)')
    plt.xlabel('Hour of Departure')
    plt.ylabel('Departure Delay (minutes)')
    plt.grid(True)
    plt.show()

    # ---------------------------------------------------------
    # Example exploration 2: Airline comparison
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='UniqueCarrier', y='DepDelay', data=df)
    plt.title('Departure Delay by Airline (Austin Flights)')
    plt.xlabel('Airline')
    plt.ylabel('Departure Delay (minutes)')
    plt.xticks(rotation=45)
    plt.show()

    # ---------------------------------------------------------
    # Example exploration 3: Delay patterns by month
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Month', y='DepDelay', data=df)
    plt.title('Departure Delay by Month (Austin Flights)')
    plt.xlabel('Month')
    plt.ylabel('Departure Delay (minutes)')
    plt.show()

    # Additional plots can be added for destination trends, worst airports, etc.
