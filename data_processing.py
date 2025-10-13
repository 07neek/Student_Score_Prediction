import pandas as pd

def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath)
        print("Dataset loaded")
        return df
    except FileNotFoundError:
        print("Not found")
        return None
def inspect_dataset(df):
    print("\n📊 First 5 rows:")
    print(df.head(), '\n')

    print("🧠 Data Types:")
    print(df.dtypes, '\n')

    print("🧼 Missing Values:")
    print(df.isnull().sum(), '\n')

    print("📏 Statistics:")
    print(df.describe(), '\n')

if __name__ == '__main__':
    df = load_dataset("Intership_project\data\student_performance_dataset.csv")
    if df is not None:
        inspect_dataset(df)

