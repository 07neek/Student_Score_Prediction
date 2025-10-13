import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_visulixation(filepath):
    df = pd.read_csv(filepath)

    plt.figure(figsize=(6, 4))
    plt.scatter(df['Study_Hours_per_Week'], df['Final_Exam_Score'], color='blue', alpha=0.6)
    plt.title("Study Hours vs Scores")
    plt.xlabel("Hours Studies")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.histplot(df["Final_Exam_Score"], bins=10, kde=True, color='green')
    plt.title("Distribution of Exam Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


    plt.figure(figsize=(4, 3))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap ="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    plot_visulixation("Intership_project\\data\\student_performance_dataset.csv")