
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
file_path = file_path = r'E:\Codecademy_Projekte\codecademy_linearregression\tennis_stats.csv'


# Laden und Untersuchen der Daten
df = pd.read_csv(file_path)
print("Head of DataFrame:")
print(df.head())
print("\nDataFrame Description:")
print(df.describe())
print("\nDataFrame Info:")
print(df.info())

# Scatterplots für verschiedene Variablen
plt.scatter(df["BreakPointsOpportunities"], df["Winnings"])
plt.title("Break Points Opportunities vs Winnings")
plt.xlabel("Break Points Opportunities")
plt.ylabel("Winnings")
plt.show()
plt.clf()

plt.scatter(df["DoubleFaults"], df["Winnings"])
plt.title("Double Faults vs Winnings")
plt.xlabel("Double Faults")
plt.ylabel("Winnings")
plt.show()
plt.clf()

plt.scatter(df["BreakPointsSaved"], df["Winnings"])
plt.title("Break Points Saved vs Winnings")
plt.xlabel("Break Points Saved")
plt.ylabel("Winnings")
plt.show()
plt.clf()

plt.scatter(df["BreakPointsConverted"], df["Winnings"])
plt.title("Break Points Converted vs Winnings")
plt.xlabel("Break Points Converted")
plt.ylabel("Winnings")
plt.show()
plt.clf()

# Lineare Regression für verschiedene Prädiktorvariablen
def linear_regression_analysis(x_var):
    x = df[[x_var]]
    y = df[["Winnings"]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) 

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = model.score(x_test, y_test)

    print(f"R^2 score for {x_var}: {score}")

# Einzelne Prädiktorvariablen
linear_regression_analysis("FirstServeReturnPointsWon")
linear_regression_analysis("BreakPointsOpportunities")
linear_regression_analysis("DoubleFaults")

# Lineare Regression mit multiplen Prädiktorvariablen
def multiple_linear_regression_analysis(variables):
    x = df[variables]
    y = df[["Winnings"]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) 

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = model.score(x_test, y_test)

    print(f"R^2 score for {variables}: {score}")

# Mehrere Prädiktorvariablen
multiple_linear_regression_analysis(["DoubleFaults", "BreakPointsOpportunities", "Wins", "Losses"])
multiple_linear_regression_analysis(["DoubleFaults", "BreakPointsOpportunities", "Ranking", "ReturnGamesWon"])
