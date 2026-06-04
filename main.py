import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv")

X = data[["Hours"]]
y = data["Score"]

model = LinearRegression()
model.fit(X, y)

hours = float(input("Enter study hours: "))

prediction = model.predict([[hours]])

score = prediction[0]

# Limit score between 0 and 100
if score > 100:
    score = 100
elif score < 0:
    score = 0

print("Predicted Score:", round(score, 2))

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Student Score Prediction")
plt.show()