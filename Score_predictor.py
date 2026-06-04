import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load Dataset
data = pd.read_csv("student_data.csv")

# Features and Target
X = data[["Hours", "Attendance", "Assignments", "PreviousScore"]]
y = data["FinalScore"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = r2_score(y_test, predictions)

while True:

    print("\n==============================")
    print("AI STUDENT PERFORMANCE PREDICTOR")
    print("==============================")
    print("1. Predict Student Score")
    print("2. View Model Accuracy")
    print("3. Generate Graph")
    print("4. Exit")

    choice = input("\nEnter Choice: ")

    if choice == "1":

        hours = float(input("Study Hours: "))
        attendance = float(input("Attendance (%): "))
        assignments = float(input("Assignment Score (%): "))
        previous = float(input("Previous Exam Score: "))

        result = model.predict(
            [[hours, attendance, assignments, previous]]
        )

        score = round(result[0], 2)

        if score > 100:
            score = 100

        if score < 0:
            score = 0

        if score >= 90:
            grade = "A+"
            performance = "Outstanding"
        elif score >= 80:
            grade = "A"
            performance = "Excellent"
        elif score >= 70:
            grade = "B"
            performance = "Good"
        elif score >= 60:
            grade = "C"
            performance = "Average"
        else:
            grade = "D"
            performance = "Needs Improvement"

        print("\n------ RESULT ------")
        print("Predicted Score:", score)
        print("Grade:", grade)
        print("Performance:", performance)

        result_df = pd.DataFrame({
            "Predicted Score": [score],
            "Grade": [grade],
            "Performance": [performance]
        })

        result_df.to_csv("results.csv", index=False)

        print("\nResult saved in results.csv")

    elif choice == "2":

        print(
            f"\nModel Accuracy: {accuracy*100:.2f}%"
        )

    elif choice == "3":

        plt.figure(figsize=(8,5))

        plt.scatter(
            y_test,
            predictions
        )

        plt.xlabel("Actual Scores")
        plt.ylabel("Predicted Scores")

        plt.title(
            "Actual vs Predicted Student Performance"
        )

        plt.show()

    elif choice == "4":

        print("Thank You!")
        break

    else:

        print("Invalid Choice")