import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ==========================
# LOAD DATASET
# ==========================

data = pd.read_csv("Student_data.csv")

X = data[["Hours", "Attendance", "Assignments", "PreviousScore"]]
y = data["FinalScore"]

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================

model = LinearRegression()
model.fit(X_train, y_train)

# ==========================
# MODEL ACCURACY
# ==========================

predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)

# ==========================
# MAIN MENU
# ==========================

while True:

    print("\n" + "=" * 35)
    print(" AI STUDENT PERFORMANCE PREDICTOR ")
    print("=" * 35)

    print("\n1. Predict Student Score")
    print("2. View Model Accuracy")
    print("3. Generate Graph")
    print("4. Exit")

    choice = input("\nEnter Choice: ")

    # ==========================
    # PREDICT SCORE
    # ==========================

    if choice == "1":

        hours = float(input("Study Hours: "))
        attendance = float(input("Attendance (%): "))
        assignments = float(input("Assignment Score (%): "))
        previous = float(input("Previous Exam Score: "))

        input_data = pd.DataFrame({
            "Hours": [hours],
            "Attendance": [attendance],
            "Assignments": [assignments],
            "PreviousScore": [previous]
        })

        result = model.predict(input_data)

        score = round(float(result[0]), 2)

        if score > 100:
            score = 100

        if score < 0:
            score = 0

        # ==========================
        # GRADE SYSTEM
        # ==========================

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

        # ==========================
        # DISPLAY RESULT
        # ==========================

        print("\n" + "=" * 30)
        print("       FINAL RESULT")
        print("=" * 30)

        print(f"Predicted Score : {score}")
        print(f"Grade           : {grade}")
        print(f"Performance     : {performance}")

        print("=" * 30)

        # ==========================
        # SAVE RESULT
        # ==========================

        result_df = pd.DataFrame({
            "Predicted Score": [score],
            "Grade": [grade],
            "Performance": [performance]
        })

        result_df.to_csv("results.csv", index=False)

        print("\nResult saved in results.csv")

        input("\nPress Enter to continue...")

    # ==========================
    # ACCURACY
    # ==========================

    elif choice == "2":

        print("\n" + "=" * 30)
        print(" MODEL PERFORMANCE ")
        print("=" * 30)

        print(f"Accuracy: {accuracy * 100:.2f}%")

        print("=" * 30)

        input("\nPress Enter to continue...")

    # ==========================
    # GRAPH
    # ==========================

    elif choice == "3":

        plt.figure(figsize=(8, 5))

        plt.scatter(y_test, predictions)

        plt.xlabel("Actual Scores")
        plt.ylabel("Predicted Scores")

        plt.title(
            "Actual vs Predicted Student Performance"
        )

        plt.grid(True)

        plt.show()

    # ==========================
    # EXIT
    # ==========================

    elif choice == "4":

        print("\nThank you for using the project!")
        break

    # ==========================
    # INVALID INPUT
    # ==========================

    else:

        print("\nInvalid Choice!")