
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Sample data
data = {
    'GPA': [2.1, 3.5, 1.8, 2.9, 3.2, 1.6, 2.3, 3.7],
    'Attendance_Percentage': [60, 85, 55, 70, 80, 50, 65, 90],
    'Backlogs': [3, 0, 4, 1, 0, 5, 2, 0],
    'Parent_Income': [30000, 80000, 20000, 50000, 75000, 15000, 45000, 95000],
    'Residential_Status': [1, 0, 1, 0, 1, 1, 0, 0],
    'Gender': [0, 1, 0, 0, 1, 0, 1, 1],
    'Participation_Score': [3, 9, 1, 6, 8, 2, 5, 10],
    'Dropout': [1, 0, 1, 0, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

X = df.drop("Dropout", axis=1)
y = df["Dropout"]

# Train the model
model = LogisticRegression(max_iter=500)
model.fit(X, y)

# Save the model
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
