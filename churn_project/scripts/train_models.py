import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# load data
df = pd.read_csv("../data/churn.csv")

# preprocessing (same as notebook)
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df = df.dropna()

y = df["Churn Value"]
X = df.drop(["Churn Label", "Churn Value"], axis=1)

X = pd.get_dummies(X, drop_first=True)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("../app/model.pkl", "wb"))

print("Model trained and saved")