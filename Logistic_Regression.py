from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def logistic_regression(x_train, x_test, y_train, y_test):
    # Create and fit model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Predict and print score
    prediction = model.predict(x_test)
    score = accuracy_score(y_test, prediction)
    print('Logistic Score:', score)
