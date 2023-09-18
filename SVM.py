from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def svm(x_train, x_test, y_train, y_test):
    # Create and fit model
    model = LinearSVC(dual='auto')
    model.fit(x_train, y_train)

    # Predict and print score
    prediction = model.predict(x_test)
    score = accuracy_score(y_test, prediction)
    print('SVM Score:', score)
