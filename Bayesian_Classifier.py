from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def bayesian_classifier(x_train, x_test, y_train, y_test):
    # Create and fit model
    model = MultinomialNB()
    model.fit(x_train, y_train)

    # Predict and print score
    prediction = model.predict(x_test)
    score = accuracy_score(y_test, prediction)
    print('Bayesian Score:', score)
