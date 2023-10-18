from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import warnings


def svm(x_train, x_test, y_train, y_test, test):
    warnings.filterwarnings('ignore')
    # Create and fit model
    model = LinearSVC(dual='auto')
    model.fit(x_train, y_train)

    # Predict and print score
    prediction = model.predict(x_test)
    score = classification_report(y_test, prediction)
    print('SVM Score:', score)

    final_prediction = model.predict(test)
    i = 0

    with open('in_domain_test_without_label.txt', 'r', encoding='utf-8') as input_file:
        input_data = input_file.readlines()

    # Write the original word and predicted tag for each line to the output file
    with (open('labeled100.txt', 'w', encoding='utf-8') as output_file):
        for original_word in input_data:
            original_word = original_word.strip()
            if original_word == "":
                output_line = "\n"
            else:
                output_line = f"{original_word}" + " " + final_prediction[i] + " \n"
                i += 1
            output_file.write(output_line)

