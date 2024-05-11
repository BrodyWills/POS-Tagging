import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from SVM import svm


def load_data(filename):
    # Load data into pandas dataframe
    return pd.read_csv(filename, header=None, delimiter=' ', names=['Token', 'POS', 'Chunking'])


def extract_features(df):
    # Initialize features list
    features = []

    # Loop through all tokens
    for index, row in df.iterrows():
        token = row['Token']

        # Extract features
        feature = {
            'word': str(token),
            'length': len(str(token)),
            'first_capital': 1 if str(token)[0].isupper() else 0,
            'all_capital': 1 if str(token).isupper() else 0,
            'next_word': str(df.loc[int(index) + 1, 'Token']) if index != df.shape[0] - 1 else '',
            'next_word2': str(df.loc[int(index) + 2, 'Token']) if index != df.shape[0] - 2 and index != df.shape[
                0] - 1 else '',
            'prev_word': str(df.loc[int(index) - 1, 'Token']) if index != 0 else '',
            'prev_word2': str(df.loc[int(index) - 2, 'Token']) if index != 1 and index != 0 else '',
            'prefix': str(token)[0:2],
            'suffix': str(token)[-2:],
        }

        # Append features to list
        features.append(feature)
    return features


def vectorize_features(train, test, test2):
    vectorizer = DictVectorizer()
    return [vectorizer.fit_transform(train), vectorizer.transform(test), vectorizer.transform(test2)]


def scale(train, test, test2):
    scaler = MaxAbsScaler()
    return [scaler.fit_transform(train), scaler.transform(test), scaler.transform(test2)]


def main():
    # Load data
    df = load_data('train.txt')
    print('Data loaded')

    test = load_data('test.txt')

    # Extract features
    features = extract_features(df)
    features_test = extract_features(test)
    print('Features extracted')

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(features, df['POS'], test_size=0.001)
    print('Data split')

    # Vectorize features
    [x_train, x_test, v_test] = vectorize_features(x_train, x_test, features_test)
    print('Vectorized features')

    # Scale features
    [x_train, x_test, v_test] = scale(x_train, x_test, v_test)
    print('Scaled')

    # Run svm
    svm(x_train, x_test, y_train, y_test, v_test)


if __name__ == '__main__':
    main()
