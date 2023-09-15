import pandas as pd


def load_data(filename):
    # Load data into pandas dataframe
    return pd.read_csv(filename, header=None, delimiter=' ', names=['Token', 'POS', 'Chunking'])


def extract_features(df):
    features = []

    # Loop through all tokens
    for index, row in df.iterrows():
        token = row['Token']

        # Extract features
        feature = {
            'word': token,
            'length': len(str(token)),
            'first_capital': str(token)[0].isupper(),
            'all_capital': str(token).isupper(),
            'number': str(token).isnumeric(),
            'next_word': df.loc[int(index) + 1, 'Token'] if index != df.shape[0] - 1 else '',
            'next_word2': df.loc[int(index) + 2, 'Token'] if index != df.shape[0] - 2 and index != df.shape[
                0] - 1 else '',
            'prev_word': df.loc[int(index) - 1, 'Token'] if index != 0 else '',
            'prev_word2': df.loc[int(index) - 2, 'Token'] if index != 1 and index != 0 else '',
        }

        # Append features to list
        features.append(feature)
    return features


def main():
    pass


if __name__ == '__main__':
    main()
