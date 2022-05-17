from collections import Counter
import json
from pathlib import Path
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def extract_features(vocab, data_dir, feature_field, tokenizer, feature_name):
    """
    Extract and save different features based on vocab of the features.
    # Parameters
    vocab : `dict[str, int]`, required.
        A map from the word type to the index of the word.
    data_dir : `Path`, required.
        Directory of the dataset
    tokenizer : `Callable`, required.
        Tokenizer with a method `.tokenize` which returns list of tokens.
    feature_name : `str`, required.
        Name of the feature, such as unigram_binary.
    # Returns
        `None`
    """
    # Extract and save the vocab and features.

    data_dir = Path(data_dir)
    #splits = ['train','test']
    splits = ['train']

    gram, mode = feature_name.split('_')
    if gram not in ['unigram', 'bigram'] or mode not in ['binary', 'count']:
        raise NotImplementedError

    for split in splits:
        datapath = data_dir.joinpath(f'{split}_clean.csv')
        print('datapath',datapath)
        data_df = pd.read_csv(datapath)
        data_df = data_df[data_df[feature_field].notna()]
        print('data df cols',data_df.columns)
        features = list(data_df[feature_field])
        
        sent_lengths = []
        values, rows, cols = [], [], []
        labels = list(data_df['label'])
        print(f"\nExtract {gram} {mode} features from {datapath}")
        for i, line in enumerate(features):
            if i % 1000 == 1:
                print(f"Processing {i}/{len(features)} row")
            #label = int(line[0])
            tokens = tokenizer.tokenize(line.strip())
            # tokens = line[1:].strip().split(  )  # Tokenizing differently affects the results.
            tokens = [t.lower() for t in tokens]
            tokens = [t if t in vocab else '<unk>' for t in tokens]
            if gram.find('bigram') != -1:
                tokens.extend(
                    [tokens[i] + ' ' + tokens[i + 1] for i in range(len(tokens) - 1)])
            feature = {}
            for tk in tokens:
                if tk not in vocab:
                    continue
                if mode == 'binary':
                    feature[vocab[tk]] = 1
                elif mode == 'count':
                    feature[vocab[tk]] = feature.get(vocab[tk], 0) + 1
                else:
                    raise NotImplementedError
            for j in feature:
                values.append(feature[j])
                rows.append(i)
                cols.append(j)
            sent_lengths.append(len(tokens))
            #labels.append(label)

        features = sparse.csr_matrix((values, (rows, cols)),
                                     shape=(len(features), len(vocab)))
        print(f"{split} feature matrix shape: {features.shape}")
        output_feature_filepath = data_dir.joinpath(f'{split}_{gram}_{mode}_features.npz')
        sparse.save_npz(output_feature_filepath, features)

        np_labels = np.asarray(labels)
        print(f"{split} label array shape: {np_labels.shape}")
        output_label_filepath = data_dir.joinpath(f'{split}_labels.npz')
        np.savez(output_label_filepath, np_labels)
        

def fit_and_eval_logistic_regression(data_dir: Path,
                                     feature_name: str,
                                     tune: bool = False) -> LogisticRegression:
    """
    Fit and evaluate the logistic regression model using the scikit-learn library.
    # Parameters
    data_dir : `Path`, required
        The data directory.
    feature_name : `str`, required.
        Name of the feature, such as unigram_binary.
    tune : `bool`, optional
        Whether or not to tune the hyperparameters of the regularization strength
        of the model of the [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
    # Returns
        model_trained: `LogisticRegression`
            The object of `LogisticRegression` after it is trained.
    """
    # Implement logistic regression with scikit-learn.
    # Print out the accuracy scores on dev and test data.

    splits = ['train', 'dev', 'test']
    features, labels = {}, {}

    for split in splits:
        features_path = data_dir.joinpath(f'{split}_{feature_name}_features.npz')
        labels_path = data_dir.joinpath(f'{split}_labels.npz')
        features[split] = sparse.load_npz(features_path)
        labels[split] = np.load(labels_path)['arr_0']
    best_dev, best_model = 0, None
    if tune:
        for c in np.linspace(-5, 5, 11):
            clf = LogisticRegression(random_state=42,
                                     max_iter=100,
                                     fit_intercept=False,
                                     C=np.exp2(c))
            clf.fit(features['train'], labels['train'])
            dev_preds = clf.predict(features['dev'])
            dev_accuracy = accuracy_score(labels['dev'], dev_preds)
            print(c, dev_accuracy)
            if dev_accuracy > best_dev:
                best_dev, best_model = dev_accuracy, clf
    else:
        best_model = LogisticRegression(random_state=42,
                                        max_iter=100,
                                        fit_intercept=False)
        best_model.fit(features['train'], labels['train'])

    preds = {
        'dev': best_model.predict(features['dev']),
        'test': best_model.predict(features['test'])
    }
    for splt, splt_preds in preds.items():
        print("{} accuracy: {:.4f}".format(splt, accuracy_score(labels[splt],
                                                                splt_preds)))
        print("{} macro f1: {:.4f}".format(
            splt, f1_score(labels[splt], splt_preds, average='macro')))

    return best_model


if __name__ == "__main__":

    vocab_filepath = data_dir.joinpath('bigram_vocab.json')
    extract_features(vocab=json.load(open(vocab_filepath)),
                    tokenizer=tokenizer,
                    data_dir=data_dir,
                    feature_name='bigram_count')
    fit_and_eval_logistic_regression(feature_name='unigram_binary',
                                 data_dir=Path('data'),
                                 tune=False)