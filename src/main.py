from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle
import argparse


def load_data(filename):
    # read all the lines from source file
    try:
        path = list(Path(__file__).parent.parent.glob(f"data/{filename}"))[0]
        with open(path, 'r', encoding='utf-8') as file:
            data = file.readlines()
    except FileNotFoundError as e:
        print(e)
    except IndexError as e:
        print(e)

    attributes = []
    targets = []
    for line in data:
        target, attribute = line.replace('\n', '').split('\t')
        targets.append(int(target))
        attribute = attribute.split(' ')
        attribute = [word for word in attribute if len(word) > 2]
        attribute = ' '.join(attribute)
        attributes.append(attribute)
    
    data = pd.DataFrame({'attribute': attributes, 'target': targets})
    return data


def evaluate(X_train, y_train, X_test, model="bayes", optimize=False):
    """ Fit and predict based on input data.

    Args:
        model (str, optional):  "bayes"
                                "multibayes"
                                "tree"
                                "knn"
                                "svc"
                                "linsvc"
                                "forest"
                                "mlp"
                                "sgd"
                                "ada"
                                "hist"
                                "boost".
                                Defaults to "bayes".
    """

    # Naive Bayes
    if model == "bayes":
        clf = GaussianNB()
    
    # Multinomial Naive Bayes
    elif model == "multibayes":
        clf = MultinomialNB()

    # Decision Tree
    elif model == "tree":
    # {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 15, 'splitter': 'random'}
    # Accuracy 0.5222222222222223
        if optimize:
            parameters = {'splitter': (['best', 'random']),
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5, 10, 15, 50],
                        'min_samples_leaf': [1, 2, 5, 10, 50]}
            grid_search = GridSearchCV(DecisionTreeClassifier(),parameters,cv=2,return_train_score=True,n_jobs=-1, verbose=4)
            grid_search.fit(X_train, y_train)
            print(grid_search.best_params_)
            clf = DecisionTreeClassifier(**grid_search.best_params_)
        else:
            params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 15, 'splitter': 'random'}
            clf = DecisionTreeClassifier(**params)

    # KNN
    elif model == "knn":
    # Accuracy: 0.53
        clf = KNeighborsClassifier(n_neighbors=5)
        if optimize:
            parameters = {'n_neighbors': ([5, 10, 50, 200]),
                        'weights': ['uniform', 'distance'],
                        'algorithm': ['auto'],
                        'leaf_size': [10, 30, 50],
                        'p': [1, 2]}
            grid_search = GridSearchCV(KNeighborsClassifier(),parameters,cv=2,return_train_score=True,n_jobs=-1, verbose=4)
            grid_search.fit(X_train, y_train)
            print(grid_search.best_params_)
            clf = KNeighborsClassifier(**grid_search.best_params_)
        else:
            params = {'algorithm': 'auto', 'leaf_size': 10, 'n_neighbors': 5, 'p': 2, 'weights': 'distance'}
            clf = KNeighborsClassifier(**params)

    # SVC
    elif model == "svc":
    # Accuracy: 0.56  
        clf = SVC()

    # linear SVC
    elif model == "linsvc":
    # {'C': 1.0, 'dual': True, 'max_iter': 1000, 'penalty': 'l2', 'random_state': True}
    # Accuracy 0.58
        if optimize:
            parameters = {'penalty': (['l1', 'l2']),
                        'dual': [True, False],
                        'C': [1.0, 2.0, 5.0, 20.0],
                        'random_state': [True, False],
                        'max_iter': [1000, 5000]}
            grid_search = GridSearchCV(LinearSVC(),parameters,cv=2,return_train_score=True,n_jobs=-1, verbose=4)
            grid_search.fit(X_train, y_train)
            print(grid_search.best_params_)
            clf = LinearSVC(**grid_search.best_params_)
        else:
            params = {'C': 1.0, 'dual': True, 'max_iter': 1000, 'penalty': 'l2', 'random_state': True}
            clf = LinearSVC(**params)

    # Random Forest Classifier
    elif model == "forest":
    # {'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
    # Accuracy 0.53
        if optimize:
            params = dict(bootstrap=True, max_depth=None, max_features='sqrt', min_samples_split=10, n_estimators=100)
            parameters = {'max_features': (['sqrt']),
                        'n_estimators': [20, 200, 500],
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5, 10, 15],
                        'min_samples_leaf': [1, 2, 5, 10],
                        'bootstrap': [True, False]}
            grid_search = GridSearchCV(RandomForestClassifier(),parameters,cv=2,return_train_score=True,n_jobs=-1, verbose=4)
            grid_search.fit(X_train, y_train)
            print(grid_search.best_params_)
            clf = RandomForestClassifier(**grid_search.best_params_)
        else:
            params = {'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
            clf = RandomForestClassifier(**params)

    # MLP classifier
    elif model == "mlp":
    # Accuracy: 0.57
        if optimize:
            parameters = {'learning_rate': ["constant", "invscaling", "adaptive"],
                        'hidden_layer_sizes': [(50, 80, 150, 30), (50, 80, 200)],
                        'alpha': [1.e-01, 1.e-02, 1.e-03, 1.e-04, 1.e-05, 1.e-06],
                        'activation': ["logistic", "relu", "tanh"]}
            grid_search = GridSearchCV(MLPClassifier(),parameters,cv=2,return_train_score=True,n_jobs=-1, verbose=4)
            grid_search.fit(X_train, y_train)
            print(grid_search.best_params_)
            clf = MLPClassifier(**grid_search.best_params_)
        else:
            params = dict(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 150, 200), activation='relu', max_iter=10000, random_state=1,
                                verbose=True, learning_rate='adaptive', tol=0.00001)
            clf = MLPClassifier(**params)

    # SGD classifier
    elif model == "sgd":
    # Accuracy 0.57
        if optimize:
            parameters = {
                            'loss':['log_loss'],
                            'penalty':['elasticnet'],
                            'n_iter_no_change':[5],
                            'alpha':np.logspace(-4, 4, 10),
                            'l1_ratio':[0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.13,0.14,0.15,0.2]
                        }
            grid_search = GridSearchCV(SGDClassifier(),parameters,cv=2,return_train_score=True,n_jobs=-1, verbose=4)
            grid_search.fit(X_train, y_train)
            print(grid_search.best_params_)
            clf = SGDClassifier(**grid_search.best_params_)
        else:
            params = {'alpha': 0.0001, 'l1_ratio': 0.07, 'loss': 'log_loss', 'n_iter_no_change': 5, 'penalty': 'elasticnet'}
            clf = SGDClassifier(**params)

    # AdaBoost classifier
    elif model == "ada":
        clf = AdaBoostClassifier(n_estimators=200)

    # HistGradientBoosting classifier
    elif model == "hist":
        clf = HistGradientBoostingClassifier(max_iter=10)

    # GradientBoosting classifier
    elif model == "boost":
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)

    else:
        print("Wrong model type")
        return -1
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    return clf, y_pred, y_pred_train


def initial_training(datafile, model="multibayes", cross_val=False):
    data = load_data(datafile)
    X, y = data["attribute"], data["target"]

    modelname = model

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # TF-IDF
    tfidf_vect = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),# max_features=2500, 
                                 stop_words=stopwords.words('english'),# token_pattern="[^\W\d_]+", #strip_accents='unicode',
                                 max_df=20, min_df=2)
    
    X_train_tfidf = tfidf_vect.fit_transform(X_train)
    X_test_tfidf = tfidf_vect.transform(X_test)
    
    vect_array = tfidf_vect.transform(X).toarray()
    df = pd.DataFrame(data=vect_array, columns=tfidf_vect.get_feature_names_out())
    print(f"Feature vector length: {len(df.columns)}")
    
    clf, y_pred, y_pred_train = evaluate(X_train_tfidf, y_train, X_test_tfidf, model=modelname, optimize=0)

    if cross_val:
        print(f"Accuracy cross validation: {round(cross_val_score(clf, vect_array, y, cv=5).mean(), 2)}")

    print(f"Accuracy on training dataset: {round(accuracy_score(y_train, y_pred_train), 2)}")
    print(f"Confussion matrix:\n {confusion_matrix(y_test, y_pred)}")

    # save the model to memory
    filepath = list(Path(__file__).parent.parent.glob(f"models"))[0]
    filename_model = f'{filepath}/{model}.sav'
    filename_vect = f'{filepath}/vect.sav'
    pickle.dump(clf, open(filename_model, 'wb'))
    pickle.dump(tfidf_vect, open(filename_vect, 'wb'))


def load_model(model):
    if model == "multibayes":
        filename = "multibayes.sav"
    elif model == "linsvc":
        filename = "linsvc.sav"
    elif model == "sgd":
        filename = "sgd.sav"
    elif model == "mlp":
        filename = "mlp.sav"
    
    path_model = list(Path(__file__).parent.parent.glob(f"models/{filename}"))[0]
    path_vect = list(Path(__file__).parent.parent.glob(f"models/vect.sav"))[0]
    
    loaded_model = pickle.load(open(path_model, 'rb'))
    loaded_vect = pickle.load(open(path_vect, 'rb'))

    return loaded_model, loaded_vect


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, required=False, default="multibayes", help="The model type: multibayes, linsvc, mlp, sgd.")
    parser.add_argument("-file_predict", type=str, required=False, help="File to perdict (has target values), filename of .txt extension in ./data folder.")
    parser.add_argument("-file_classify", type=str, required=False, help="File to classify, filename of .txt extension in ./data folder")
    parser.add_argument("-file_train", type=str, required=False, help="File to train on, filename of .txt extension in ./data folder.")
    parser.add_argument("-cross_val", type=bool, required=False, default=False, help="If cross validation is to be used")
    args = parser.parse_args()

    model = args.model
    cross_val = args.cross_val
    learn = False

    if args.file_predict:
        file_predict = args.file_predict
        path = list(Path(__file__).parent.parent.glob(f"data/{args.file_predict}"))[0]
        if not path.exists():
            print("The target directory doesn't exist")
            raise SystemExit(1)

    if args.file_train:
        learn = True
        file_train = args.file_train
        path = list(Path(__file__).parent.parent.glob(f"data/{args.file_train}"))[0]
        if not path.exists():
            print("The target directory doesn't exist")
            raise SystemExit(1)

    if args.file_classify:
        file_classify = args.file_classify
        path = list(Path(__file__).parent.parent.glob(f"data/{args.file_classify}"))[0]
        if not path.exists():
            print("The target directory doesn't exist")
            raise SystemExit(1)


    if learn:
        initial_training(datafile=file_train, model=model, cross_val=args.cross_val)

    if args.file_predict:
        clf, vect = load_model(model=model)  # load appropriate model
        data = load_data(file_predict)
        X, y = data["attribute"], data["target"]
        X_vect = vect.transform(X)
        pred = clf.predict(X_vect)
        
        print(f"Prediction score:\n{classification_report(y, pred)}")

        print(f"Predicitons:\n{pred}")

    if not args.file_predict and not args.file_train:
        print("Nothing happened. Please input arguments.")
