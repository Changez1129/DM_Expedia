import pdb
import pandas as pd
import numpy as np
import warnings

from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def get_map5(model, x, y):
    result = model.predict_proba(x)
    score = 0
    for i,val in enumerate(result):
        sorted_id = np.argsort(-val)
        for index in range(5):
            if sorted_id[index] == y[i]:
                score += 1/(index+1)
                break
    score /= len(result)
    return score


def get_dataset(x_path, y_path):
    # load datasets
    print("load dataset...")
    train = pd.read_csv(x_path).dropna()
    destinations = pd.read_csv(y_path)
    # train.shape

    train['date_time'] = pd.to_datetime(train['date_time'])
    train['year'] = train['date_time'].dt.year
    train['month'] = train['date_time'].dt.month
    train = train.drop(
        ["Unnamed: 0", "srch_ci", "srch_co", "date_time"], axis=1)
    # train.corr()["hotel_cluster"]

    train_booked = train.loc[train["is_booking"] == 1]
    train_totals = [train.groupby(['srch_destination_id', 'hotel_country', 'hotel_market', 'hotel_cluster'])[
        'is_booking'].agg(['sum', 'count'])]
    train_summary = pd.concat(train_totals).groupby(level=[0, 1, 2, 3]).sum()
    train_summary.dropna(inplace=True)
    # train_summary.head()

    train_summary['sum_and_count'] = 0.85 * \
        train_summary['sum'] + 0.15*train_summary['count']
    train_summary = train_summary.groupby(
        level=[0, 1, 2]).apply(lambda x: x.astype(float)/x.sum())
    train_summary.reset_index(inplace=True)
    # train_summary.head()

    train_pivot = train_summary.pivot_table(
        index=['srch_destination_id', 'hotel_country', 'hotel_market'], columns='hotel_cluster', values='sum_and_count').reset_index()
    # train_pivot.head()

    train_booked = pd.merge(train_booked, destinations,
                            how='left', on='srch_destination_id')
    train_booked = pd.merge(train_booked, train_pivot, how='left', on=[
                            'srch_destination_id', 'hotel_country', 'hotel_market'])
    train_booked.fillna(0, inplace=True)
    # train_booked.shape
    x = train_booked.drop(['user_id', 'hotel_cluster', 'is_booking'], axis=1)
    y = train_booked.hotel_cluster
    print("x shape: {} y shape: {}".format(x.shape, y.shape))
    return x, y

if __name__ == '__main__':

    data_train_path = "../input/train_train.csv"
    destination_train_path = "../input/destinations_train.csv"

    data_test_path = "../input/train_test.csv"
    destination_test_path = "../input/destinations_test.csv"

    # get dataset
    x_train, y_train = get_dataset(data_train_path, destination_train_path)
    x_test, y_test = get_dataset(data_test_path, destination_test_path)

    warnings.simplefilter("ignore")
    # KNN
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(x_train,y_train)
    map5_knn = get_map5(knn, x_test, y_test)
    print("map@5 of knn:{}".format(map5_knn))

    # LR
    lr = LogisticRegression(multi_class='ovr')
    lr.fit(x_train,y_train)
    map5_lr = get_map5(lr, x_test, y_test)
    print("map@5 of lr:{}".format(map5_lr))

    # Random Forest 
    rf = RandomForestClassifier(n_estimators=273,max_depth=10,random_state=0)
    rf.fit(x_train, y_train)
    map5_rf = get_map5(rf, x_test, y_test)
    print("map@5 of random forest:{}".format(map5_rf))

    # Naive Bayes
    nb = GaussianNB(priors=None)
    nb.fit(x_train, y_train)
    map5_nb = get_map5(nb, x_test, y_test)
    print("map@5 of naive bayes:{}".format(map5_nb))
