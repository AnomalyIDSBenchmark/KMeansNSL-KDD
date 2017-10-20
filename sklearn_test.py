import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

def load_data(file_path, cols=None):
    COL_NAMES = ["duration", "protocol_type", "service", "flag", "src_bytes",
                "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "labels"]

    data = pd.read_csv(file_path, names=COL_NAMES, index_col=False)

    NOM_IND = [1, 2, 3]
    BIN_IND = [6, 11, 13, 14, 20, 21]
    # Need to find the numerical columns for normalization
    NUM_IND = list(set(range(40)).difference(NOM_IND).difference(BIN_IND))

    # Scale all numerical data to [0-1]
    data.ix[:, NUM_IND] = minmax_scale(data.ix[:, NUM_IND])
    labels = data['labels']
    # Binary labeling
    labels = labels.apply(lambda x: x if x =='normal' else 'anomaly')
    del data['labels']
    data = pd.get_dummies(data)
    if cols is None:
        cols = data.columns
    else:
        map_data = pd.DataFrame(columns=cols)
        map_data = map_data.append(data)
        data = map_data.fillna(0)
        data = data[cols]
    return [data, labels, cols]


def evaluate_kmeans(data, labels, clf=None):
    if clf is None:
        clf = KMeans(n_clusters=4,init='random').fit(data)
    preds = clf.predict(data)
    ans = pd.DataFrame({'label':labels.values, 'kmean':preds})
    ans = ans.groupby(['kmean', 'label']).size()
    print(ans)

    # Get the larger number from each cluster
    correct = sum([anom if anom > norm else norm for anom, norm in zip(ans[::2],ans[1::2])])
    print("Total accuracy: {0:.1%}".format(correct/sum(ans)))
    return clf


train_data, train_labels, cols = load_data('data/KDDTrain+.csv')
clf = evaluate_kmeans(train_data, train_labels)

test_data, test_labels, cols = load_data('data/KDDTest+.csv', cols)
evaluate_kmeans(test_data, test_labels, clf)
