"""K-Means Classifier"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

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

ATTACKS = {
    'normal': 'normal',

    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',

    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',

    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'worm': 'R2L',

    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'httptunnel': 'U2R',
    'ps': 'U2R',
    'sqlattack': 'U2R',
    'xterm': 'U2R'
}

class KMeansNSL():

    def __init__(self):
        self.clf = None
        self.training = []
        self.cols = None
        self.testing = []

    def load_data(self, filepath):
        data = pd.read_csv(filepath, names=COL_NAMES, index_col=False)
        # Shuffle data
        data = data.sample(frac=1).reset_index(drop=True)
        NOM_IND = [1, 2, 3]
        BIN_IND = [6, 11, 13, 14, 20, 21]
        # Need to find the numerical columns for normalization
        NUM_IND = list(set(range(40)).difference(NOM_IND).difference(BIN_IND))

        # Scale all numerical data to [0-1]
        data.iloc[:, NUM_IND] = minmax_scale(data.iloc[:, NUM_IND])
        labels = data['labels']
        del data['labels']
        data = pd.get_dummies(data)
        return [data, labels]

    def load_training_data(self, filepath):
        data, labels = self.load_data(filepath)
        self.cols = data.columns
        self.training = [data, labels]

    def load_test_data(self, filepath):
        data, labels = self.load_data(filepath)
        map_data = pd.DataFrame(columns=self.cols)
        map_data = map_data.append(data)
        data = map_data.fillna(0)
        self.testing = [data[self.cols], labels]

    def train_clf(self):
        self.clf = KMeans(n_clusters=4, init='random').fit(self.training[0])

    def test_clf(self, train=False):
        if train:
            data, labels = self.training
        else:
            data, labels = self.testing
        preds = self.clf.predict(data)
        bin_labels = labels.apply(lambda x: x if x =='normal' else 'anomaly')
        ans = pd.DataFrame({'label':bin_labels.values, 'kmean':preds})
        return ans

    def evaluate_results(self, ans=None, train=False):
        if not ans:
            ans = self.test_clf(train)
        ans = ans.groupby(['kmean', 'label']).size()
        print(ans)

        # Get the larger number from each cluster
        correct = sum([anom if anom > norm else norm for anom, norm in zip(ans[::2], ans[1::2])])
        print("Total accuracy: {0:.1%}".format(correct/sum(ans)))
        return ans
