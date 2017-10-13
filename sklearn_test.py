import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"]

data = pd.read_csv('data/KDDTrain+.csv', names=col_names, index_col=False)
nom_ind = [1, 2, 3]
bin_ind = [6, 11, 13, 14, 20, 21]
num_ind = list(set(range(40)).difference(nom_ind).difference(bin_ind))

data.ix[:, num_ind] = minmax_scale(data.ix[:, num_ind])
labels = data['labels']
labels = labels.apply(lambda x: 0 if x =='normal' else 1)
del data['labels']
kmeans_data = pd.get_dummies(data)
kmeans = KMeans(n_clusters=4,init='random').fit_predict(kmeans_data)

ans = pd.DataFrame({'label':labels.values, 'kmean':kmeans})
x = ans.groupby(['label', 'kmean']).size()
