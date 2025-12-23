import torch
import faiss
import numpy as np
from time import perf_counter
import scipy.sparse as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from torch.nn.functional import normalize
import torch_clustering



def build_affinity_matrix(X, k):
    X = X.clone().detach()
    X = X.cpu().numpy()
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    _, ind = index.search(X, k + 1)
    dist = np.array([np.linalg.norm(X[i] - X[ind[i][1:]], axis=1) for i in range(X.shape[0])])
    dist = torch.tensor(dist)
    aff = torch.exp(-dist ** 2 / 2)
    W = torch.zeros(X.shape[0], X.shape[0])
    for i in range(X.shape[0]):
        W[i, ind[i][1:]] = aff[i]
        W[ind[i][1:], i] = aff[i]
    adj = np.array(W)
    normalization = 'NormAdj'
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    return adj





def normalize_embeddings(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm



def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a = normalize_embeddings(a, eps)
    b = normalize_embeddings(b, eps)

    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt



def normalized_adjacency(adj):
    adj = adj
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()



def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()



def fetch_normalization(type):
    switcher = {
        'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        'NormAdj': normalized_adjacency,  # A' = (D)^-1/2 * ( A) * (D)^-1/2
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func



def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



def neraest_labels(instance, labels):
    instance = instance.clone().detach()
    instance = instance.cpu().numpy()
    labels = labels.clone().detach()
    labels = labels.cpu().numpy()
    index = faiss.IndexFlatL2(instance.shape[1])  # 使用 FlatL2 索引
    index.add(instance)
    k = 1
    distances, indices = index.search(instance, k)
    topK_labels = labels[indices]
    topK_labels = torch.from_numpy(topK_labels)
    topK_labels = torch.squeeze(topK_labels, dim=1)
    return torch.FloatTensor(topK_labels)



def knn_retrieve(V, T):
    V = V.clone().detach()
    V = V.cpu().numpy()
    T = T.clone().detach()
    T = T.cpu().numpy()
    nbrs = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    nbrs.fit(V, np.arange(V.shape[0]))
    distances, indices = nbrs.kneighbors(V)
    T_labels = T[indices]
    T_labels = torch.from_numpy(T_labels)
    T_labels = torch.squeeze(T_labels, dim=1)
    return T_labels



def cluster_pred(x, class_num):
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    g = x.cpu().detach().numpy()
    kmeans.fit_predict(g)
    labels = kmeans.labels_
    n_samples = len(labels)
    onehot_labels = np.zeros((n_samples, class_num))
    onehot_labels[np.arange(n_samples), labels] = 1
    predicted = torch.tensor(onehot_labels).to(torch.float32)
    return predicted



def compute_centers(x, psedo_labels,class_num):
    n_samples = x.size(0)
    if len(psedo_labels.size()) > 1:
        weight = psedo_labels.T
    else:
        weight = torch.zeros(class_num, n_samples).to(x)
        weight[psedo_labels, torch.arange(n_samples)] = 1
    weight = normalize(weight, p=1, dim=1)
    centers = torch.mm(weight, x)
    centers = normalize(centers, dim=1)
    return centers


def get_W(mv_data,num_views,k):
    W = []
    for i in range(num_views):
        result_graph = get_knn_graph(mv_data[i], k)
        W.append(result_graph)
    return W

def get_knn_graph(data, k):
    num_samples = data.size(0)
    graph = torch.zeros(num_samples, num_samples, dtype=torch.int32, device=data.device)

    for i in range(num_samples):
        distance = torch.sum((data - data[i])**2, dim=1)
        _, small_indices = torch.topk(distance, k, largest=False)
        graph[i, small_indices[1:]] = 1

    result_graph = torch.max(graph, graph.t())

    return result_graph


def compute_correlation_matrix(hs):
    num_views = len(hs)
    corr_matrix = torch.zeros((num_views, num_views))
    for i in range(num_views):
        for j in range(num_views):
            if i != j:
                corr_matrix[i, j] = torch.nn.functional.cosine_similarity(hs[i], hs[j], dim=1).mean()
    return corr_matrix


def psedo_labeling(dataset,num_clusters):
    psedo_labels = clustering(dataset,num_clusters)
    return psedo_labels


def clustering(features,num_clusters):
    kwargs = {
        'metric': 'cosine',
        'distributed': False,
        'random_state': 0,
        'n_clusters': num_clusters,
        'verbose': False
    }
    clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
    f = features.detach()
    psedo_labels = clustering_model.fit_predict(f.to(dtype=torch.float64))

    return psedo_labels


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()