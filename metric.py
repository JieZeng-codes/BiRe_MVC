from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from utils.utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from matplotlib.ticker import FuncFormatter



def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = v_measure_score(label, pred)#归一化互信息
    ari = adjusted_rand_score(label, pred)#调整兰德指数
    acc = cluster_acc(label, pred)#准确率
    pur = purity(label, pred)#纯度
    return nmi, ari, acc, pur


def inference(loader, model, device, view, data_size,args):
    model.eval()
    soft_vector = []
    pred_vectors = []
    Hs = []
    Zs = []
    for v in range(view):
        pred_vectors.append([])
        Hs.append([])
        Zs.append([])
    labels_vector = []
    fusion_vector = []
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, f, adjs = model.forward_fusion(xs,args)
            hc = torch.cat(hs, dim=1)


        labels_vector.extend(y.numpy())
        fusion_vector.extend(f.cpu().detach().numpy())
        # fusion_vector.extend(hc.cpu().detach().numpy())

    # num, class_num = preds.shape
    unique_labels = np.unique(labels_vector)
    class_num = len(unique_labels)
    kmeans = KMeans(n_clusters=class_num, n_init=10)
    kmeans_vectors = kmeans.fit_predict(fusion_vector)
    kmeans_vectors = kmeans_vectors.flatten()
    labels_vector = np.array(labels_vector).reshape(data_size)

    return labels_vector, kmeans_vectors, fusion_vector


def valid(model, device, dataset, view, data_size, class_num, eval_h=False,args=None):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    labels_vector, kmeans_vectors, fusion_vector = inference(test_loader, model, device, view, data_size,args)
    nmi_k, ari_k, acc_k, pur_k = evaluate(labels_vector, kmeans_vectors)

    return acc_k, nmi_k, ari_k,  pur_k, kmeans_vectors, fusion_vector,labels_vector


def t_SNE(X, y, dataset, epoch, n_components=2, perplexity=10, learning_rate=200, random_state=42, title=""):

    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                learning_rate=learning_rate, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)

    unique_labels = np.unique(y)
    palette = sns.color_palette("hls", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(8, 8), dpi=800)

    for i in range(X_tsne.shape[0]):
        label = y[i]
        color = color_map[label]
        ax.text(X_tsne[i, 0], X_tsne[i, 1], str(label),
                fontsize=10, color=color,
                ha='center', va='center', weight='bold')

    margin = 5
    x_min, x_max = X_tsne[:, 0].min() - margin, X_tsne[:, 0].max() + margin
    y_min, y_max = X_tsne[:, 1].min() - margin, X_tsne[:, 1].max() + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_dir = "t_SNE/"
    os.makedirs(save_dir, exist_ok=True)
    adjusted_epoch = epoch - 200
    filename = f"{dataset}_{adjusted_epoch}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=800, bbox_inches='tight', pad_inches=0)
    print(f"Saved high-res image to: {save_path}")

    plt.show()

def plot_all_metrics(history, dataset_name,
                     ylim_left=(0.3,1), ylim_right=(-20, 110),
                     fontsize_title=20, fontsize_label=20, fontsize_tick=20, fontsize_legend=12):

    epochs = list(range(len(history["acc"])))
    max_epoch = max(epochs)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(epochs, history["acc"], label="ACC", color="#8552a1", linewidth=3, linestyle='-')
    ax1.plot(epochs, history["nmi"], label="NMI", color="#f58220", linewidth=3, linestyle='-')
    ax1.plot(epochs, history["ari"], label="ARI", color="#1d953f", linewidth=3, linestyle='-')
    ax1.set_xlabel("Epoch", color='black', fontsize=fontsize_label)
    ax1.set_ylabel("ACC, NMI and ARI", color='black', fontsize=fontsize_label)
    ax1.tick_params(axis="y", direction="in", colors='black', labelsize=fontsize_tick)
    ax1.tick_params(axis="x", direction="in", colors='black', labelsize=fontsize_tick)
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax1.grid(False)
    ax1.set_xlim(left=0, right=max_epoch)
    ax1.set_ylim(bottom=ylim_left[0], top=ylim_left[1])

    def left_formatter(x, pos):
        if x == 0:
            return '0'
        return f'{x:g}'
    ax1.yaxis.set_major_formatter(FuncFormatter(left_formatter))

    ax2 = ax1.twinx()
    ax2.plot(epochs[1:], history["loss"][1:], label="Loss", color="#d62728", linewidth=3, linestyle='-')
    ax2.set_ylabel("Loss", color='black', fontsize=fontsize_label)
    ax2.tick_params(axis="y", direction="in", colors='black', labelsize=fontsize_tick)
    ax2.spines['right'].set_color('black')
    ax2.spines['top'].set_color('black')
    ax2.grid(False)
    ax2.set_xlim(left=0, right=max_epoch)
    ax2.set_ylim(bottom=ylim_right[0], top=ylim_right[1])

    def right_formatter(x, pos):
        if x == 0:
            return '0'
        return f'{x:g}'
    ax2.yaxis.set_major_formatter(FuncFormatter(right_formatter))

    plt.title(dataset_name, fontsize=fontsize_title, fontweight="bold", color='black')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right",  fontsize=fontsize_legend)

    plt.tight_layout()
    plt.savefig(f"D:/论文/论文/想法/图片/{dataset_name}.svg", bbox_inches="tight")
    plt.show()