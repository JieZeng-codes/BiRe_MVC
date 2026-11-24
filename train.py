from torch import nn
from Network import Network
from metric import valid, evaluate,t_SNE,plot_all_metrics
import torch.nn.functional as F
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import warnings
from utils.utils import *
import time
from BT_filtering import *


st = time.time()
torch.set_num_threads(4)
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

Dataname = 'BBCSport'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument('--neighbor_num', default=5, type=int)
parser.add_argument('--feature_dim', default=512, type=int)
parser.add_argument('--filter_dim', default=128, type=int)
parser.add_argument('--tau', default=0.1, type=float)
parser.add_argument('--gamma', default=0.01, type=float)
parser.add_argument('--lambda1', default=0.1, type=float)
parser.add_argument('--lambda2', default=0.01, type=float)
parser.add_argument('--lambda3', default=1, type=float)
parser.add_argument('--eta', default=1.0, type=float)
parser.add_argument('-k', type=int, default=5)
parser.add_argument('-a', type=float, default=50)


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

seed = 10


if args.dataset == 'BBCSport':
    args.neighbor_num = 5
    args.lambda2 = 0.01
    args.lambda3 = 1
    args.mse_epochs = 200
    args.con_epochs = 250
if args.dataset == 'BBC':
    args.neighbor_num = 3
    args.lambda2 = 1
    args.lambda3 = 0.05
    args.mse_epochs = 200
    args.con_epochs = 300
if args.dataset == 'caltech101':
    args.neighbor_num = 12
    args.lambda2 = 1
    args.lambda3 = 0.001
    args.mse_epochs = 200
    args.con_epochs = 200
if args.dataset == 'Reuters':
    args.neighbor_num = 5
    args.lambda2 = 0.005
    args.lambda3 = 0.5
    args.mse_epochs = 200
    args.con_epochs = 300
if args.dataset == 'Scene_15':
    args.neighbor_num = 3
    args.lambda1 = 0.1
    args.lambda2 = 0.1
    args.lambda3 = 1
    args.mse_epochs = 200
    args.con_epochs = 200
if args.dataset == 'BDGP':
    args.learning_rate = 0.00002
    args.neighbor_num = 3
    args.lambda2 = 1
    args.lambda3 = 1
    args.mse_epochs = 200
    args.con_epochs = 200
if args.dataset == 'RGB-D':
    args.learning_rate = 0.0002
    args.neighbor_num = 8
    args.lambda2 = 0.5
    args.lambda3 = 0.01
    args.mse_epochs = 200
    args.con_epochs = 300
if args.dataset == 'Digit4k':
    args.neighbor_num = 5
    args.lambda2 = 0.01
    args.lambda3 = 1
    args.mse_epochs = 200
    args.con_epochs = 200
if args.dataset == 'NUS':
    args.neighbor_num = 5
    args.lambda2 = 0.05
    args.lambda3 = 1
    args.mse_epochs = 200
    args.con_epochs = 200
if args.dataset == 'MNIST':
    args.neighbor_num = 5
    args.lambda2 = 0.01
    args.lambda3 = 1
    args.mse_epochs = 200
    args.con_epochs = 250
    seed = 10
if args.dataset == 'Digit4k':
    args.neighbor_num = 5
    args.lambda2 = 0.01
    args.lambda3 = 1
    args.mse_epochs = 200
    args.con_epochs = 200
    seed = 10
if args.dataset == 'NUSWIDEOBJ':
    args.neighbor_num = 7
    args.lambda2 = 0.01
    args.lambda3 = 1
    args.mse_epochs = 200
    args.con_epochs = 100
    seed = 10
if args.dataset == 'cifar10':
    args.neighbor_num = 5
    args.lambda2 = 0.01
    args.lambda3 = 1
    args.mse_epochs = 100
    args.con_epochs = 100
    seed = 10
if args.dataset == 'MNIST_USPS':
    args.neighbor_num = 5
    args.lambda2 = 0.01
    args.lambda3 = 1
    args.mse_epochs = 200
    args.con_epochs = 200
    seed = 10

print('neighbor_num: {}'.format(args.neighbor_num), 'lambda1: {}'.format(args.lambda1),
      'lambda2: {}'.format(args.lambda2), 'lambda3: {}'.format(args.lambda3))



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True





setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)


data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)



def add_noise(input, noise_factor=0.1):
    noise = torch.randn_like(input) * noise_factor
    return input + noise



def pretrain(epoch):
    tot_loss = 0.
    MSE = nn.MSELoss()
    xns = []
    all_labels = []
    all_hs = [[] for _ in range(view)]
    for batch_idx, (xs, labels, _) in enumerate(data_loader):
        for v in range(view):
            xn = add_noise(xs[v])
            xns.append(xn)
            xns[v] = xns[v].to(device)
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, hs, f = model(xns)

        for v in range(view):
            all_hs[v].append(hs[v].detach().cpu())

        all_labels.append(labels.cpu())

        loss_list = []
        for v in range(view):
            loss_list.append(F.mse_loss(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()


    for v in range(view):
        all_hs[v] = torch.cat(all_hs[v], dim=0)

    all_labels = torch.cat(all_labels, dim=0)

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    return all_hs, all_labels, all_hs,tot_loss



def model_train(epoch):
    total_loss = 0.
    last_model_state_dict = model.state_dict()
    for batch_idx, (xs, labels, _) in enumerate(data_loader):
        xns = []
        for v in range(view):

            xn = add_noise(xs[v])
            xns.append(xn)
            xns[v] = xns[v].to(device)
            xs[v] = xs[v].to(device)

        optimizer.zero_grad()
        xrs, hs, zs = model(xns)
        hs, f, adjs = model.forward_fusion(xs,args)


        loss_list = []

        psedo_label = psedo_labeling(f,class_num)
        y_matrix = (psedo_label.view(-1, 1) == psedo_label.view(1, -1)).int()
        q_centers = compute_centers(f, psedo_label,class_num)


        for v in range(view):

            hs[v] = filter(adjs[v], f, hs[v],device)
            w = get_knn_graph(hs[v], args.neighbor_num)
            k_centers = compute_centers(hs[v], psedo_label,class_num)

            loss_list.append(F.mse_loss(xs[v], xrs[v]))
            loss_list.append(criterion.forword_feature(f.T, hs[v].T))
            loss_list.append(criterion.compute_cluster_loss(q_centers, k_centers, psedo_label,class_num))
            loss_list.append(criterion.feature_loss(hs[v], f, w, y_matrix))

        loss_list.append(criterion.BarlowTwins_multi(device, hs,args.filter_dim))




        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(total_loss / len(data_loader)))
    return f, labels, last_model_state_dict,total_loss/len(data_loader)



accs = []
nmis = []
purs = []
aris = []


if not os.path.exists('./models'):
    os.makedirs('./models')


T = 1
for i in range(T):
    print("ROUND:{}".format(i + 1))
    model = Network(view, dims, args.feature_dim, args.filter_dim, class_num, args.neighbor_num, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.gamma,args.lambda1, args.lambda2, args.lambda3, args.eta,
                      device).to(device)

    epoch = 1
    loss = 0.0

    while epoch <= args.mse_epochs:
        zs, labels, hs , loss= pretrain(epoch)
        epoch += 1

    best_avg = 0
    ac_best = 0
    nm_best = 0
    ari_best = 0
    epoch_best = 0
    avg = 0
    best_fusion_vector = []
    best_kmeans_vectors = []
    best_labels_vector = []

    while epoch <= args.mse_epochs + args.con_epochs:

        fusion, labels, last_model_state_dict ,Loss = model_train(epoch)

        # acc_k, nmi_k, ari_k, pur_k, kmeans_vectors, fusion_vector, labels_vector= valid(model, device, dataset, view, data_size, class_num,
        #                                                 eval_h=False, args=args)

        # avg = (acc_k + nmi_k + ari_k) / 3
        # if avg > best_avg:
        #     ac_best = acc_k
        #     nm_best = nmi_k
        #     ari_best = ari_k
        #     best_avg = avg
        #     epoch_best = epoch
        #     best_fusion_vector = fusion_vector
        #     best_labels_vector = labels_vector
        #     print(f"Best model saved at epoch {epoch} with ac_best {ac_best}")
            # best_model_state_dict = last_model_state_dict
            # torch.save(best_model_state_dict, './models/' + args.dataset + '.pth')

        if epoch % 10 == 0:
            acc_k, nmi_k, ari_k, pur_k, kmeans_vectors, fusion_vector, labels_vector = valid(model, device, dataset,
                                                                                             view, data_size, class_num,
                                                                                             eval_h=False, args=args)
            print("Clustering results on kmeans clustering: ")
            print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc_k, nmi_k, ari_k, pur_k))

        epoch += 1

    # print(f"Best model saved at epoch {epoch_best} with ac_best: {ac_best} nm_best:{nm_best} ari_best:{ari_best}")

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')








