import torch
from Network import Network
from metric import valid,evaluate,t_SNE
import argparse
from dataloader import load_data
from sklearn.cluster import KMeans
import random
import numpy as np


Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument('--neighbor_num', default=5, type=int)
parser.add_argument('--feature_dim', default = 512, type=int)
parser.add_argument('--filter_dim', default=128, type=int)
parser.add_argument('--tau', default= 0.1 , type=float)
parser.add_argument('--gamma', default=0.01, type=float)
parser.add_argument('--lambda1', default=0.1, type=float)
parser.add_argument('--lambda2', default=0.01, type=float)
parser.add_argument('--lambda3', default=1, type=float)
parser.add_argument('--eta', default = 1.0, type=float)
parser.add_argument('-k', type=int, default=5)
parser.add_argument('-a', type=float, default=10)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if args.dataset== "caltech101":
    seed = 3
    setup_seed(seed)
if args.dataset== "RGB-D":
    seed = 15
    setup_seed(seed)



model = Network(view, dims, args.feature_dim, args.filter_dim, class_num, args.neighbor_num, device)
model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)

print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
acc_k, nmi_k, ari_k, pur_k, kmeans_vectors, fusion_vector, labels_vector = valid(model, device, dataset, view, data_size, class_num,eval_h=False,args=args)
print("Clustering results on kmeans clustering: ")
print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc_k, nmi_k, ari_k, pur_k))
