from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import hdf5storage

class BBCSport(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'BBCSport.mat')['Y'].astype(np.int32).reshape(544, )
        self.V1 = scipy.io.loadmat(path + 'BBCSport.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'BBCSport.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 544

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(3183)
        x2 = self.V2[idx].reshape(3203)

        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class BBC(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'BBC.mat')['Y'].astype(np.int32).reshape(685, )
        self.V1 = scipy.io.loadmat(path + 'BBC.mat')['X'][0,0].astype(np.float32).toarray()
        self.V2 = scipy.io.loadmat(path + 'BBC.mat')['X'][0,1].astype(np.float32).toarray()
        self.V3 = scipy.io.loadmat(path + 'BBC.mat')['X'][0,2].astype(np.float32).toarray()
        self.V4 = scipy.io.loadmat(path + 'BBC.mat')['X'][0,3].astype(np.float32).toarray()

    def __len__(self):
        return 685

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(4659)
        x2 = self.V2[idx].reshape(4633)
        x3 = self.V3[idx].reshape(4665)
        x4 = self.V4[idx].reshape(4684)
        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3),torch.from_numpy(x4)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class caltech101(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'caltech101.mat')['gt'].astype(np.int32).reshape(8677, )
        self.V1 = scipy.io.loadmat(path + 'caltech101.mat')['X'][0,0].T.astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'caltech101.mat')['X'][0,1].T.astype(np.float32)

    def __len__(self):
        return 8677

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(4096)
        x2 = self.V2[idx].reshape(4096)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Reuters(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Reuters.mat')['Y'].astype(np.int32).reshape(1200, )
        self.V1 = scipy.io.loadmat(path + 'Reuters.mat')['X'][0,0].astype(np.float32).toarray()
        self.V2 = scipy.io.loadmat(path + 'Reuters.mat')['X'][0,1].astype(np.float32).toarray()
        self.V3 = scipy.io.loadmat(path + 'Reuters.mat')['X'][0,2].astype(np.float32).toarray()
        self.V4 = scipy.io.loadmat(path + 'Reuters.mat')['X'][0,3].astype(np.float32).toarray()
        self.V5 = scipy.io.loadmat(path + 'Reuters.mat')['X'][0,4].astype(np.float32).toarray()

    def __len__(self):
        return 1200

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(2000)
        x2 = self.V2[idx].reshape(2000)
        x3 = self.V3[idx].reshape(2000)
        x4 = self.V4[idx].reshape(2000)
        x5 = self.V5[idx].reshape(2000)
        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3),torch.from_numpy(x4),torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Scene_15(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Scene_15.mat')['Y'].astype(np.int32).reshape(4485, )
        self.V1 = scipy.io.loadmat(path + 'Scene_15.mat')['X'][0,0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Scene_15.mat')['X'][0,1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Scene_15.mat')['X'][0,2].astype(np.float32)

    def __len__(self):
        return 4485

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(20)
        x2 = self.V2[idx].reshape(59)
        x3 = self.V3[idx].reshape(40)
        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class BDGP(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'BDGP.mat')['Y'].astype(np.int32).reshape(2500, )
        self.V1 = scipy.io.loadmat(path + 'BDGP.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'BDGP.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 2500

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(1750)
        x2 = self.V2[idx].reshape(79)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class RGB(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'RGB-D.mat')['Y'].astype(np.int32).reshape(1449, )
        self.V1 = scipy.io.loadmat(path + 'RGB-D.mat')['X'][0,0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'RGB-D.mat')['X'][0,1].astype(np.float32)

    def __len__(self):
        return 1449

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(2048)
        x2 = self.V2[idx].reshape(300)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MNIST(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST.mat')['Y'].astype(np.int32).reshape(10000, )
        self.V1 = scipy.io.loadmat(path + 'MNIST.mat')['X'][0,0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST.mat')['X'][0,1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'MNIST.mat')['X'][0,2].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(30)
        x2 = self.V2[idx].reshape(9)
        x3 = self.V3[idx].reshape(30)
        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Digit4k(Dataset):
    def __init__(self, path):
        self.Y = hdf5storage.loadmat(path + 'Digit4k.mat')['Y'].astype(np.int32).reshape(4000,)
        self.V1 = hdf5storage.loadmat(path + 'Digit4k.mat')['X'][0,0].astype(np.float32)
        self.V2 = hdf5storage.loadmat(path + 'Digit4k.mat')['X'][0,1].astype(np.float32)

    def __len__(self):
        return 4000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(30)
        x2 = self.V2[idx].reshape(30)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class NUS(Dataset):
    def __init__(self, path):
        self.Y = hdf5storage.loadmat(path + 'NUS.mat')['Y'].astype(np.int32).reshape(2400,)
        self.V1 = hdf5storage.loadmat(path + 'NUS.mat')['X'][0,0].astype(np.float32)
        self.V2 = hdf5storage.loadmat(path + 'NUS.mat')['X'][0,1].astype(np.float32)
        self.V3 = hdf5storage.loadmat(path + 'NUS.mat')['X'][0,2].astype(np.float32)
        self.V4 = hdf5storage.loadmat(path + 'NUS.mat')['X'][0,3].astype(np.float32)
        self.V5 = hdf5storage.loadmat(path + 'NUS.mat')['X'][0,4].astype(np.float32)
        self.V6 = hdf5storage.loadmat(path + 'NUS.mat')['X'][0,5].astype(np.float32)

    def __len__(self):
        return 2400

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(64)
        x2 = self.V2[idx].reshape(144)
        x3 = self.V3[idx].reshape(73)
        x4 = self.V4[idx].reshape(128)
        x5 = self.V5[idx].reshape(225)
        x6 = self.V6[idx].reshape(500)
        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3), torch.from_numpy(x4),torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class NUSWIDEOBJ(Dataset):
    def __init__(self, path):
        self.Y = hdf5storage.loadmat(path + 'NUSWIDEOBJ.mat')['Y'].astype(np.int32).reshape(30000,)
        self.V1 = hdf5storage.loadmat(path + 'NUSWIDEOBJ.mat')['X'][0,0].astype(np.float32)
        self.V2 = hdf5storage.loadmat(path + 'NUSWIDEOBJ.mat')['X'][0,1].astype(np.float32)
        self.V3 = hdf5storage.loadmat(path + 'NUSWIDEOBJ.mat')['X'][0,2].astype(np.float32)
        self.V4 = hdf5storage.loadmat(path + 'NUSWIDEOBJ.mat')['X'][0,3].astype(np.float32)
        self.V5 = hdf5storage.loadmat(path + 'NUSWIDEOBJ.mat')['X'][0,4].astype(np.float32)

    def __len__(self):
        return 30000
    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(65)
        x2 = self.V2[idx].reshape(226)
        x3 = self.V3[idx].reshape(145)
        x4 = self.V4[idx].reshape(74)
        x5 = self.V5[idx].reshape(129)
        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class cifar10(Dataset):
    def __init__(self, path):
        self.Y = hdf5storage.loadmat(path + 'cifar10.mat')['Y'].astype(np.int32).reshape(50000,)
        self.V1 = hdf5storage.loadmat(path + 'cifar10.mat')['X'][0,0].astype(np.float32)
        self.V2 = hdf5storage.loadmat(path + 'cifar10.mat')['X'][0,1].astype(np.float32)
        self.V3 = hdf5storage.loadmat(path + 'cifar10.mat')['X'][0,2].astype(np.float32)

    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(512)
        x2 = self.V2[idx].reshape(2048)
        x3 = self.V3[idx].reshape(1024)
        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MNIST_USPS(Dataset):
    def __init__(self, path):
        X1 = hdf5storage.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        X2 = hdf5storage.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)
        self.Y = hdf5storage.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = X1.reshape(X1.shape[0], -1)
        self.V2 = X2.reshape(X2.shape[0], -1)

    def __len__(self):
        return 5000
    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "BBCSport":
        dataset = BBCSport('./data/')
        dims = [3183, 3203]
        view = 2
        data_size = 544
        class_num = 5
    elif dataset == "BBC":
        dataset = BBC('./data/')
        dims = [4659, 4633, 4665, 4684]
        view = 4
        data_size = 685
        class_num = 5
    elif dataset == "caltech101":
        dataset = caltech101('./data/')
        dims = [4096, 4096]
        view = 2
        data_size = 8677
        class_num = 101
    elif dataset == "Reuters":
        dataset = Reuters('./data/')
        dims = [2000,2000,2000,2000,2000]
        view = 5
        data_size = 1200
        class_num = 6
    elif dataset == "Scene_15":
        dataset = Scene_15('./data/')
        dims = [20,59,40]
        view = 3
        data_size = 4485
        class_num = 15
    elif dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750,79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "RGB-D":
        dataset = RGB('./data/')
        dims = [2048,300]
        view = 2
        data_size = 1449
        class_num = 13
    elif dataset == "MNIST":
        dataset = MNIST('./data/')
        dims = [30, 9, 30]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Digit4k":
        dataset = Digit4k('./data/')
        dims = [30, 30]
        view = 2
        data_size = 4000
        class_num = 4
    elif dataset == "NUS":
        dataset = NUS('./data/')
        dims = [64, 144, 73, 128, 225, 500]
        view = 6
        data_size = 2400
        class_num = 12
    elif dataset == "NUSWIDEOBJ":
        dataset = NUSWIDEOBJ('./data/')
        dims = [65, 226, 145, 74, 129]
        view = 5
        data_size = 30000
        class_num = 31
    elif dataset == "cifar10":
        dataset = cifar10('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
    elif dataset == "MNIST_USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        data_size = 5000
        class_num = 10
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

