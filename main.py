import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.models as models
import matplotlib.pyplot as plt

from args import args
from mri_datasets import get_dataloader

import warnings
warnings.filterwarnings('ignore')


class Encoder3D(nn.Module):
    def __init__(self, projection_dim):
        super(Encoder3D, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=7, stride=2, padding=3),
            # nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=1),

            nn.Conv3d(8, 8, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(8, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=1),

            nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=1, padding=1),
            # nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,3,1), stride=1),

            nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=1, padding=1),
            # nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,3,1), stride=1),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(97344, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x):
        # print('here 1', x.shape, x.device)  # here 1 torch.Size([10, 1, 156, 156, 64])
        x = self.conv_layers(x)
        # print('here 2', x.shape, x.device)  # here 2 torch.Size([10, 64, 13, 13, 9])
        x = torch.flatten(x, 1)
        # print('here 3', x.shape, x.device)  # here 3 torch.Size([10, 97344])
        x = self.fc_layers(x)
        # print('here 4', x.shape, x.device)  # here 4 torch.Size([10, 256])
        return x
    
class Encoder3D_fourlier(nn.Module):
    def __init__(self, projection_dim):
        super(Encoder3D_fourlier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=7, stride=2, padding=3),
            # nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=1),

            nn.Conv3d(16, 16, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(16, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=1),

            nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=1, padding=1),
            # nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,3,1), stride=1),

            nn.Conv3d(64, 64, kernel_size=(3,3,1), stride=1, padding=1),
            # nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(3,3,1), stride=1),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(97344, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x):
        # print('here 1', x.shape, x.device)  # here 1 torch.Size([10, 1, 156, 156, 64])
        x = self.conv_layers(x)
        # print('here 2', x.shape, x.device)  # here 2 torch.Size([10, 64, 13, 13, 9])
        x = torch.flatten(x, 1)
        # print('here 3', x.shape, x.device)  # here 3 torch.Size([10, 97344])
        x = self.fc_layers(x)
        # print('here 4', x.shape, x.device)  # here 4 torch.Size([10, 256])
        return x
    

class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(encoder.fc_layers[-1].out_features, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        return x


def validate(model, valid_dl, local_rank, temperature):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for x1, x2 in valid_dl:
            x1 = x1.cuda(local_rank)  #  torch.Size([batch size, 156, 156, 64])
            x2 = x2.cuda(local_rank)
            z1 = model(x1)  #  torch.Size([batch size , projection dim])
            z2 = model(x2)
            loss = nt_xent_loss(z1, z2, temperature)
            val_loss.append(loss.item())
    return np.mean(val_loss)


def train(model, train_dl, valid_dl, optimizer, local_rank, num_epochs, temperature):
    loss_lst = []
    val_loss_lst = []
    model.train()
    with torch.cuda.device(local_rank):
        for epoch in tqdm(range(num_epochs)):
            loss_epoch = []
            for x1, x2 in train_dl:
                x1 = x1.cuda(local_rank)  #  torch.Size([batch size, 156, 156, 64])
                x2 = x2.cuda(local_rank)
                z1 = model(x1)  #  torch.Size([batch size , projection dim])
                z2 = model(x2)
                loss = nt_xent_loss(z1, z2, temperature)
                loss_epoch.append(loss.item())
                optimizer.zero_grad()
                # loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
            loss_lst.append(np.mean(loss_epoch))

            val_loss = validate(model, valid_dl, local_rank, temperature)
            val_loss_lst.append(val_loss)
    return loss_lst, val_loss_lst


def nt_xent_loss(z1, z2, tempature):
    """
    compute the contrastive loss between two arrays

    :param z1: the first input array, with torch.Size([batch_size , projection_dim]) 
    :param z2: the second input array, with torch.Size([batch_size , projection_dim]) 

    return loss
    """
    # Normalize the representations.
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)  

    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t())
    
    # Exponentiate the similarity matrix and mask out the self-similarity
    exp_sim_matrix = torch.exp(sim_matrix / tempature)
    mask = torch.eye(2 * z1.size(0), device=z.device).bool()
    exp_sim_matrix = exp_sim_matrix.masked_fill(mask, 0)

    # # Extract the positive pairs and scale by batch size
    pos_sim_1 = torch.diag(exp_sim_matrix, z1.size(0))
    pos_sim_2 = torch.diag(exp_sim_matrix, -z1.size(0))
    pos_sim = torch.cat([pos_sim_1, pos_sim_2], dim=0)

    # Compute the NT-Xent loss for each example
    sum_exp_sim = torch.sum(exp_sim_matrix, dim=1)
    loss = -torch.log(pos_sim / sum_exp_sim)

    return loss.mean()


def main(local_rank, args, results):
    torch.distributed.init_process_group(backend="nccl", rank=local_rank, world_size=args.ngpus_per_proc * args.nprocs)
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    batch_size = int(args.batch_size / args.nprocs)
    train_dl = get_dataloader(mode=args.mode, batch_size=batch_size, seed=args.seed, train=True)
    valid_dl = get_dataloader(mode=args.mode, batch_size=batch_size, seed=args.seed, train=False)
    encoder = Encoder3D(projection_dim=args.projection_dim) if args.mode == 'augmentation' else Encoder3D_fourlier(projection_dim=args.projection_dim)
    model = SimCLR(encoder=encoder, projection_dim=args.projection_dim)
    model = model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_lst, val_loss_lst = train(model, train_dl, valid_dl, optimizer, local_rank, num_epochs=args.epochs, temperature=args.temperature)

    # Append results to the shared-memory list
    results.append((loss_lst, val_loss_lst))

    # save model
    if local_rank == 0:
        # torch.save(model.module.state_dict(), "./models/model_" + str(args.mode) + "_b" + str(args.batch_size) + "_p" + str(args.projection_dim) + "_e" + str(args.epochs) + ".pth")
        torch.save(model.module, "./models/model_" + str(args.mode) + "_b" + str(args.batch_size) + "_p" + str(args.projection_dim) + "_e" + str(args.epochs) +  "_lr" + str(args.learning_rate) + ".pt")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # output setting information to confirm
    print("Setting Information to confirm:")
    print("\t mode =", args.mode)
    print("\t batch size =", args.batch_size)
    print("\t projection dim =", args.projection_dim)
    print("\t epochs =", args.epochs)
    print("\t learning_rate =", args.learning_rate)
    print("\t temperature =", args.temperature)
    # print("\t ngpus_per_proc =", args.ngpus_per_proc)
    # print("\t nprocs =", args.nprocs)
    print("*"*30)

    # Create a shared-memory list
    manager = mp.Manager()
    results = manager.list()
    # Spawn processes
    mp.spawn(fn=main, nprocs=args.nprocs, args=(args, results))
    results = list(results)

    loss_train, loss_val = [], []
    for idx, (loss_lst, val_loss_lst) in enumerate(results):
        # print(f"Results from process {idx} - Loss List: {loss_lst}, Validation Loss List: {val_loss_lst}")
        loss_train.append(loss_lst)
        loss_val.append(val_loss_lst)
    
    plt.plot(np.array(loss_train).mean(axis=0), label='train loss')
    plt.plot(np.array(loss_val).mean(axis=0), label='val loss')
    plt.legend()
    plt.savefig("loss_fig/loss_" + str(args.mode) + "_b" + str(args.batch_size) + "_p" + str(args.projection_dim) + "_e" + str(args.epochs) +  "_lr" + str(args.learning_rate) + ".png")
