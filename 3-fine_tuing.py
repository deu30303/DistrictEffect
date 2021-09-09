import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataloader import *
from model import *
from utils import *

def arg_parser():
    parser = argparse.ArgumentParser(description='Fine tuning Parser')
    parser.add_argument('--mode', default="ordinal", type=str, help='proxy train mode (ordinal or nightlight)')
    parser.add_argument('--m-path', default="./model/proxy_ordinal.ckpt", type=str, help='pretrained model path')
    parser.add_argument('--proxy-meta', default='./metadata/proxy_metadata_train.csv', type=str, help='proxy metadata path')
    parser.add_argument('--cluster-meta', default='./metadata/metadata.csv', type=str, help='cluster metadata path')
    parser.add_argument('--proxy-root', default='./data/proxy/', type=str, help='proxy image path')
    parser.add_argument('--cluster-root', default='./data/pruned/', type=str, help='cluster image path')
    parser.add_argument('--proxy-batch', default=40, type=int, help='proxy batch size')  
    parser.add_argument('--cluster-batch', default=256, type=int, help='cluster batch size')
    parser.add_argument('--thr1', '--threshold1', default=0, type=int, help='rural score threshold')
    parser.add_argument('--thr2', '--threshold2', default=10, type=int, help='city score threshold')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--epochs', default=5, type=int, help='total epochs')
    parser.add_argument('--c-num', default=30, type=int, help='number of clusters') 
    
    return parser.parse_args()


def main_ordinal(args): 
    # dataloader define
    train_transform = transforms.Compose([
                      transforms.Resize(256),
                      transforms.RandomGrayscale(p=0.1),                
                      transforms.RandomResizedCrop(224),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_proxy = OproxyDataset(metadata = args.proxy_meta, root_dir = args.proxy_root, 
                                transform = transforms.Compose([RandomRotate(),ToTensor(),Grayscale(prob = 0.1),
                                                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    proxyloader = torch.utils.data.DataLoader(train_proxy, batch_size=args.proxy_batch, shuffle=True, num_workers=4)
    clusterset = EmbeddingDataset(args.cluster_meta, args.cluster_root, valid_transform)
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=args.cluster_batch, shuffle=False, num_workers=4)
    trainset = EmbeddingDataset(args.cluster_meta, args.cluster_root, train_transform)
    celoader = torch.utils.data.DataLoader(trainset, batch_size=args.cluster_batch, shuffle=True, num_workers=4)
    
     # model define
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = models.resnet18(pretrained = True)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    model = BinMultitask(net, feature_size, args.thr1, args.thr2, ordinal=True)
    model.W.requires_grad = False
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.m_path)['state_dict'], strict = True)
    model.cuda()
    print("Load finished")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    p_label, c_head = extract_plabel(args, clusterloader, model, N = len(clusterset))
    
    for epoch in range(args.epochs):
        train_ordinal(args, proxyloader, celoader, model, p_label, c_head, criterion, optimizer, epoch)
        
    cp_name = '{}_{}'.format('ordinal_finetune', args.c_num)
    save_checkpoint({'state_dict': model.state_dict()}, "./model", arch_name = cp_name) 
  

                
def train_ordinal(args, proxyloader, celoader, model, p_label, c_head, criterion, optimizer, epoch):
    model.train()
    
    proxy_iter = iter(proxyloader)
    for batch_idx, (cluster_image, idexes) in enumerate(celoader):
        try:
            proxy_image, proxy_target = proxy_iter.next()
        except StopIteration:
            proxy_iter = iter(proxyloader)
            proxy_image, proxy_target = proxy_iter.next()
         
        proxy_image = torch.autograd.Variable(proxy_image.cuda())
        proxy_target = torch.autograd.Variable(proxy_target.cuda())
        _, _, logit = model(proxy_image)
        # Soft Label Cross Entropy Loss
        proxy_loss = torch.mean(torch.sum(-proxy_target * torch.log(logit), 1))
        
        cluster_image, idexes = cluster_image.cuda(), idexes.cuda()
        cluster_target = p_label[idexes].squeeze()
        embed, _ , _ = model(cluster_image) 
        logit = c_head(embed)
        cluster_loss = criterion(logit, cluster_target)
        
        loss = proxy_loss + cluster_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print('[Epoch: %d, Batch: %d] loss: %.3f embed : %.3f class : %.3f' % (epoch + 1, batch_idx, loss, cluster_loss, proxy_loss))


def main_nl(args): 
    # dataloader define
    train_transform = transforms.Compose([
                      transforms.Resize(256),
                      transforms.RandomGrayscale(p=0.1),                
                      transforms.RandomResizedCrop(224),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_proxy = NproxyDataset(metadata = args.proxy_meta, root_dir = args.proxy_root, 
                                transform = transforms.Compose([RandomRotate(),ToTensor(),Grayscale(prob = 0.1),
                                                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    proxyloader = torch.utils.data.DataLoader(train_proxy, batch_size=args.proxy_batch, shuffle=True, num_workers=4)
    clusterset = EmbeddingDataset(args.cluster_meta, args.cluster_root, valid_transform)
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=args.cluster_batch, shuffle=False, num_workers=4)
    trainset = EmbeddingDataset(args.cluster_meta, args.cluster_root, train_transform)
    celoader = torch.utils.data.DataLoader(trainset, batch_size=args.cluster_batch, shuffle=True, num_workers=4)
    
     # model define
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = models.resnet18(pretrained = True)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    model = BinMultitask(net, feature_size, args.thr1, args.thr2, ordinal=False)
    model.W.requires_grad = False
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.m_path)['state_dict'], strict = True)
    model.cuda()
    print("Load finished")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    p_label, c_head = extract_plabel(args, clusterloader, model, N = len(clusterset))
    
    for epoch in range(args.epochs):
        train_nl(args, proxyloader, celoader, model, p_label, c_head, criterion, optimizer, epoch)
        
    cp_name = '{}_{}'.format('nl_finetune', args.c_num)
    save_checkpoint({'state_dict': model.state_dict()}, "./model", arch_name = cp_name) 
  

                
def train_nl(args, proxyloader, celoader, model, p_label, c_head, criterion, optimizer, epoch):
    model.train()
    
    proxy_iter = iter(proxyloader)
    for batch_idx, (cluster_image, idexes) in enumerate(celoader):
        try:
            proxy_image, proxy_target = proxy_iter.next()
        except StopIteration:
            proxy_iter = iter(proxyloader)
            proxy_image, proxy_target = proxy_iter.next()

        proxy_image = torch.autograd.Variable(proxy_image.cuda())
        proxy_target = torch.autograd.Variable(proxy_target.cuda().float())
        _, scores, _ = model(proxy_image)
        scores = scores.squeeze().float()
        i_output = scores - torch.mean(scores)
        t_output = proxy_target - torch.mean(proxy_target)
        # Pearson Maximization Loss
        numerator = torch.sum(i_output * t_output)
        denominator = (torch.sqrt(torch.sum(i_output ** 2) + 1e-3) * torch.sqrt(torch.sum(t_output ** 2)) + 1e-3)
        proxy_loss = - numerator / denominator
        
        cluster_image, idexes = cluster_image.cuda(), idexes.cuda()
        cluster_target = p_label[idexes].squeeze()
        embed, _ , _ = model(cluster_image) 
        logit = c_head(embed)
        cluster_loss = criterion(logit, cluster_target)
        
        loss =  proxy_loss + cluster_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print('[Epoch: %d, Batch: %d] loss: %.3f embed : %.3f class : %.3f' % (epoch + 1, batch_idx, loss, cluster_loss, proxy_loss))
            
def extract_plabel(args, clusterloader, model, N):
    features, scores = compute_features_scores(clusterloader, model, N, args.cluster_batch) 
    city_idx = np.where(scores >= args.thr2)[0]
    rural_idx = np.where(scores < args.thr2)[0]
    print("Number of city, rural : {}, {}".format(city_idx.shape[0], rural_idx.shape[0]))
    city_features = features[city_idx]
    rural_features = features[rural_idx]
    kmeans = Kmeans(int(args.c_num / 2))
    _, city_label = kmeans.cluster(city_features)
    _, rural_label = kmeans.cluster(rural_features)
    rural_label = rural_label + int(args.c_num / 2)
    
    # generating pseudo label
    p_label = np.zeros((N, 1), dtype='int64')
    p_label[city_idx] = city_label
    p_label[rural_idx] = rural_label
    p_label_l = p_label.tolist()
    p_label = torch.tensor(p_label_l).cuda()
          
    # generating network head
    c_head = nn.Linear(512, args.c_num)
    c_head.weight.data.normal_(0, 0.01)
    c_head.bias.data.zero_()
    c_head.cuda()
          
    return p_label, c_head
    
            
def save_checkpoint(state, dirpath, arch_name):
    filename = '{}.ckpt'.format(arch_name)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)

    
if __name__ == '__main__':
    args = arg_parser()
    if args.mode == 'ordinal':
        main_ordinal(args)
    elif args.mode == 'nightlight':
        main_nl(args)
          