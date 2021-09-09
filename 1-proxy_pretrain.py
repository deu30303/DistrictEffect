import os
import torch
import argparse
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import *
from dataloader import *

def arg_parser():
    parser = argparse.ArgumentParser(description='Proxy Pretrain Parser')
    parser.add_argument('--mode', default="ordinal", type=str, help='proxy pretrain mode (ordinal or nightlight)')
    parser.add_argument('--root-dir', default="./data/proxy/", type=str, help='proxy image path')
    parser.add_argument('--train-meta', default="./metadata/proxy_metadata_train.csv", type=str, help='train metadata path')
    parser.add_argument('--test-meta', default="./metadata/proxy_metadata_test.csv", type=str, help='train metadata path')
    parser.add_argument('--thr1', '--threshold1', default=0, type=int, help='rural score threshold')
    parser.add_argument('--thr2', '--threshold2', default=10, type=int, help='city score threshold')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='learning rate')
    parser.add_argument('--batch-size', default=50, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='total epochs') 
    parser.add_argument('--workers', default=4, type=int, help='number of workers') 
    
    return parser.parse_args()    

def main_ordinal(args):
    # Generate DataLoader
    train_proxy = OproxyDataset(metadata = args.train_meta, 
                               root_dir = args.root_dir,
                               transform=transforms.Compose([RandomRotate(),ToTensor(),Grayscale(prob = 0.1),
                                                             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    test_proxy = OproxyDataset(metadata = args.test_meta, 
                              root_dir = args.root_dir,
                              transform=transforms.Compose([ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    
    train_loader = torch.utils.data.DataLoader(train_proxy, batch_size=args.batch_size, shuffle=True, num_workers=4)    
    test_loader = torch.utils.data.DataLoader(test_proxy, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    
    # Generate Model 
    net = models.resnet18(pretrained = True)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    model = BinMultitask(net, feature_size, args.thr1, args.thr2, ordinal=True)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    # Train and Test for Ordinal Regression
    best_acc = 0
    for epoch in range(args.epochs):
        train_ordinal(train_loader, model, optimizer, epoch, args.batch_size)
        if (epoch + 1) % 10 == 0:
            acc = test_ordinal(test_loader, model)
            if acc > best_acc:
                print('state_saving...')
                save_checkpoint({'state_dict': model.state_dict()}, './model', model, 'proxy_ordinal')
                best_acc = acc
                
def train_ordinal(train_loader, model, optimizer, epoch, batch_size):
    model.train()
    count = 0                                                       
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):   
        inputs, targets = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(targets.cuda())
        _, _, logit = model(inputs)
        # Soft Label Cross Entropy Loss
        loss = torch.mean(torch.sum(-targets * torch.log(logit), 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
        
    total_loss /= count
    print('[Epoch: %d] loss: %.5f' % (epoch + 1, total_loss))     
           
def test_ordinal(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    acc = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(targets.cuda())
            _, _, logit = model(inputs)
            _, predicted = torch.max(logit, 1)
            _, answer =  torch.max(targets, 1)
            total += inputs.size(0)
            correct += (predicted == answer).sum().item()
        acc = (correct / total) * 100.0
        print('Test Acc : %.2f' % (acc))
    
    return acc

def main_nl(args):
    # Generate DataLoader 
    train_proxy = NproxyDataset(metadata = args.train_meta, 
                               root_dir = args.root_dir,
                               transform=transforms.Compose([RandomRotate(),ToTensor(),Grayscale(prob = 0.1),
                                                             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))    
    train_loader = torch.utils.data.DataLoader(train_proxy, batch_size=args.batch_size, shuffle=True, num_workers=4)       
    
    # Generate Model 
    net = models.resnet18(pretrained = True)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    model = BinMultitask(net, feature_size, args.thr1, args.thr2, ordinal=False)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    # Train and Test for Nightlight Proxy
    best_acc = 0
    for epoch in range(args.epochs):
        train_nl(train_loader, model, optimizer, epoch, args.batch_size)
        
    print('state_saving...')
    save_checkpoint({'state_dict': model.state_dict()}, './model', model, 'proxy_nl')
            
def train_nl(train_loader, model, optimizer, epoch, batch_size):
    model.train()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(targets.cuda().float())
        _, scores, _ = model(inputs)
        scores = scores.squeeze().float()
        i_output = scores - torch.mean(scores)
        t_output = targets - torch.mean(targets)
        # Pearson Maximization Loss
        numerator = torch.sum(i_output * t_output)
        denominator = (torch.sqrt(torch.sum(i_output ** 2) + 1e-3) * torch.sqrt(torch.sum(t_output ** 2)) + 1e-3)
        loss = - numerator / denominator  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('[Epoch: %d, Batch: %d] loss: %.3f' % (epoch + 1, batch_idx, loss))
    
def save_checkpoint(state, dirpath, model, arch_name):
    filename = '{}.ckpt'.format(arch_name)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)
        
if __name__ == '__main__':
    args = arg_parser()
    if args.mode == 'ordinal':
        main_ordinal(args)
    elif args.mode == 'nightlight':
        main_nl(args)