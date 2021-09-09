import os
import glob
import torch
import argparse
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from model import *
from dataloader import *

def arg_parser():
    parser = argparse.ArgumentParser(description='Data Pruning Parser')
    parser.add_argument('--model', type=str, default='./model/proxy_ordinal.ckpt', help='model path')
    parser.add_argument('--thr1', '--threshold1', default=0, type=int, help='rural score threshold')
    parser.add_argument('--thr2', '--threshold2', default=10, type=int, help='city score threshold')
    parser.add_argument('--path', type=str, default='./data/pruned', help='image path to remove uninhabited areas') 
    return parser.parse_args()

def main(args):
    net = models.resnet18(pretrained = True)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    model = BinMultitask(net, feature_size, args.thr1, args.thr2, ordinal=True)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)    

    model.load_state_dict(torch.load(args.model)['state_dict'], strict=True)    
    model.cuda()
    remove_uninhabited(args, model, args.path)


def remove_uninhabited(args, model, path):
    model.eval()
    transform = transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    dir_list = os.listdir(path)
    dir_list.sort()
    
    total = 0
    for d_num in dir_list:
        count = 0
        file_list = glob.glob('{}/{}/*.png'.format(args.path, d_num))
        for file in file_list:
            image = Image.open(file)
            image = transform(image).unsqueeze(0).cuda()
            _, _, logit = model(image)
            logit = logit.squeeze()
            environment = logit[2]
            if environment >= 0.5:
                os.remove(file)
                count += 1
        total += count
        print("Dir - {} : {} remove".format(d_num, count))
    print("Total : {} remove".format(total))

if __name__ == '__main__':
    args = arg_parser()
    main(args)