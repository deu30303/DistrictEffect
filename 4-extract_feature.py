import os
import csv
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from model import *
from dataloader import *


def arg_parser():
    parser = argparse.ArgumentParser(description='Feature Extractor Parser')
    parser.add_argument('--mode', default="embed", type=str, help='feature extraction mode (score or embed)')
    parser.add_argument('--m-path', default="./model/proxy_ordinal.ckpt", type=str, help='extraction model path')
    parser.add_argument('--metadata', default='./metadata/kr_entire_demographics.csv', type=str, help='metadata path')
    parser.add_argument('--root', default='./data/pruned/', type=str, help='image path')
    parser.add_argument('--thr1', '--threshold1', default=0, type=int, help='rural score threshold')
    parser.add_argument('--thr2', '--threshold2', default=10, type=int, help='city score threshold')
    
    return parser.parse_args()

def main(args):
    net = models.resnet18(pretrained = False)
    feature_size = net.fc.in_features
    net.fc = nn.Sequential()
    model = BinMultitask(net, feature_size, args.thr1, args.thr2, ordinal=False)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.load_state_dict(torch.load(args.m_path)['state_dict'], strict = True)
    model.cuda()
    print("Load finished")
    
    if args.mode == 'score':
        extract_score(args, model)
        print("Score extraction finished")
    elif args.mode == 'embed':
        reduce_district(args, model)
        extract_embed(args)
        print("Embed extraction finished")
        
def extract_score(args, model):        
    model.eval()
    
    valid_transform = transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    with open('./local_score/local_score.csv', 'w', newline='') as csvfile:
        wr = csv.writer(csvfile,  delimiter=',')
        wr.writerow(["image", "directory", "score"])
        with torch.no_grad():
            file_list = glob.glob('{}/*/*.png'.format(args.root))
            for file in file_list:
                file_name = file.split('/')[-1].split('.')[0]
                district = file.split('/')[-2]
                image = Image.open(file)
                image = valid_transform(image)
                _, score, _ = model(image.unsqueeze(0).cuda())
                wr.writerow([file_name, district, max(0,score[0].item())])

def reduce_district(args, model):
    model.eval()
    
    with torch.no_grad():
        district_data = DistrictDataset(metadata = args.metadata, 
                                        root_dir = args.root,
                                        transform=transforms.Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
        district_loader = torch.utils.data.DataLoader(district_data, batch_size=1, shuffle=False, num_workers=1)
        print(len(district_data))
        
        for batch in enumerate(district_loader):
            input_images = torch.autograd.Variable(batch[1]['images'][0]).cuda()
            folder_idx = batch[1]['directory'][0]
            print("Reducing Train data - idx : {}".format(folder_idx))
            embed, _, _ = model(input_images)
            np.savetxt("./data/reduced/{}.csv".format(folder_idx), embed.cpu().detach().numpy())

def extract_embed(args):
    train_district = ReducedDataset(metadata = args.metadata, root_dir = './data/reduced/')
    X = []
    num = []
    directory = []
    
    for i in range(len(train_district) +1):
        if train_district[i] == -1:
            continue
        X.append(train_district[i]['images'])
        num.append(train_district[i]['num'])
        directory.append(train_district[i]['directory'])
        
    first = True
    for images in X:
        if first:
            train_for_pca = images
            first = False
        else:
            images = images.reshape(-1,512)
            train_for_pca = np.concatenate([train_for_pca, images])    

    pca = PCA(n_components = 3)
    pca.fit(train_for_pca)
    reduced_X = []
    count = 0
    for images in X:
        images = images.reshape(-1,512)
        train_pca = pca.transform(images)
        train_x = np.append(np.concatenate([np.mean(train_pca, axis = 0), np.std(train_pca, axis = 0)]), num[count])
        reduced_X.append(train_x)    
        count += 1
    reduced_X = np.array(reduced_X)
    new_reduced_X = np.concatenate((np.array(directory).reshape(-1, 1), reduced_X), axis=1)
    df = pd.DataFrame(new_reduced_X, columns=['directory'] + ['embed{}'.format(i) for i in range(7)])
    cp_name = os.path.split(args.m_path)[-1].split('.')[0]
    df.to_csv("./district_summary/{}_summary.csv".format(cp_name), index = False)
    
if __name__ == '__main__':
    args = arg_parser()
    main(args)