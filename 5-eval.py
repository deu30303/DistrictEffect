import csv
import copy
import random 
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

def arg_parser():
    parser = argparse.ArgumentParser(description='Eval Parser')
    parser.add_argument('--metadata', default='./metadata/kr_entire_demographics.csv', type=str, help='metadata path')
    parser.add_argument('--item', default='TOTPOP_CY', type=str, help='Economic indicator item')
    parser.add_argument('--ensemble-list', default='./district_summary/ensemble_list.txt', type=str, help='embedding list txt file path')
    parser.add_argument('--score-path', default='./local_score/local_score.csv', type=str, help='local score path')
    parser.add_argument('--train-ratio', default=0.8, type=float, help='train ratio')
    parser.add_argument('--train-count', default=100, type=int, help='train count')
    
    return parser.parse_args()

def main(args):
    district_info = make_district_dict(args)
    x, y, d_num, d_list = train_preprocess(args, district_info)
    r2_list = []
    
    for i in range(0, args.train_count):
        random.seed(i)
        train_district = random.sample(range(len(x)), int(len(x) * (args.train_ratio)))
        train_district.sort()
        test_district = []
        
        for i in range(0, len(x)):
            if i not in train_district:
                test_district.append(i)

        train_dnum = d_num[train_district]
        train_x = x[train_district]
        train_y = y[train_district]
        test_x = x[test_district]
        test_y = y[test_district]
        test_district_id = d_list[test_district]


        tx_shape = train_x.shape
        ty_shape = train_y.shape

        # District Augmentation
        for i in range(0, 10):
            rand_idx = [i for i in range(0, train_dnum.shape[0])]
            random.shuffle(rand_idx)
            train_x_t = train_x[rand_idx]
            train_y_t = train_y[rand_idx]
            train_dnum_t = train_dnum[rand_idx]
            
            train_x_m = np.zeros(tx_shape)
            train_y_m = np.zeros(ty_shape)

            for i in range(0, train_dnum.shape[0]):
                b1 =  train_dnum[i] /(train_dnum[i] + train_dnum_t[i])
                b2 =  train_dnum_t[i] /(train_dnum[i] + train_dnum_t[i])
                train_x_m[i] = b1*train_x[i] + b2*train_x_t[i]
                train_y_m[i] = np.log(b1*np.exp(train_y[i]) + b2*np.exp(train_y_t[i]))
            train_x = np.concatenate((train_x, train_x_m), axis=0)
            train_y = np.concatenate((train_y, train_y_m), axis=0)

        reg = RandomForestRegressor(max_depth=100, n_estimators = 200)
        reg.fit(train_x, train_y)
        predict = reg.predict(test_x)

        predict_alpha = {}
        for i in range(0, len(test_district_id)):
            predict_alpha[test_district_id[i]] = predict[i]

        score_list = {} 
        actual_score_list = {}
        for district_id in test_district_id:
            score_list[district_id] =  district_info[district_id]['score']  * np.exp(predict_alpha[district_id])
            actual_score_list[district_id] = district_info[district_id]['gt']

        score_result = np.array(list(score_list.values()))
        actual_score_list = np.array(list(actual_score_list.values()))
        r2 = r2_score(actual_score_list, score_result)
        print("R2 Score: {}".format(r2))
        r2_list.append(r2)
    
    # Remove Outlier
    r2_list_cp = copy.deepcopy(r2_list)
    r2_list_cp.sort()
    r2_list_cp = r2_list_cp[int(0.1*len(r2_list_cp)):]
    print("R2 mean : {}, R2 std {}".format(np.mean(r2_list_cp), np.std(r2_list_cp)))

    
def make_district_dict(args):
    # make ground truth dict
    pd_gt = pd.read_csv(args.metadata)
    gt_dict = {}
    for i in range(0, len(pd_gt)):
        directory_id = pd_gt['Directory'][i]
        gt_dict[directory_id] = pd_gt[args.item][i]
    
    # make district dict
    pd_score = pd.read_csv(args.score_path)
    district_info = {}
    d_info = {'gt' : 0, 'scale' : 0, 'score' : 0, 'num' : 0}
    
    for i in range(0, len(pd_gt)):
        directory_id = int(pd_gt['Directory'][i])
        d_temp = copy.deepcopy(d_info)
        district_info[directory_id] = d_temp
    
    for i in range(0, len(pd_score)):
        img_name = pd_score.iloc[i]['image']
        directory_id = int(pd_score.iloc[i]['directory'])
        district_info[directory_id]['gt'] = gt_dict[directory_id]
        district_info[directory_id]['score'] += min(float(pd_score.iloc[i]['score']), 20)
        district_info[directory_id]['num'] += 1
        
    for i in range(0, len(pd_gt)):
        directory = int(pd_gt['Directory'][i])
        district_info[directory]['scale'] = np.log(district_info[directory]['gt']  / district_info[directory]['score']) 
        
    return district_info

def train_preprocess(args, district_info):
    d_sum = []
    d_scale = []
    d_num = []
    
    d_list = np.array(list(district_info.keys()))
    d_info_list = list(district_info.values())
    for i in range(0, len(d_info_list)):
        d_sum.append(d_info_list[i]['score'])
        d_scale.append(d_info_list[i]['scale'])
        d_num.append(d_info_list[i]['num'])

    d_score_sum = np.array(d_sum).reshape(-1,1)
    d_num = np.array(d_num).reshape(-1,1)
    
    ensemble_list = []
    with open(args.ensemble_list) as file:
        ensemble_list = file.readlines()
        ensemble_list = list(map(lambda s: s.strip(), ensemble_list))
    
    X = []
    for d_summary in ensemble_list:
        df_x = pd.read_csv("./district_summary/{}".format(d_summary))
        x = df_x.values.tolist() 
        x = np.array(sorted(x, key=lambda x_entry: x_entry[0]))
        x = np.concatenate((x, d_score_sum), axis = 1)
        x = x[:,[1,2,3,4,5,6,-1]]
        X.append(x)
    X = np.concatenate((X), axis = 1)
    Y = np.array(d_scale)
    
    return X, Y, d_num, d_list

if __name__ == '__main__':
    args = arg_parser()
    main(args)