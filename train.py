import argparse
from dataset_loader import DataLoader
from utils import random_planetoid_splits, get_adj2, set_seed
from model import SHGCN
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time
import pandas as pd



def RunExp(args,dataset, data, Net, percls_trn, val_lb,seed):

    def train(model, optimizer, data, hat_A, hat_A2):
        model.train()
        optimizer.zero_grad()
        out = model(data, hat_A, hat_A2)
        nll = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss = nll
        loss.backward()
        optimizer.step()
        del out

    def test(model, data, hat_A, hat_A2):
        model.eval()
        logits = model(data, hat_A, hat_A2)
        accs, losses, preds = [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses


    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    #percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    #val_lb = int(round(args.val_rate * len(data.y)))

    # randomly split dataset
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb, seed)
    hat_A, hat_A2 = get_adj2(data)
    hat_A = hat_A.to(device)
    hat_A2 = hat_A2.to(device)


    tmp_net = Net(dataset, args)
    model, data = tmp_net.to(device), data.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        train(model, optimizer, data,  hat_A, hat_A2)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data, hat_A, hat_A2)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc


        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:',epoch)
                    #df = pd.DataFrame(att1)
                    #df.to_csv('./weight/cornell/{}/att1_pre.csv'.format(RP), index=False, header=False)
                    #df2 = pd.DataFrame(att2)
                    #df2.to_csv('./weight/cornell/{}/att2_pre.csv'.format(RP), index=False, header=False)

                    break
    return test_acc, best_val_acc,  time_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.07, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dim', type=int, default=16, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.6, help='dropout for neural networks.')

    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')

    parser.add_argument('--dataset', type=str, choices=['Cora','Citeseer','Pubmed','Computers','Photo','Chameleon','Squirrel','Actor','Texas','Cornell'],
                        default='Texas')
    parser.add_argument('--device', type=int, default=3, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['SHGCN'], default='SHGCN')

    args = parser.parse_args()
    #set_seed(args.seed)
    #10 fixed seeds for splits
    SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]

    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    Net = SHGCN

    dataset = DataLoader(args.dataset)
    data = dataset[0]

    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(val_rate * len(data.y)))
    TrueLBrate = (percls_trn * dataset.num_classes + val_lb) / len(data.y)
    print('True Label rate: ', TrueLBrate)

    results = []
    time_results=[]
    for RP in tqdm(range(args.runs)):
        args.seed=SEEDS[RP]
        set_seed(args.seed)
        test_acc, best_val_acc, time_run = RunExp(args,dataset, data, Net, percls_trn, val_lb,args.seed)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc])
        print(f'run_{str(RP+1)} \t test_acc: {test_acc:.4f}')


    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
    print(f'test acc mean = {test_acc_mean:.4f} ± {test_acc_std * 100:.4f}')

    values=np.asarray(results)[:,0]
    uncertainty=np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95)-values.mean()))

    #print(uncertainty*100)
    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty*100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
