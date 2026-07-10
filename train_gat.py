from model import BotGAT, BotGCN, BotRGCN
from Dataset import Twibot22
import torch
from torch import nn
from utils import accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData

import pandas as pd
import sys
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
set_seed(seed)

device = 'cuda:0'
embedding_size,dropout,lr,weight_decay=64,0.1,1e-2,5e-2


root='./processed_data/'

dataset=Twibot22(root=root,device=device,process=False,save=False)
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()


model = BotGAT(cat_prop_size=3, num_prop_size=5, embedding_dimension=64).to(device)
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)


def train(epoch):
    model.train()
    output, _ = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),)
    return acc_train, acc_val, loss_train

def test():
    model.eval()
    output, _ = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_test = loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    output=output.max(1)[1].to('cpu').detach().numpy()
    label=labels.to('cpu').detach().numpy()
    f1=f1_score(label[test_idx],output[test_idx])
    #mcc=matthews_corrcoef(label[test_idx], output[test_idx])
    precision=precision_score(label[test_idx],output[test_idx])
    recall=recall_score(label[test_idx],output[test_idx])
    fpr, tpr, thresholds = roc_curve(label[test_idx], output[test_idx], pos_label=1)
    Auc=auc(fpr, tpr)
    print("Test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            "precision= {:.4f}".format(precision),
            "recall= {:.4f}".format(recall),
            "f1_score= {:.4f}".format(f1),
            #"mcc= {:.4f}".format(mcc),
            "auc= {:.4f}".format(Auc),
            )
    
model.apply(init_weights)

best_val_acc = -1.0
best_model_path = f'./processed_data/botgat_best_{seed}.pt'

epochs=800
for epoch in range(epochs):
    _, acc_val, _ = train(epoch)
    current_val = acc_val.item()
    if current_val > best_val_acc:
        best_val_acc = current_val
        torch.save(model.state_dict(), best_model_path)
        print('Saved best model at epoch {:04d} with acc_val: {:.4f}'.format(epoch + 1, best_val_acc))

print('Best validation accuracy: {:.4f}'.format(best_val_acc))
print('Best model saved to: {}'.format(best_model_path))

model.load_state_dict(torch.load(best_model_path))
test()