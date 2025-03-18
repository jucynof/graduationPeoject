import numpy as np
import torch
from torch import softmax

def Evaluate1(acc,loss,w):
    scores=np.zeros(len(acc))
    maxLoss=np.max(loss)
    maxAcc=np.max(acc)
    for i in range(len(acc)):
        scores[i]=(w*acc[i]/maxAcc)+((1-w)*(1-loss[i]/maxLoss))
    scores=softmax(torch.Tensor(scores),dim=0)
    return scores.cpu().numpy()
