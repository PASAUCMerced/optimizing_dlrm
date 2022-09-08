import torch
import numpy as np 


# PyTroch version

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    results = outputs.reshape(-1).detach().numpy()
    #results = np.append(results, results*2) scale things back
    results = np.unique(results)
    results = results.astype(int)
    label = labels.reshape(-1).detach().numpy()
    label = np.unique(labels)

    #for i in results:
    #    results = np.append(results, prefetch_neighbour(i))
    #results = np.unique(results)

    #intersection = (outputs.reshape(-1) & labels.reshape(-1)).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    #intersection = (results & label).sum((1, 2))
    intersection = len(np.intersect1d(results, label))
    #print(intersection)
    #union = (results | label).sum((1, 2))
    union = min(len(label), len(results)) # output size
    iou = 1 - (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    accuracy = (intersection + SMOOTH) / (union + SMOOTH)
    #union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    #iou = (intersection + SMOOTH) / (union + SMOOTH) 
    
    thresholded = torch.tensor(iou, requires_grad=True)
    #print(thresholded)
    return accuracy, thresholded  # Or thresholded.mean() if you are interested in average across the batch
    
    
# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)
    
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    thresholded = 1- np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    
    return thresholded  # Or thresholded.mean()

def prefetch_neighbour(res):
    ret=[]
    for i in range(5):
        ret.append(i+res)
    return ret