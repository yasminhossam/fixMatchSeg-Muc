import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Any, List, Tuple

def train_segmenter(model: torch.nn.Module,
                    train_dataloader_l: DataLoader,
                    train_dataloader_u: DataLoader,
                    val_dataloader_l: DataLoader,
                    warmup: int = 2,
                    patience: int = 5,
                    max_epochs: int = 100) -> None:
    """Train the segmentation model

    Parameters
    ----------
    model
        The segmentation model to be trained
    train_dataloader_l:
        An iterator which returns batches of training images and masks from the
        labeled training dataset
    train_dataloader_u:
        An iterator which returns batches of weakly and strongly augmented training images from 
        the unlabeled training dataset
    val_dataloader_l:
        An iterator which returns batches of training images and masks from the
        labeled validation dataset
    warmup: int, default: 2
        The number of epochs for which only the upsampling layers (not trained by the classifier)
        should be trained
    patience: int, default: 5
        The number of epochs to keep training without an improvement in performance on the
        validation set before early stopping
    max_epochs: int, default: 100
        The maximum number of epochs to train for
    """
    best_state_dict = model.state_dict()
    best_loss = 1
    patience_counter = 0
    for i in range(max_epochs):
        if i <= warmup:
            # we start by 'warming up' the final layers of the model
            optimizer = torch.optim.Adam([pam for name, pam in
                                          model.named_parameters() if 'pretrained' not in name])
        else:
            optimizer = torch.optim.Adam(model.parameters())

        train_data, val_data = _train_segmenter_epoch(model, optimizer, train_dataloader_l,
                                                    train_dataloader_u, val_dataloader_l)
        if np.mean(val_data) < best_loss:
            best_loss = np.mean(val_data)
            patience_counter = 0
            best_state_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter == patience:
                print("Early stopping!")
                model.load_state_dict(best_state_dict)
                return None


def _train_segmenter_epoch(model: torch.nn.Module,
                           optimizer: Optimizer,
                           train_dataloader_l: DataLoader,
                           train_dataloader_u: DataLoader,
                           val_dataloader_l: DataLoader,
                           ) -> Tuple[List[Any], List[Any]]:
    
    t_losses, v_losses = [], []
    labeled_iter = iter(train_dataloader_l)
    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()

    #inputs_u_w and inputs_u_s are weakly and strongly augmented unlabeled images respectively
    for (inputs_u_w, inputs_u_s), _ in train_dataloader_u:
        try:
            inputs_x, targets_x = labeled_iter.next()
        except StopIteration:
            labeled_iter = iter(train_dataloader_l)
            inputs_x, targets_x = labeled_iter.next()

        optimizer.zero_grad()
        inputs_x, inputs_u_w, inputs_u_s = inputs_x.float().to(device), inputs_u_w.to(device), inputs_u_s.to(device)
        inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).float().to(device)
        targets_x = targets_x.to(device)
        preds = model(inputs)
        lim = targets_x.shape[0]
        preds_x = preds[:lim] 
        preds_u_w, preds_u_s = preds[lim:].chunk(2)

        BCE_s = F.binary_cross_entropy(preds_x, targets_x.unsqueeze(1))
        loss_s = BCE_s
        
        targets_u = preds_u_w.squeeze(1).clone().detach()
        preds_label = preds_u_w.squeeze(1).clone().detach()
        preds_label[preds_label>0.5]=1
        preds_label[preds_label<=0.5]=0

        bg = (targets_u<=0.5)*targets_u #background
        fg = (targets_u>0.5)*targets_u  #foreground
        pixel_mean_bg = torch.true_divide(bg.sum((1,2)),(bg!=0).sum((1,2)))
        pixel_mean_fg = torch.true_divide(fg.sum((1,2)),(fg!=0).sum((1,2)))
        mask = (pixel_mean_fg>=0.9)|(pixel_mean_bg<=0.1)
        
        BCE_u = torch.mean(F.binary_cross_entropy(preds_u_s, preds_label.unsqueeze(1))*mask)
        loss_u = BCE_u 
        
        loss = loss_s + loss_u
        loss.backward()
        optimizer.step()
        t_losses.append(loss.item())

    with torch.no_grad():
        model.eval()
        for val_x, val_y in val_dataloader_l:
            val_x, val_y = val_x.float().to(device), val_y.to(device)
            val_preds = model(val_x)
            val_BCE = F.binary_cross_entropy(val_preds, val_y.unsqueeze(1))
            val_loss = val_BCE 
            v_losses.append(val_loss.item())

    
    print(f'Train loss: {np.mean(t_losses)}, Val loss: {np.mean(v_losses)}')

    return t_losses, v_losses
