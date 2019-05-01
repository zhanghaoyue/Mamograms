import copy
from skorch.callbacks import Callback
from lib.Utils import calculate_loss
import numpy as np


class EpochLossCallback(Callback):
    """
    A callback that calculate the loss (likely test) based on the given dataloader, criterion.
    This callback will get the loss and save into the net.history. It will also determine the 
    in that particular epoch, the loss is the best (compared to previous epochs).
    """
    def __init__(self, loader, criterion, lower_is_better=True, mode='test',apply_log=True): 
        """
        :params loader: dataloader
        :params criterion: what pytorch criterion to be used to calculate the loss
        :params lower_is_better: for the loss, is that lower is better (probably yes)
        :params mode: what loss is that (probably 'test')
        :params apply_log: apply torch.log() or not (probably yes)
        """

        self.loader = loader[mode]
        self.criterion = criterion()
        self.lower_is_better = lower_is_better
        if mode =='train':
            self.mode = mode 
        else:
            self.mode = 'valid'
            
    def initialize(self):
        self.best_score_ = np.inf if self.lower_is_better else -np.inf
        return self
    
    def on_epoch_end(self, net, **kwargs):
        # calculate the loss and probs
        current_score, preds = calculate_loss(net, self.loader,self.criterion)
        
        # record the info into the network history
        net.history.record('%s_loss' % self.mode, current_score)
        net.history.record('%s_preds' % self.mode, preds)
        
        # determine if this is the best score or not
        is_best = self._is_best_score(current_score)
        if is_best is None:
            return
        
        # record the if it is the best and update
        net.history.record('%s_loss_best' % self.mode, is_best)
        if is_best:
            self.best_score_ = current_score
            
    def _is_best_score(self, current_score):
        if self.lower_is_better is None:
            return None
        if self.lower_is_better:
            return current_score < self.best_score_
        return current_score > self.best_score_
        





