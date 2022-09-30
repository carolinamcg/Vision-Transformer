import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import os
import numpy as np

import metrics

plt.style.use('ggplot')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.

    https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion,
        path
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            
            os.makedirs(path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict(),
                'loss': criterion,
                }, path+'best_model.pth')








class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model, factor, training_steps, lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0):
    #training_steps = n_epohs* n_batches
    return NoamOpt(model.encoder.d_model, factor, 0.3*training_steps,
            torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))






# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# gets x and y elements to compute the batxhes
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data
        self.indexes = np.arange(len(self.data))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.indexes[index]

        # Find list of IDs
        X = self.data[ID, ]
        Y = self.labels[ID, ]

        return X, Y








class Training:
    def __init__(self, model, epochs, model_name, criterion=F.cross_entropy, 
                metrics= ["Loss", "Accuracy"], base_path=os.getcwd()):
        self.model = model
        self.metrics = metrics
        self.history = {'train_Loss': [], 'train_Accuracy': []}
        self.optimizer = None
        self.epochs = epochs
        self.criterion = criterion
        self.model_name = model_name
        #self.initialize = initialize
        self.base_path = base_path
        self.saver = SaveBestModel()

    def print_parameters(self):
        # total parameters and trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.\n")

    def get_std_opt(self, training_steps, factor=2, warmup_rate=0.3, lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0):
        #training_steps = n_epohs* n_batches
        return NoamOpt(self.model.encoder.d_model, factor, int(warmup_rate*training_steps),
                torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
    
    def initialise_w(self, model):
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        print("Initialising transformers weights...")
        for p in model.parameters():
            if p.dim() > 1 and p.dim() < 3: #only for linear layers' weights (no biases or conv layers w)
                nn.init.xavier_uniform_(p)

    def save_model(self):
        """
        Function to save the trained model to disk.
        """
        print(f"Saving final model...")
        path = self.base_path + 'weights/{}/'.format(self.model_name)
        os.makedirs(path, exist_ok=True)
        torch.save({
                    'epoch': self.epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
                    'loss': self.criterion,
                    }, path+'final_model.pth')

    def save_plots(self):
        """
        Function to save the loss and accuracy plots to disk.
        """
        path = self.base_path + 'results/plots/{}/'.format(self.model_name)
        os.makedirs(path, exist_ok=True)

        for metric in self.metrics:
            # metric plots
            plt.figure(figsize=(10, 7))
            for k in self.history.keys():
                if metric in k:
                    plt.plot(
                        self.history[k], color='green', linestyle='-', 
                        label=k
                    )

            plt.xlabel('Epochs')
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(path+metric+'.png')
            plt.close()


    def train(self, train_loader, valid_loader=None, checkpoint_metric='val_Loss', 
            initialise=True, factor=2, warmup_rate=0.3, lr=0, betas=(0.9, 0.99), eps=1e-9, 
            weight_decay=0.0, save_plots=True):
        
        if initialise:
            self.initialise_w(self.model)
            print("Done!")
        
        nb_batches_train = len(train_loader)
        if self.optimizer == None:
            training_steps = self.epochs * nb_batches_train
            self.optimizer = self.get_std_opt(training_steps, factor=2, warmup_rate=warmup_rate,
                                        lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        if valid_loader is not None:
            self.history['val_Loss']  = []
            self.history['val_Accuracy']  = []
        
        for epoch in range(self.epochs):
            train_iterator = iter(train_loader)
            train_acc = 0
            self.model.train()
            losses = 0.0

            for batch in train_iterator:
                x = batch[0].float()
                y = batch[1].float()
                x = x.to(device)
                y = y.to(device)
                
                #print("LEARNING RATE: ", optimizer._rate)
                out = self.model(x)  # ①

                loss = self.criterion(out, y)  #Cross_Entrpy = nn.LogSoftmax (last model layer) + F.NNL_loss
                                                #reduction=mean --> computes the mean loss over all samples in the batch
                                                # this is important if you hacve batches with different sizes
                
                self.model.zero_grad()  # ③

                loss.backward()  # ④
                losses += loss.item()

                self.optimizer.step()  # ⑤
                
                #train_acc += (out.argmax(1) == y.argmax(1)).cpu().numpy().mean() #mean over the images in the batch
                train_acc += metrics.classification_accuracy(out, y)
            
            print(f"Training loss at epoch {epoch} is {losses / nb_batches_train}")
            print(f"Training accuracy: {train_acc / nb_batches_train}")

            self.history['train_Loss'].append(losses / nb_batches_train)
            self.history['train_Accuracy'].append(train_acc / nb_batches_train)

            if valid_loader is not None:
                print('Evaluating on validation:')
                self.evaluate(valid_loader)

            self.saver(
                 self.history[checkpoint_metric][-1], epoch, 
                 self.model, self.optimizer, self.criterion,
                 path = self.base_path+'weights/{}/'.format(self.model_name)
            )

        # save the trained model weights for a final time
        self.save_model()
        if save_plots:
            self.save_plots()

    def evaluate(self, data_loader):
        data_iterator = iter(data_loader)
        nb_batches = len(data_loader)
        self.model.eval()
        acc = 0 
        losses = 0
        for batch in data_iterator:
            #self.model.eval()
            x = batch[0].float()
            y = batch[1].float()
            x = x.to(device)
            y = y.to(device)
                    
            out = self.model(x)

            loss = self.criterion(out, y)  # ②
                
            losses += loss.item()

            #acc += (out.argmax(1) == y.argmax(1)).cpu().numpy().mean()
            acc += metrics.classification_accuracy(out, y)

        print(f"Eval loss: {losses / nb_batches}")
        print(f"*************Eval accuracy: {acc / nb_batches}")

        self.history['val_Loss'].append(losses / nb_batches)
        self.history['val_Accuracy'].append(acc / nb_batches)
