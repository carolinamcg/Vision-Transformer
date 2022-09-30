from vit_transformer import *
from train_eval import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_Softargmax = nn.Softmax  # fix wrong name

def plot_history(history, n_epochs):
    # summarize history for accuracy
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(0, n_epochs)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":

    #missing import data, data split and data loader (batch dividion and data generators) first

    x = torch.empty((128, 1, 52, 52)).float()
    x = x.to(device)

    model = TransformerClassifier(num_layers=1, d_model=2, num_heads=1, 
                         conv_hidden_dim=4, patch_size=2, num_answers=15, dropout_rate=0.1, 
                         max_pos_emb=5000, in_channels=1, 
                         cnn=False, nchan_l1=8, l1_kw=7, nchan_l2=16, l2_kw=7, 
                         special_token=1, add_pos_emb=True, avgpool=True)#d_k = 16/4 = 4
    model.to(device)

    #y = model(x)
    epochs = 50
    train_eval = Training(model, epochs, "transf_classif_CNN_emb_CLStoken", criterion=F.cross_entropy, base_path='/content/drive/MyDrive/Anansi_00/idtracker/')
    #train_eval.print_parameters()

    train_eval.train(x, warmup_rate=0.3, save_plots=True)
    #plot_history(train_eval.history, epochs)
