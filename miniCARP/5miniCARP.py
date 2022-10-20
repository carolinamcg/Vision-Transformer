import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import (get_layerw, get_weights, partition_data, plot_history,
                       visualize_pe2D, visualize_MAHD)
from model_ViT import *
from train_eval import *

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
nn_Softargmax = nn.Softmax  # fix wrong name

#SETTINGS

image_size = 52
patch_size = 4    
in_channels=1
num_heads = 8
num_layers = 5
NoAmpt = False
# path and directory
base_path='/home/carolina/Anansi-00/ViTransformer/miniCarp/' #path where you want to save your model's weights and results
model_name = "ViT_miniCarp_WeightInit01_150epochs_LR0003_WD0.001_NL5" #dir name to store your model's w and results
# Data SPLIT
VALIDATION_FRAMES = 1000
TEST_FRAMES = 1000
epochs = 150

CARP_PATH = '/home/carolina/Anansi-00/ViTransformer/data/miniCARP/'

def load_carp(CARP_PATH, fraction=1.0):
    images = np.load(os.path.join(CARP_PATH,'miniCARP_images.npy'))
    labels = np.load(os.path.join(CARP_PATH,'miniCARP_labels.npy'))
    num_images = int(np.floor(images.shape[0]*fraction))
    return images[:num_images], labels[:num_images]

def one_hot(target, num_classes = 15):
    '''
    Converts a vector with all possible indexes/classes in one-hot encoded matrix
    (len(ids), num_classes)
    '''
    #print(target)
    return np.squeeze(np.eye(num_classes)[target])


if __name__ == "__main__":

    images_uint8, all_labels = load_carp(CARP_PATH, fraction=0.1)

    print(images_uint8.shape, all_labels.shape)
    print(images_uint8.dtype, all_labels.dtype)
    print(images_uint8.min(), images_uint8.max())
    print(all_labels, all_labels.min(), all_labels.max())
    NUM_LABELS = all_labels.max()+1

    ## Standardize images
    mean = images_uint8.mean()
    std = images_uint8.std()
    print(mean, std)
    all_images = (images_uint8 - images_uint8.mean()) / images_uint8.std()
    print("New range: {} to {}".format(all_images.min(), all_images.max()))

    labels = partition_data(one_hot(all_labels), validation=VALIDATION_FRAMES, test=TEST_FRAMES)
    images = partition_data(all_images)

    print("Validation and test are fractions {:.2g} and {:.2g}".format(images['validation'].shape[0]/all_images.shape[0],
                                                                   images['test'].shape[0]/all_images.shape[0]))
    
    images_t = {key: images[key].reshape((-1, 1, images[key].shape[1], images[key].shape[2])) for key in ['train', 'validation', 'test']}
    x_train, x_val, x_test = images_t['train'], images_t['validation'], images_t['test']
    y_train, y_val, y_test = labels['train'], labels['validation'], labels['test']

    print('train data: {}; lables {}'.format(x_train.shape, y_train.shape))
    print('test : {}; lables {}'.format(x_val.shape, y_val.shape))
    print('test : {}; lables {}'.format(x_test.shape, y_test.shape))

    # CREATE GENERATORS
    params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 1}

    # Generators
    training_set = Dataset(x_train, y_train)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    print("Creating val generator...")
    validation_set = Dataset(x_val, y_val)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    print("Creating test generator...")
    params_test = {'batch_size': 1,
            'shuffle': False,
            'num_workers': 1}

    test_set = Dataset(x_test, y_test)
    test_generator = torch.utils.data.DataLoader(test_set, **params_test)

    # MODEL

    assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
    num_patches = (image_size // patch_size) ** 2
    patch_dim = in_channels * patch_size ** 2
    print(f"Number of patches = {num_patches} with dimension {patch_dim} each")

    model = ViT(num_layers=num_layers, d_model=32, num_heads=num_heads, conv_hidden_dim=128, patch_size=patch_size, 
            num_answers=15, att_dropout_rate=0, dropout_rate=0,
            num_patches=num_patches, no_embed_classtoken=False, standard1Dpe=False, in_channels=in_channels, 
            cnn=False, class_token=True, add_pos_emb=True, pool='', classif_hidden=128,
            pre_logits=False, weight_init=True)
    model.to(device)
    
    if NoAmpt:
        optimizer = None
        factor = 0.5
        th = 0.1 #0.1 = 10 epochs of nb_batches=22
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.001)
        factor=0
        th=0
    train_eval = Training(model, epochs, model_name, optimizer=optimizer, criterion=F.cross_entropy, base_path=base_path, debug=True)
    train_eval.print_parameters()
    
    train_eval.train(training_generator, validation_generator, checkpoint_metric='val_Loss', factor=factor, 
        warmup_rate=th, lr=0, betas=(0.9, 0.99), eps=1e-9, 
        weight_decay=0.0, save_plots=True)
    #plot_history(train_eval.history, epochs)
    
    #Visualize pos embeddings:
    w_path = '{}weights/{}/best_model.pth'.format(base_path, model_name)
    w_dict = get_weights(w_path)
    print('Loading pre-trained weights from ' + w_path)
    targets, preds, attn_w_dicts = train_eval.test(test_generator, load_weights=w_dict)


    model.load_state_dict(w_dict)

    params_name = 'pos_embed.pe'
    path_save = base_path + 'results/plots/{}/'.format(model_name)
    pe = get_layerw(w_dict, params_name, reshape_size=(-1, model.d_model))
    cos_sim = visualize_pe2D(pe[1:, ...], num_patches, path_save=path_save)


    visualize_MAHD(attn_w_dicts[0], patch_size, num_heads, path_save=base_path + 'results/plots/{}/bestw_'.format(model_name))

