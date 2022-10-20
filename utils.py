from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import torch




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

def partition_data(data, validation=1000, test=1000):
    assert validation + test < data.shape[0]
    return {"train": data[validation:-test],
        "validation": data[:validation],
        "test": data[-test:]}

def plot_history(history, n_epochs):
    plt.figure()
    for metric in ['Accuracy', 'Loss']:
      # summarize history for accuracy and loss
      legend = []
      for k in history.keys(): 
        if metric in k:
          plt.plot(history[k])
          legend.append(k)
      plt.title('model' + metric)
      plt.ylabel(metric)
      plt.xlabel('epoch')
      plt.legend(legend, loc='upper left')
      plt.show()

## GET PRE-TRAINED WEIGHTS

def get_weights(weights_path):
    model_cp = torch.load(weights_path)
    model_epoch = model_cp['epoch']
    print(f"Model was saved at {model_epoch} epochs in " + weights_path + "\n")
    return model_cp['model_state_dict'] 

def get_layerw(model_state_dict, params_name, reshape_size=None): 
    params = model_state_dict[params_name]
    print("Getting model parameters: " + params_name + "with size = ", params.size())
    if reshape_size is not None:
        params = torch.reshape(params, reshape_size)
        print("Resized to: ", params.size())
    params = params.cpu().data.numpy()
    return params


# INVESTIGATING VISOPN TRANSFORMERS REPRESENTATIONS

#VISUALIZE EMBEDDINGS 
def visualize_pe2D(pe, num_patches, path_save=None):
    ''' VISUALIZE POS EMBEDDINGS COS SIMILARITY between each pair of patches'''
    n_patches_perdim = int(math.sqrt(num_patches))
    #pe_2D = pe.reshape(n_patches_perdim, n_patches_perdim, -1) #size=(num_patches, d_model)

    cos_similarity = np.zeros((n_patches_perdim, n_patches_perdim, num_patches))

    for row in range(cos_similarity.shape[0]):
        for col in range(cos_similarity.shape[1]):
            #patch = pe_2D[np.newaxis,row, col, :]
            patch = pe[row*n_patches_perdim + col, :] #size=(1,64)
            sim = cosine_similarity(patch[np.newaxis, ...], pe) #size=(1,num_patches)
            cos_similarity[row, col, :] = sim

    fig, ax = plt.subplots(n_patches_perdim, n_patches_perdim, figsize=(12,12))
    for i in range(n_patches_perdim):
        for j in range(n_patches_perdim):
            patch_sim = cos_similarity[i,j,:].reshape(n_patches_perdim, n_patches_perdim)
            #plots the cos sim between the patch corresponding to position (i,j) in the image and all the other patches
            im = ax[i,j].imshow(patch_sim) #vmin=-0.6, vmax=0.6
            if j==0:
                ax[i,j].set_ylabel(f"Row_{i}")
            if i==n_patches_perdim-1:
                ax[i,j].set_xlabel(f"Col_{j}")

    im.cmap.set_under("w")
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), label='Intensity')
    

    if path_save is not None:
        os.makedirs(path_save, exist_ok=True)
        plt.savefig(path_save+'PosEmbedds_cossim.png')
    #plt.show()
    #plt.close()
    return cos_similarity


#MEAN ATTENTION DISTNACE
#https://github.com/sayakpaul/probing-vits/blob/main/notebooks/mean-attention-distance-1k.ipynb
def compute_distance_matrix(patch_size, num_patches, length):
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    #print(f"Minimum and maximum geometric distance: {distance_matrix.min()} and {distance_matrix.max()}")
    return distance_matrix


def compute_mean_attention_dist(patch_size, attention_weights):
    num_cls_tokens = 1 #+1 in the attention dimensions

    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    attention_weights = attention_weights[
        ..., num_cls_tokens:, num_cls_tokens:
    ].detach().cpu().numpy()  # Removing the CLS token
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert length**2 == num_patches, "Num patches is not perfect square"

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    # The attention_weights along the last axis adds to 1
    # this is due to the fact that they are softmax of the raw logits
    # summation of the (attention_weights * distance_matrix)
    # should result in an average distance per token.
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(
        mean_distances, axis=-1
    )  # Sum along last axis to get average distance per token
    mean_distances = np.mean(
        mean_distances, axis=-1
    )  # Now average across all the tokens

    return mean_distances

def visualize_MAHD(attention_score_dict, patch_size, num_heads, path_save=None):
    print(f"Num Heads: {num_heads}.")
    legends = [f"h{i}" for i in range(num_heads)]

    plt.figure(figsize=(9, 9))

    for k in attention_score_dict.keys():
        mean_distances = compute_mean_attention_dist(patch_size, attention_score_dict[k][:, :, :, :])
        idx = int(k.split("_")[-1]) #layer index
        x = [idx] * num_heads #each hyead distnace will be plotted in the same x coordinate
        y = mean_distances[0, :]
        plt.scatter(x=x, y=y, label=k)

        for i, txt in enumerate(legends):
            plt.annotate(txt, (x[i] + 0.15, y[i] + 0.15))

    plt.legend(loc="upper left")
    plt.xlabel("NN Depth (layer)", fontsize=14)
    plt.ylabel("Attention Distance", fontsize=14)
    plt.title("Mean Attention Distance", fontsize=14)
    plt.grid()

    if path_save is not None:
        #os.makedirs(path_save, exist_ok=True)
        plt.savefig(path_save+'MAD.png')
    #plt.show()
    #plt.close()