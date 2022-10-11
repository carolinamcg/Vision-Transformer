# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np 

import math
from weight_init import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_Softargmax = nn.Softmax  # fix wrong name

'''
Missing:
- Label smoothing
- Pre-training with Masked Model
- Save txt with model settings/input arguments
- Generalize metrics, self.history during training (not only loss and acc)
- GridSearch function
- Cross Validation function
- Plot training nad val times during training steps
'''


#MHA
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, att_dropout_rate=0, dropout_rate=0, d_input=None, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model #q, k, v dim * num_heads = d_k * num_heads
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input #input embeddings dimensions
            
        # Make sure that the embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0

        self.d_k = d_model // self.num_heads #dimension of each q, k, and v corresponds to d_model/num_heads
        # = input_emb_dimension / num_heads. So, if d_input=d_model=4 and num_heads=2, W_q = (d_xq, d_model) = (input_embeddings_dim, d_k * num_heads) = (4, 2*2)
        # each sequence element will have 2 queries of dimension 2 (Q = (bs, num_heads, seq_length, d_k) = (bs, 2, n, 2))
        
        # These are still of dimension d_model. They will be split into number of heads 
        self.W_q = nn.Linear(d_xq, d_model, bias=qkv_bias)
        self.W_k = nn.Linear(d_xk, d_model, bias=qkv_bias)

        self.att_dropout = nn.Dropout(p=att_dropout_rate)
        self.proj_dropout = nn.Dropout(p=dropout_rate)

        self.W_v = nn.Linear(d_xv, d_model, bias=qkv_bias) #(d_input, d_model)=(d_input, d_k*num_heads)
        
        # Outputs of all sub-layers need to be of dimension d_model
        self.W_h = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        #batch_size = Q.size(0) 
        #k_length = K.size(-2) 
        
        # Scaling by d_k so that the soft(arg)max doesnt saturate
        Q = Q / np.sqrt(self.d_k)                         # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(Q, K.transpose(2,3))          # (bs, n_heads, q_length, k_length)

        #if mask is not None:
        #  scores = scores.masked_fill(mask == 0, -1e9)
        
        A = nn_Softargmax(dim=-1)(scores)   # (bs, n_heads, q_length, k_length)

        A = self.att_dropout(A)
        
        # Get the weighted average of the values
        H = torch.matmul(A, V)     # (bs, n_head1= datasetss, q_length, dim_per_head)

        return H, A 

        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (heads X depth)
        Return after transpose to put in shape (batch_size X num_heads X seq_length X d_k)
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        """
        Combine the heads again to get (batch_size X seq_length X (num_heads times d_k))
        """
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
    

    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.size()

        # After transforming, split into num_heads 
        Q = self.split_heads(self.W_q(X_q), batch_size)  # (bs, n_heads, q_length, dim_per_head), dim_per_head = d_k
        K = self.split_heads(self.W_k(X_k), batch_size)  # (bs, n_heads, k_length, dim_per_head)
        V = self.split_heads(self.W_v(X_v), batch_size)  # (bs, n_heads, v_length, dim_per_head)
        #print(Q.size(), K.size(), V.size(), self.W_q.weight .size())
        
        # Calculate the attention weights for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V) #(bs, n_heads, q_length, d_k), (bs, n_heads, q_length, k_length)
        #print(A.size(), H_cat.size())
        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat, batch_size) #(bs, q_length, n_heads*d_k)=(bs, q_length, d_model)
        #print(H_cat.size())
        # Final linear layer  
        H = self.W_h(H_cat)          # (bs, q_length, d_model)
        H = self.proj_dropout(H)
        
        return H, A



#FEED-FOWARD

class CNN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout_rate):
        super().__init__()
        self.k1convL1 = nn.Linear(d_model,    hidden_dim)
        self.activation = nn.GELU() #completly differentiable form of ReLU
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.k1convL2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.k1convL2(x)
        x = self.dropout2(x)
        return x



#ENCODER

##POSITIONAL EMBEDDING

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, embed_len, standard1Dpe=True):
        super(PositionalEncoding, self).__init__()
        if standard1Dpe:
            # Compute the santarde 1D positional encodings (sinusoidal) for language models
            self.pe = nn.Embedding(embed_len, d_model)
            #self.pe = nn.Identity() # just return the input without any manipulation and can be used to e.g. replace other layers.
            self.create_sinusoidal_embeddings(nb_p=embed_len,
                                dim=d_model,
                                E=self.pe.weight)
        else:
            #Learn the positional embeddings while training
            self.pe = nn.Parameter(torch.randn(1, embed_len, d_model), requires_grad=True) #*0.02

    def create_sinusoidal_embeddings(self, nb_p, dim, E):
        position = torch.arange(0, nb_p).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                            -(math.log(10000.0) / dim))
        E[:, 0::2] = torch.sin(position * div_term)
        E[:, 1::2] = torch.cos(position * div_term)
        E = E.unsqueeze(0)
        E.requires_grad = False
        E = E.to(device)



##INPUT EMBEDDINGS

class Embeddings(nn.Module):
    
    def __init__(self, d_model, patches_size, in_channels=1, 
            cnn=False, nchan_l1=16, l1_kw=7, nchan_l2=32, l2_kw=7, nchan_l3=32, l3_kw=5,
            class_token=False):
        '''
        Creates the embedding weights usiing convolutionak layers
        Applies these to the input images to simultaneously patch and embedd them

        :param cnn (bool): if False, embeddings will be just 1 conv layer to directly patch and embedd each patch into d_model dimensions
        :params nchan_l1, l1_kw, nchan_l2, l2_kw: dimensions of the 1st two layers of a CNN model for embedding (and patchin ing the final layer)
                                                only matter when cnn=True
        :param class_token (int): if True, a CLS token need to be created and added to the top of each sequence of image patches
        '''
        super().__init__()
        self.d_model = d_model #embedded_dimensions
        self.patches_size = patches_size

        # We can merge s2d+emb into a single conv; it's the same.
        # returns image batch=4D array (bs, self.hidden_size, n_patches, n_patches)
        # CHANNELS FIRST
        if not cnn:
            self.patch_embeddings = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.d_model, #embedded_dimensions
                kernel_size=self.patches_size, 
                stride=self.patches_size,
                padding='valid') #no padding, so if h % patches_size !=0, this doesn't 
                #consider the excedent rows/columns of the image
                #number of patches = the smaller integer possible in h / patch_size
        else:
            self.patch_embeddings = self.CNN_emb(in_channels, nchan_l1, l1_kw, nchan_l2, l2_kw, nchan_l3, l3_kw)


    def CNN_emb(self, in_channels, nchan_l1=16, l1_kw=7, nchan_l2=32, l2_kw=7, nchan_l3=32, l3_kw=5):
        patch_embeddings = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=nchan_l1, #embedded_dimensions
                    kernel_size=l1_kw, 
                    padding='same'),
            nn.BatchNorm2d(nchan_l1),
            nn.ReLU(),

            nn.Conv2d(in_channels=nchan_l1,
                out_channels=nchan_l2, #embedded_dimensions
                kernel_size=l2_kw, 
                padding='same'),
            nn.BatchNorm2d(nchan_l2),
            nn.ReLU(),
            #nn.Dropout(p=0.25),sn't 
                #consider the excedent rows/columns of the image
                #number of patches = the smaller integer possible in h / patch_si

            nn.Conv2d(in_channels=nchan_l2,
                out_channels=nchan_l3, #embedded_dimensions
                kernel_size=l3_kw, 
                padding='same'),
            nn.BatchNorm2d(nchan_l3),
            nn.ReLU(),
            #nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=nchan_l3,
                out_channels=self.d_model, #embedded_dimensions
                kernel_size=self.patches_size, 
                stride=self.patches_size,
                padding='valid'), #no padding, so if h % patches_size !=0, this doesn't 
                #consider the excedent rows/columns of the image
                #number of patches = the smaller integer possible in h / patch_size
            #nn.Dropout(p=0.5)
        )
        return patch_embeddings

    def forward(self, x):
        x = self.patch_embeddings(x) #word_embeddings
        bs, c, h, w = x.size()
        # Here, x is a grid of embeddings.
        #print(x.size(), self.special_token)
        x = torch.reshape(x, (bs, -1, c)) #(bs, n_patches*n_patches, d_model)
        #CHANNELS LAST
        # turning x into shape (bs, seq_length, d_model)
        return x





##ENCODERLAYER

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, conv_hidden_dim, att_dropout_rate=0, dropout_rate=0):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.mha = MultiHeadAttention(d_model, num_heads, att_dropout_rate=att_dropout_rate, dropout_rate=dropout_rate)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.cnn = CNN(d_model, conv_hidden_dim, dropout_rate)
    
    def forward(self, x):
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'

        # Layer norm before MHA
        x_norm1 = self.layernorm1(x)  # (batch_size, input_seq_len, d_model)
        # Multi-head attention 
        attn_output, attn_weights = self.mha(x_norm1, x_norm1, x_norm1)  # (batch_size, input_seq_len, d_model), (bs, n_heads, q_length, k_length)
        x_DO = self.dropout1(attn_output) #NEEDED???????????
        x1 = x_DO + x
        
        # Layer norm after adding the residual connection 
        x_norm2 = self.layernorm2(x1)  # (batch_size, input_seq_len, d_model)
        # Feed forward 
        cnn_output = self.cnn(x_norm2)  # (batch_size, input_seq_len, d_model)
        #Final residual connection 
        x2 = x1 + cnn_output # (batch_size, input_seq_len, d_model)

        return x2, attn_weights




##ENCODER

class ViT(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, conv_hidden_dim, patch_size, 
                 num_answers, att_dropout_rate=0, dropout_rate=0,
                 num_patches=28, no_embed_classtoken=False, standard1Dpe=True, in_channels=1, 
                 cnn=False, nchan_l1=16, l1_kw=7, nchan_l2=32, l2_kw=5, nchan_l3=32, l3_kw=5,
                 class_token=False, add_pos_emb=True, pool="avg", classif_hidden=None,
                 pre_logits=False, weight_init=True):
        super().__init__()

        # ENCODER SETTINGS
        self.d_model = d_model
        self.num_layers = num_layers
        self.add_pos_emb = add_pos_emb
        #     - CLS TOKEN
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) if class_token else None #radn or zeros???????????????
        self.num_prefix_tokens = 1 if class_token else 0
        #     - POS EMBEDDINGS
        self.no_embed_classtoken = no_embed_classtoken #if True, we don't had pos_embedding to CLS token
        embed_len = num_patches if no_embed_classtoken else num_patches + self.num_prefix_tokens
        #     - LAYERS
        self.embedding = Embeddings(d_model, patch_size, in_channels=in_channels, 
                                    cnn=cnn, nchan_l1=nchan_l1, l1_kw=l1_kw, 
                                    nchan_l2=nchan_l2, l2_kw=l2_kw, nchan_l3=nchan_l3, l3_kw=l3_kw,
                                    class_token=class_token)
        self.pos_embed = PositionalEncoding(d_model, embed_len, standard1Dpe)
        self.pos_dropout = nn.Dropout(p=dropout_rate)

        self.enc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_layers.append(EncoderLayer(d_model, num_heads, conv_hidden_dim, 
                    att_dropout_rate=att_dropout_rate, dropout_rate=dropout_rate))

        self.attn_weights_dict = {} #saves the attention scores for each encoder layer/block and each head
        
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)


        # HEAD SETTINGS
        self.class_token = class_token
        self.pre_logits = pre_logits #if true, returns token representations/features before the classifier head and after the defined pool operation
        self.pool = pool 
        assert self.pool in ('avg','max', ''), "Invalid option for pool"
        if not pre_logits or num_answers>0:
            assert class_token!=False or pool!='', "Image classification will not be possible unless a pool operation is defined ('avg', 'max') or a CLS token is used"

        if classif_hidden is None:
            self.classifier = nn.Linear(d_model, num_answers) if num_answers > 0 else nn.Identity()
            #if classif_hidden is None, we just want a linear layer as classifier head or no head
            #when num_answers are not given
        else:
            #when classif_head is no None, is the dimension on the MLP classifier head
            #when this parameter is given, it is because num_answers > 0
            assert num_answers > 0, "Number of classes need to be > 0 to have an MLP head with hidden size = classif_hidden"
            self.classifier = nn.Sequential(nn.Linear(d_model, classif_hidden),
                                        nn.GELU(),
                                        nn.Linear(classif_hidden, num_answers)
                                        )


        if weight_init: # != 'skip':
            trunc_normal_(self.pos_embed.pe, std=.02)
            if self.cls_token is not None:
                nn.init.normal_(self.cls_token, std=1e-6)
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight) #VIT_Jax Github
            #trunc_normal_(module.weight, std=.02) #""" ViT weight initialization, original timm impl (for reproducibility) """
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6)
        elif isinstance(module, nn.Conv2d):
            if module.bias is not None:
                nn.init.zeros_(module.bias)



    def _pos_embed(self, x):
        bs, _, _ = x.size()
        if self.no_embed_classtoken:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed.pe
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(bs, -1, -1), x), dim=1) 
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(bs, -1, -1), x), dim=1)
            x = x + self.pos_embed.pe
        return self.pos_dropout(x)              
        
    def forward_features(self, x):
        x = self.embedding(x) # Transform to (batch_size, input_seq_length, d_model)
        if self.add_pos_emb:
          x = self._pos_embed(x)

        for i in range(self.num_layers):
            x, A = self.enc_layers[i](x)
            self.attn_weights_dict[f"layer_{i}"] = A

        x = self.LayerNorm(x)
        return x # (batch_size, input_seq_len, d_model)
    


    def forward_head(self, x):
        if self.pool=='avg':
            #do the mean/max over all patches/words in each image/sequence, instead of using CLS token
            x = x[:, self.num_prefix_tokens:, :].mean(dim=1) #mean over rows/words/tokens/sequence elements
            # x = (bs, d_model) --> mean probability of each emb dim feature/property/cluster happen
            # in that sequence, independently of the position (considering all the depicted relevant contexts/rows in the sequence)
        elif self.pool=='max':
            x, _ = x[:, self.num_prefix_tokens:, :].max(x, dim=1)
        else:
            x = x[:, 0, :] if self.class_token else x
            #return classification tokens (bs, d_model) or token representation (bs, seq_length, d_model)

        x = x if self.pre_logits else self.classifier(x)
        return x
    


    def forward(self, x):
        x = self.forward_features(x) #(bs, seq_length, d_model)
        x = self.forward_head(x)
        return x, self.attn_weights_dict

    




#TRANSFORMER CLASSIFIER

''' 
class TransformerClassifier(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, conv_hidden_dim, patch_size, 
                 num_answers, dropout_rate=0, 
                 max_pos_emb=5000, standard1Dpe=True, in_channels=1, 
                 cnn=False, nchan_l1=16, l1_kw=7, nchan_l2=32, l2_kw=5, nchan_l3=32, l3_kw=5,
                 class_token=False, add_pos_emb=True, pool="avg", classif_hidden=None):
        super().__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, conv_hidden_dim, patch_size,
                         dropout_rate=dropout_rate, max_pos_emb=max_pos_emb, standard1Dpe=standard1Dpe, 
                         in_channels=in_channels, 
                         cnn=cnn, nchan_l1=nchan_l1, l1_kw=l1_kw, nchan_l2=nchan_l2, l2_kw=l2_kw, 
                         nchan_l3=nchan_l3, l3_kw=l3_kw,
                         class_token=class_token, add_pos_emb=add_pos_emb)

        if classif_hidden is None:
            self.classifier = nn.Linear(d_model, num_answers)
        else:
            self.classifier = nn.Sequential(nn.Linear(d_model, classif_hidden),
                                        nn.GELU(),
                                        nn.Linear(classif_hidden, num_answers)
                                        )

        self.class_token = class_token
        self.pool = None if class_token else pool #if class_token==False: computes the mean or max over all the words/patches in each sequence
        #for each embedde dim. So (bs, seq_length, d_model) --> (bs, d_model) 
        assert self.pool=='avg' or self.pool=='max' or self.pool==None, "Invalid option for pool"

    def forward(self, x):
        x = self.encoder(x) #(bs, seq_length, d_model)
        
        if self.pool=='avg':
            #do the mean/max over all patches/words in each image/sequence, instead of using CLS token
            x = torch.mean(x, dim=1) #mean over rows/words/tokens/sequence elements
            # x = (bs, d_model) --> mean probability of each emb dim feature/property/cluster happen
            # in that sequence, independently of the position (considering all the depicted relevant contexts/rows in the sequence)
        elif self.pool=='max':
            x, _ = torch.max(x, dim=1)
        else:
            x = x[:, 0, :] #classification_embedding (bs, d_model) 

        x = self.classifier(x)
        return x
'''