import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np 

import math, copy, time
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nn_Softargmax = nn.Softmax  # fix wrong name

'''
Missing:
- Label smoothing
- Initialisation
- Pre-training with Masked Model
- Save txt with model settings/input arguments
- Generalize metrics, self.history during training (not only loss and acc)
- GridSearch function
- Cross Validation function
'''


#MHA
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0, d_input=None):
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
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.W_v = nn.Linear(d_xv, d_model, bias=False) #(d_input, d_model)=(d_input, d_k*num_heads)
        
        # Outputs of all sub-layers need to be of dimension d_model
        self.W_h = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0) 
        k_length = K.size(-2) 
        
        # Scaling by d_k so that the soft(arg)max doesnt saturate
        Q = Q / np.sqrt(self.d_k)                         # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(Q, K.transpose(2,3))          # (bs, n_heads, q_length, k_length)

        #if mask is not None:
        #  scores = scores.masked_fill(mask == 0, -1e9)
        
        A = nn_Softargmax(dim=-1)(scores)   # (bs, n_heads, q_length, k_length)

        A = self.dropout(A)
        
        # Get the weighted average of the values
        H = torch.matmul(A, V)     # (bs, n_heads, q_length, dim_per_head)

        return self.dropout(H), A 

        
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
        
        return H, A



#FEED-FOWARD

class CNN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout_rate):
        super().__init__()
        self.k1convL1 = nn.Linear(d_model,    hidden_dim)
        self.k1convL2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.GELU() #completly differentiable form of ReLU
    
        self.dropout1 = nn.Dropout(p=dropout_rate)
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
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

#pe = PositionalEncoding(20, 0)
#h = pe.forward(Variable(torch.zeros(1, 100, 20)))


##INPUT EMBEDDINGS

class Embeddings(nn.Module):
    
    def __init__(self, d_model, patches_size, in_channels=1, 
            cnn=False, nchan_l1=16, l1_kw=7, nchan_l2=32, l2_kw=7, nchan_l3=32, l3_kw=5,
            special_token=0):
        '''
        Creates the embedding weights usiing convolutionak layers
        Applies these to the input images to simultaneously patch and embedd them

        :param cnn (bool): if False, embeddings will be just 1 conv layer to directly patch and embedd each patch into d_model dimensions
        :params nchan_l1, l1_kw, nchan_l2, l2_kw: dimensions of the 1st two layers of a CNN model for embedding (and patchin ing the final layer)
                                                only matter when cnn=True
        :param special_token (int): if 0 = no special token in added; if 1 = CLS token is added; if 2: CLS and SEP tokens are added
        '''
        super().__init__()
        self.d_model = d_model #embedded_dimensions
        self.patches_size = patches_size
        self.special_token = special_token

        # We can merge s2d+emb into a single conv; it's the same.
        # returns image batch=4D array (bs, self.hidden_size, n_patches, n_patches)
        # CHANNELS FIRST
        if not cnn:
            self.patch_embeddings = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.d_model-self.special_token, #embedded_dimensions - the dim for the futurely added special tokens
                kernel_size=self.patches_size, 
                stride=self.patches_size,
                padding='valid') #no padding, so if h % patches_size !=0, this doesn't 
                #consider the excedent rows/columns of the image
                #number of patches = the smaller integer possible in h / patch_size
        else:
            self.patch_embeddings = self.CNN_emb(in_channels, nchan_l1, l1_kw, nchan_l2, l2_kw, nchan_l3, l3_kw)

        self.activation = nn.Tanh()

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
            #nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=nchan_l2,
                out_channels=nchan_l3, #embedded_dimensions
                kernel_size=l3_kw, 
                padding='same'),
            nn.BatchNorm2d(nchan_l3),
            nn.ReLU(),
            #nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=nchan_l3,
                out_channels=self.d_model-self.special_token, #embedded_dimensions - the dim for the futurely added special tokens
                kernel_size=self.patches_size, 
                stride=self.patches_size,
                padding='valid'), #no padding, so if h % patches_size !=0, this doesn't 
                #consider the excedent rows/columns of the image
                #number of patches = the smaller integer possible in h / patch_size
            #nn.Dropout(p=0.5)
        )
        return patch_embeddings


    def special_tokens(self,x):
        """We apply this after the batch passed through the cnn (x = (number_seq, seq_length, d_model))
        It adds the cls token at the start of the sequence
        it also adds a new "channel" to the cnn channels
        where cls has value 1 and everything else has value 0
        """
            
        # add the special value of the CLS token
        # We padd with zeros
        # Then one row at the start (The CLS token)
        # and one row at the end (the SEP token)
        # We then need to add +1 for the CLS token on the CLS feature (1st feature)
        # and similarly +1 for the SEP token on the SEP feature (2nd feature)

        if self.special_token==2:
            # add 2 channels at the start of the matrix (2,0) = 2 channels added on the top of the last dim
            # and 2 positions to the 'sequence', one at the end and one (SEP) at the start (CLS) (1,1)
            x = F.pad(x, (2,0,1,1))
            # mark the 1st channel of every 1st position in each sequence as 1, to mark a CLS token
            x[:,0,0]=1
            # mark the 2nd channel of every last position as 1, to mark a SEP token
            x[:,-1,1]=1
        elif self.special_token==1:
            # add 1 channels at the start of the matrix (1,0) = 1 ch added on the top of the last dim
            # and 1 position to the 'sequence', at the start (CLS) (1,0)
            x = F.pad(x, (1,0,1,0))
            # mark the 1st channel of every 1st position in each sequence as 1, to mark a CLS token
            x[:,0,0]=1
        return x


    def forward(self, x):
        x = self.patch_embeddings(x) #word_embeddings
        x = self.activation(x)
        bs, c, h, w = x.size()
        # Here, x is a grid of embeddings.
        #print(x.size(), self.special_token)
        x = torch.reshape(x, (bs, -1, c)) #(bs, n_patches*n_patches, d_model - special_token)
        #CHANNELS LAST
        # turning x into shape (bs, seq_length, d_model)

        if self.special_token != 0:
            x = self.special_tokens(x) 
        return x


##ENCODERLAYER

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, conv_hidden_dim, dropout_rate=0):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate=dropout_rate)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.cnn = CNN(d_model, conv_hidden_dim, dropout_rate)
    
    def forward(self, x):
        assert x.ndim == 3, f'Expected (batch, seq, hidden) got {x.shape}'

        # Layer norm before MHA
        x_norm1 = self.layernorm1(x)  # (batch_size, input_seq_len, d_model)

        # Multi-head attention 
        attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)

        x_DO = self.dropout1(attn_output)
        x1 = x_DO + x
        
        # Layer norm after adding the residual connection 
        x_norm2 = self.layernorm2(x1)  # (batch_size, input_seq_len, d_model)
        
        # Feed forward 
        cnn_output = self.cnn(x_norm2)  # (batch_size, input_seq_len, d_model)
        
        #Final residual connection 
        x2 = x1 + cnn_output # (batch_size, input_seq_len, d_model)

        return x2

##ENCODER

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ff_hidden_dim, patch_size,
                 dropout_rate=0, max_pos_emb=5000, in_channels=1, 
                 cnn=False, nchan_l1=16, l1_kw=7, nchan_l2=32, l2_kw=5, nchan_l3=32, l3_kw=5,
                 special_token=0, add_pos_emb=True):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.add_pos_emb = add_pos_emb

        self.embedding = Embeddings(d_model, patch_size, in_channels=in_channels, 
                                    cnn=cnn, nchan_l1=nchan_l1, l1_kw=l1_kw, 
                                    nchan_l2=nchan_l2, l2_kw=l2_kw, nchan_l3=nchan_l3, l3_kw=l3_kw,
                                    special_token=special_token)
        self.pe = PositionalEncoding(d_model, dropout_rate, max_pos_emb)

        self.enc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_layers.append(EncoderLayer(d_model, num_heads, ff_hidden_dim, dropout_rate=dropout_rate))
        
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        
    def forward(self, x):
        x = self.embedding(x) # Transform to (batch_size, input_seq_length, d_model)

        if self.add_pos_emb:
          x = self.pe(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        x = self.LayerNorm(x)
        return x  # (batch_size, input_seq_len, d_model)



#TRANSFORMER CLASSIFIER

class TransformerClassifier(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, conv_hidden_dim, patch_size, 
                 num_answers, dropout_rate=0, 
                 max_pos_emb=5000, in_channels=1, 
                 cnn=False, nchan_l1=16, l1_kw=7, nchan_l2=32, l2_kw=5, nchan_l3=32, l3_kw=5,
                 special_token=0, add_pos_emb=True,
                 avgpool=True):
        super().__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, conv_hidden_dim, patch_size,
                         dropout_rate=dropout_rate, max_pos_emb=max_pos_emb, in_channels=in_channels, 
                         cnn=cnn, nchan_l1=nchan_l1, l1_kw=l1_kw, nchan_l2=nchan_l2, l2_kw=l2_kw, 
                         nchan_l3=nchan_l3, l3_kw=l3_kw,
                         special_token=special_token, add_pos_emb=add_pos_emb)
        self.dense = nn.Linear(d_model, num_answers)

        self.special_token = special_token
        self.avgpool = avgpool #if true and special_token=0, computes the mean over all the words/patches in each sequence
                            #for each embedde dim. So (bs, seq_length, d_model) --> (bs, d_model) 


    def forward(self, x):
        x = self.encoder(x) #(bs, seq_length, d_model)
        
        if self.special_token==0:
            #do the mean/max over all patches/words in each image/sequence, instead of using CLS token
            if self.avgpool:
                x = torch.mean(x, dim=1) #mean over rows/words/tokens/sequence elements
                # x = (bs, d_model) --> mean probability of each emb dim feature/property/cluster happen
                # in that sequence, independently of the position (considering all the depicted relevant contexts/rows in the sequence)
            else:
                x, _ = torch.max(x, dim=1)
        else:
            x = x[:, 0, :] #classification_embedding (bs, d_model) 
        x = self.dense(x)
        return x
