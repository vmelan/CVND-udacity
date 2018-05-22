import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        ''' Initialize the layers of this model.'''
        super(DecoderRNN, self).__init__()
        
        # Keep track of hidden_size for initialization of hidden state
        self.hidden_size = hidden_size
        
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=embed_size, \
                            hidden_size=hidden_size, # LSTM hidden units 
                            num_layers=1, # number of LSTM layer
                            bias=True, # use bias weights b_ih and b_hh
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0, # Not applying dropout 
                            bidirectional=False, # unidirectional LSTM
                           )
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)                     

        # initialize the hidden state
        # self.hidden = self.init_hidden()
        
    def init_hidden(self, batch_size):
    	""" At the start of training, we need to initialize a hidden state;
    		there will be none because the hidden state is formed based on previously seen data.
    		So, this function defines a hidden state with all zeroes
	    	The axes semantics are (num_layers, batch_size, hidden_dim)
	    """
    	return (Variable(torch.zeros(1, batch_size, self.hidden_size)), \
				Variable(torch.zeros(1, batch_size, self.hidden_size)))

    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """
        # Initialize the hidden state
        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden(self.batch_size) # features is of shape (batch_size, embed_size)

        # Create embedded word vectors for each word in the captions
        embeds = self.word_embeddings(captions)

        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(embeds.view(len(captions), self.batch_size, -1), self.hidden) # input of shape (seq_len, batch, input_size)  

        # Flatten
        lstm_out = lstm_out.view(lstm_out.size(0), -1) 

        # Fully connected layer
        outputs = self.linear(lstm_out)

        return outputs

    def sample(self, inputs):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass