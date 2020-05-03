import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import init



    


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        # I used Xavier initialization 
        #https://discuss.pytorch.org/t/how-to-initialize-the-conv-layers-with-xavier-weights-initialization/8419/2
        # https://stats.stackexchange.com/questions/229669/when-should-i-use-the-normal-distribution-or-the-uniform-distribution-when-using
        
        init.xavier_uniform_(self.embed.weight)
        
        self.embed.bias.data.fill_(0.0)
                
        
        # we add batch normalization in order to improve the training.
        self.bn = nn.BatchNorm1d(embed_size)
                

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features) 
        return features
    

    
## In order to implement the Decoder I took ideas from the following web pages
## https://machinetalk.org/2019/02/08/text-generation-with-pytorch/
## https://tsdaemon.github.io/2018/07/08/nmt-with-pytorch-encoder-decoder.html#fn:bengio2014
## https://knowledge.udacity.com/questions/56826
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):                                
        super(DecoderRNN, self).__init__()        
        
        self.hidden_size = hidden_size 
        self.vocab_size = vocab_size 
        self.num_layers = num_layers
        
        
        # we define the embedding layer. 
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        init.xavier_uniform_(self.embedding.weight)
                
        
        
        # we define the lstm neural network.
        self.lstm = nn.LSTM(input_size=embed_size, 
                            hidden_size=hidden_size, 
                            num_layers = num_layers, 
                            batch_first=True, 
                            bidirectional=False)
        
        
        # I used Xavier initialization 
        # https://discuss.pytorch.org/t/how-to-initialize-weights-bias-of-rnn-lstm-gru/2879/6
        
        
        for layer_p in self.lstm._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    init.xavier_uniform_(self.lstm.__getattr__(p))        
                    
                if 'bias' in p:
                    init.normal_(self.lstm.__getattr__(p), 0.0, 0.01)        
                                                        
        # we define the linear layer that output the word.
        self.dense = nn.Linear(hidden_size, vocab_size)
        
        init.xavier_uniform(self.dense.weight)
        self.dense.bias.data.fill_(0.0)
       
    # we define a method that initialize the tensor containing 
    # the output features and the hidden state.
    def init_hidden_layer(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).to('cuda'), torch.zeros(1, batch_size, self.hidden_size).to('cuda')
    
    def forward(self, features, captions):
        
        # we exclude the end token.
        captions = captions[:, :-1]
        
        self.batch_size = features.shape[0]
        
        self.hidden = self.init_hidden_layer(self.batch_size)
        
        # we apply embedding layer to the word
        emb = self.embedding(captions)
        
        # we concatenate the features from the image and the embedding word by the 1th dimension.
        inputs = torch.cat((features.unsqueeze(dim=1), emb), dim=1)
        
        # we feed the emb concatenated tensor from above to the lstm neural network 
        lstm_output, self.hidden = self.lstm(inputs, self.hidden)
        
        # outputs produces a distribution that represents the most likely next word. 
        outputs = self.dense(lstm_output)
        
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # we store the word predictions in a list 
        predicted_ids = []
        
        # we will generate a caption of a length of 20 words. 
        for i in range(max_len):            
            
            hiddens, states = self.lstm(inputs, states)
            
            outputs = self.dense(hiddens.squeeze(1))
            
            _, prediction = outputs.max(1)
                        
            
            if prediction.item() == 1:
                break
                
            predicted_ids.append(prediction.item())
            
            inputs = self.embedding(prediction).unsqueeze(1)                    
        
        return predicted_ids
                