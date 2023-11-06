import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        return
    
    def forward(self, features, captions):
        captions = captions[:, :-1]     #remove the <end> token
        embedded = self.word_embeddings(captions)
        
        features_prep = torch.unsqueeze(features, 1)   ##add extra dimension to feature in order to fit with embedded
        
        embedded = torch.cat((features_prep, embedded), dim=1)
        
        lstm_out, (c, h) = self.lstm(embedded)
        #print("Output dimensions are: ", lstm_out.shape)
        
        linear_out = self.fc(lstm_out)
        
        return linear_out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        outputs = []
        
        for i in range(max_len):
            out, new_states = self.lstm(inputs, states)      ##added states as output in order to continue with the prediction
            out = out.squeeze(dim=1)    #reduce dimension from (1, 1, hidden_size) to (1, hidde_size) for input into fc
            
            pred = self.fc(out)          #output dimension is (1, vocab_size)
            
            value, index = torch.max(pred, dim=1)    #find the max value and its index in dim=1(columns)
            
            pythonIndex = index.item()     #return the basic python number
            
            outputs.append(pythonIndex)
            
            if pythonIndex == 1:
                break             ###if index is <end> token, break the loop
            
            inputs = self.word_embeddings(index)
            inputs = inputs.unsqueeze(1)                ##add an extra dimension to fit the input of the lstm
            states = new_states
        
        return outputs