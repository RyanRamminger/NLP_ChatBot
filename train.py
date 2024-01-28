#creating training data

import json
from nltk_utils import tokenize, stem, bag_of_words #our functions 
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet


with open('intents.json', 'r') as f:
    intents = json.load(f)

#testing if json file works
#print(intents)

all_words = [] 
tags = []
xy = []
#in the json file, the objects will be dictionaries beginning with the 'intents' key followed by an array of all the different text, patterns, responces
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']: #looping over the array with key 'patterns'
        w = tokenize(pattern) #we want to put these into the all_words array
        all_words.extend(w) #using extend because w output will be an array with several words
        xy.append((w, tag)) #appending the corresponding labels. xy knows the pattern 'w' with corresponding 'tag'

ignore_words = ["?", "!", ".", "."]
all_words = [stem(w) for w in all_words if w not in ignore_words] #words here have gone though our stem function, they are lowercase, chopped, and removed if in 'ignore_words'
all_words = sorted(set(all_words))  #the set removes duplicate words, sorted returns a list
tags = sorted(set(tags)) #doing the same with tags, 

#creating the bag of words for training data
x_train = [] #bag of words inside
y_train = []
for (pattern_sentence, tag) in xy: #unpacking our tuple 'xy.append((w,tag))'
    bag = bag_of_words(pattern_sentence, all_words)  #here, 'bag_of_words' already applied tokenization
    x_train.append(bag)

    label = tags.index(tag) #labels the tags in order (delivery = 0, funny = 1...), gives us numbers to our labels
    y_train.append(label) #1 hot encoded vector, but with pytorch we only want Class label (CrossEntropyLoss)

x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = torch.LongTensor(y_train)  # Convert to LongTensor


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    #dataset[idx]  
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

#parameters that we can change
batch_size = 5 #8
hidden_size = 18 #8
output_size = len(tags)  #number of different classes/tags
input_size = len(x_train[0])  #number of each bag of words created 'all_words' array
#testing out model
# print(input_size, len(all_words))
# print(output_size, tags)
learning_rate = 0.001
num_epochs = 1000 


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:  # training loop
        words = words.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward pass and optimizer step, empty gradients first
        optimizer.zero_grad()
        loss.backward()  # back propagation
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

#all is left to do is to save the data in a dictionary
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words, 
    "tags": tags    
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. File saved to {FILE}')
