


import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Direct Policy Gradient network. 

Takes in STATE_DIM size vector, and outputs ACTIONS_DIM size vector. 
Action select (sampling), and reward backprop is done in another function.
Make sure to subtract a baseline! 
'''

class PolicyNet(nn.Module):
    def __init__(self, state_dim, actions_dim, hidden_dim=64):
        super(PolicyNet, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, actions_dim)


    def forward(self,x):
        x = F.relu(self.input_layer(x))
        return F.softmax(self.hidden(x)) # return a valid prob distribution..or not!

