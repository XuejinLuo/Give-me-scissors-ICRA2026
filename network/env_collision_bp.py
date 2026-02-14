# bp.py
import torch
import torch.nn as nn

class BP(nn.Module):
    def __init__(self, input_dim, nodes_per_layer=[256, 256, 256, 256, 256]):
        super(BP, self).__init__()
        
        
        layers = []
        layers.append(nn.Linear(input_dim, nodes_per_layer[0]))  
        layers.append(nn.LeakyReLU())
        
        for i in range(1, len(nodes_per_layer)):  
            layers.append(nn.Linear(nodes_per_layer[i-1], nodes_per_layer[i]))
            layers.append(nn.LeakyReLU())
        
        layers.append(nn.Linear(nodes_per_layer[-1], 1))  
        layers.append(nn.LeakyReLU())
        
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        output = self.model(input)
        return output

def create_model(args):
    nodes_per_layer = [int(x) for x in args.nodes_per_layer.split(",")]
    model = BP(args.input_num, nodes_per_layer=nodes_per_layer)
    return model