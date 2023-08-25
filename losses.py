# Only Test

from fastai.vision import *

from modules.model import Model

import re
from utils import onehot

from radical_tree import *

class MultiLosses(nn.Module):
    def __init__(self, one_hot=True):
        super().__init__()
    
    def forward(self, outputs, *args):
        
        a = torch.Tensor([1]).requires_grad_(True).cuda()
        return a


