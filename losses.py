
"""

radicalwieght  0.5*(1+radicalweight)

"""


from fastai.vision import *

from modules.model import Model

import re
from utils import onehot

from radical_tree import *

class MultiLosses(nn.Module):
    def __init__(self, one_hot=True):
        super().__init__()
        self.ce = SoftCrossEntropyLoss() if one_hot else torch.nn.CrossEntropyLoss() 
        self.bce = torch.nn.BCELoss()
        #!
        self.max_length_radical = 33
        self.null_label = 0
        self.null_char = u'\u2591'
        
        self.label_to_char = self._read_charset("data/charset_zh.txt")
        self.char_to_label = dict(map(reversed, self.label_to_char.items()))
        
        self.label_to_char_radical = self._read_charset("data/radicals.txt")
        self.char_to_label_radical = dict(map(reversed, self.label_to_char_radical.items()))
        self.num_classes_radical = len(self.label_to_char_radical)
        

        files = open("data/decompose.txt",'r',encoding='utf-8').readlines()    
        self.radical = {}
        for line in files:
            items = line.strip('\n').strip().split(':')
            ch = items[0]
            ch_radical=items[1].split(' ')
            self.radical[ch]=ch_radical
    
    #!
    def _read_charset(self, filename):
        pattern = re.compile(r'(\d+)\t(.+)')
        charset = {}
        charset[self.null_label] = self.null_char
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                m = pattern.match(line)
                assert m, f'Incorrect charset file. line #{i}: {line}'
                label = int(m.group(1)) + 1
                char = m.group(2)
                charset[label] = char
        return charset    
    
    def textToNum(self, text, length=None, padding=True, case_sensitive=False):
        if not case_sensitive:
            text = text.lower()
            
        length = length
        special=re.findall("&[a-z]+-[0-9a-z]+;", text)      
        n=len(special)
        """if padding:
            text = text + self.null_char * (length - (len(text)-n*9))"""
        if not case_sensitive:
            text = text.lower()
        
        labels=[]
        
        if n==0:
            labels = [self.char_to_label_radical[char] for char in text]
        else:
            specialnum=[]
            for i in range(len(special)):
                specialnum.append(self.char_to_label_radical[special[i]])
            text_spl=[]
            for i in range(len(special)):
                if i==0:
                    loc=text.find(special[i])
                    text_spl=[text[:loc],text[loc+10:]]
                    #text_spl=text.split(special[i])
                else:
                    loc=text_spl[i].find(special[i])
                    t=[text_spl[i][:loc],text_spl[i][loc+10:]]
                    #t=text_spl[i].split(special[i])
                    text_spl.pop()
                    for tch in t:
                        text_spl.append(tch)
            for i in range(len(text_spl)):
                for j in range(len(text_spl[i])):
                    num=self.char_to_label_radical[text_spl[i][j]]
                    labels.append(num)
                if i!=len(text_spl)-1:
                    labels.append(specialnum[i])
        
        if len(labels)>length-1:
            labels=labels[:(length-1)]
        
        if padding:
            t=length-len(labels)
            for i in range(t):
                labels.append(0)
            
        return labels
     
    @property
    def last_losses(self):
        return self.losses

    def _flatten(self, sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def _merge_list(self, all_res):
        if not isinstance(all_res, (list, tuple)):
            return all_res
        def merge(items):
            if isinstance(items[0], torch.Tensor): return torch.cat(items, dim=0)
            else: return items[0]
        res = dict()
        for key in all_res[0].keys():
            items = [r[key] for r in all_res]
            res[key] = merge(items)
        return res

    def _ce_loss(self, output, gt_labels, gt_lengths, idx=None, record=True):
        
        only_Character = False
        only_Radical = False
        only_Radical_alignment = False
        both_CharacterRadical = False
        if 'logits' in output.keys() and 'logits_radical' not in output.keys():
            only_Character = True
            #print('！！！only_Character！！！')
            
        if 'logits' not in output.keys() and 'logits_radical' in output.keys():
            if len(gt_labels[1][1])==961:
                only_Radical = True
                #print('！！！only_Radical！！！')
            elif len(gt_labels[1][1])==7935:
                only_Radical_alignment = True
                #print('！！！only_Radical_alignment！！！')
            else:
                #print('！！！label_length_wrong！！！')
                assert 1==0
        
        if 'logits' in output.keys() and 'logits_radical' in output.keys():
            both_CharacterRadical = True
            #print('！！！both_CharacterRadical！！！')
        
        #############################################################################
        if only_Character==True:
            #print('only_Character')
            
            loss_name = output.get('name')
            pt_logits, weight = output['logits'], output['loss_weight']

            assert pt_logits.shape[0] % gt_labels.shape[0] == 0
            iter_size = pt_logits.shape[0] // gt_labels.shape[0]
            if iter_size > 1:
                #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                gt_labels = gt_labels.repeat(3, 1, 1)
                gt_lengths = gt_lengths.repeat(3)
            flat_gt_labels = self._flatten(gt_labels, gt_lengths)
            flat_pt_logits = self._flatten(pt_logits, gt_lengths)

            nll = output.get('nll')
            if nll is not None:
                loss = self.ce(flat_pt_logits, flat_gt_labels, softmax=False) * weight
            else:
                loss = self.ce(flat_pt_logits, flat_gt_labels) * weight
        
        #############################################################################
        if only_Radical==True:
            #print('only_Radical')
            
            loss_name = output.get('name')
            pt_logits, weight = output['logits_radical'], output['loss_weight']

            assert pt_logits.shape[0] % gt_labels.shape[0] == 0
            iter_size = pt_logits.shape[0] // gt_labels.shape[0]
            if iter_size > 1:
                #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                gt_labels = gt_labels.repeat(3, 1, 1)
                gt_lengths = gt_lengths.repeat(3)
            flat_gt_labels = self._flatten(gt_labels, gt_lengths)
            flat_pt_logits = self._flatten(pt_logits, gt_lengths)

            nll = output.get('nll')
            if nll is not None:
                loss = self.ce(flat_pt_logits, flat_gt_labels, softmax=False) * weight
            else:
                loss = self.ce(flat_pt_logits, flat_gt_labels) * weight
                
        #############################################################################
        if only_Radical_alignment==True or both_CharacterRadical==True:
            
            labels=[]
            for i in range(len(gt_labels)):
                t=[]
                for j in range(len(gt_labels[i])):
                    r=gt_labels[i][j].cpu()
                    r=r.numpy().tolist()
                    if r.index(1)!=0:
                        t.append(r.index(1))
                labels.append(t)
            

            char_labels=[]
            for i in range(len(labels)):
                chars=""
                for j in range(len(labels[i])):
                    chars=chars+self.label_to_char[labels[i][j]]
                char_labels.append(chars)
                #print(chars)
            
            radical_labels=[]         
            
            radical_labels_div=[]     
            
            for i in range(len(char_labels)):
                lab=char_labels[i]
                radical_label=""
                radical_label_div=[]
                for ch in lab:
                    tmp = []
                    if ch not in self.radical.keys():
                        ch_radical=ch
                    else:
                        ch_radical=self.radical[ch]
                    for j in ch_radical:
                        radical_label=radical_label+j
                        tmp.append(j)
                    radical_label_div.append(tmp)
                    
                radical_labels.append(radical_label)
                radical_labels_div.append(radical_label_div)
                #print(radical_label)
                
             
            radical_weight = []
            for i in range(len(radical_labels_div)):
                tmp = []
                for j in range(len(radical_labels_div[i])):
                    Tree = radical_tree(2)
                    idx = [0]
                    nowTree = [Tree]
                    get_radical_tree(radical_labels_div[i][j], idx, nowTree)

                    treeWeight = []
                    get_tree_weight(Tree, 1, treeWeight)
                    
                    #lra = 1/len(radical_labels_div[i][j])
                    #le = len(radical_labels_div[i][j])
                    p = 0.5
                    for k in range(len(treeWeight)):
                        tmp.append([ p*(1 + treeWeight[k]) ])
                    #tmp.append(treeWeight)
                
                l1 = len(tmp)
                l2 = self.max_length_radical+1
                if l1 > l2-1:
                    tmp = tmp[:(l2-1)]
                
                l1 = len(tmp)
                t = l2-l1
                for v in range(t):
                    tmp.append([1])
                
                radical_weight.append(tmp)
                
            
            num_radical_labels=[]
            num_radical_labels_nopadding=[]
            for i in range(len(radical_labels)):
                text=radical_labels[i].lower()
                num_radical_label=self.textToNum(text, length=self.max_length_radical+1, case_sensitive=True)
                num_radical_labels.append(num_radical_label)
                #print(num_radical_label)

                num_radical_label_nopadding=self.textToNum(text, length=self.max_length_radical+1, padding=False,case_sensitive=True)
                num_radical_labels_nopadding.append(num_radical_label_nopadding)
            
            gt_lengths_radical=[]            
            for i in range(len(num_radical_labels_nopadding)):
                gt_lengths_radical.append(len(num_radical_labels_nopadding[i])+1)
            gt_lengths_radical=torch.tensor(gt_lengths_radical).to(dtype=torch.long)
            #print(gt_lengths_radical)
            
            
            
            for i in range(len(num_radical_labels)):   
                tgt = onehot(num_radical_labels[i], self.num_classes_radical)    
                
                nowWeight = torch.Tensor(radical_weight[i])   
                tgt = tgt * nowWeight
               
                tgt = tgt.unsqueeze(0)      
                
                if i==0:
                    gt_labels_radical=tgt
                else:
                    gt_labels_radical=torch.cat((gt_labels_radical,tgt),0)
            gt_labels_radical=gt_labels_radical.cuda()


            #######################################################################################
            if only_Radical_alignment==True:
                #print('only_Radical')
            
                loss_name = output.get('name')
                pt_logits_radical, weight = output['logits_radical'], output['loss_weight']

                assert pt_logits_radical.shape[0] % gt_labels_radical.shape[0] == 0
                iter_size = pt_logits_radical.shape[0] // gt_labels_radical.shape[0]
                if iter_size > 1:
                    #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    gt_labels_radical = gt_labels_radical.repeat(3, 1, 1)
                    gt_lengths_radical = gt_lengths_radical.repeat(3)
                flat_gt_labels_radical = self._flatten(gt_labels_radical, gt_lengths_radical)
                flat_pt_logits_radical = self._flatten(pt_logits_radical, gt_lengths_radical)

                nll = output.get('nll')
                if nll is not None:
                    loss = self.ce(flat_pt_logits_radical, flat_gt_labels_radical, softmax=False) * weight
                else:
                    loss = self.ce(flat_pt_logits_radical, flat_gt_labels_radical) * weight
            
            
            #######################################################################################
            if both_CharacterRadical==True:          
                #print('both_CharacterRadical')  
                
                loss_name = output.get('name')
                pt_logits, weight = output['logits'], output['loss_weight']
                #!
                pt_logits_radical = output['logits_radical']

                assert pt_logits.shape[0] % gt_labels.shape[0] == 0
                iter_size = pt_logits.shape[0] // gt_labels.shape[0]
                if iter_size > 1:
                    gt_labels = gt_labels.repeat(3, 1, 1)
                    gt_lengths = gt_lengths.repeat(3)
                flat_gt_labels = self._flatten(gt_labels, gt_lengths)
                flat_pt_logits = self._flatten(pt_logits, gt_lengths)


                #!
                assert pt_logits_radical.shape[0] % gt_labels_radical.shape[0] == 0
                iter_size_radical = pt_logits_radical.shape[0] // gt_labels_radical.shape[0]
                if iter_size_radical > 1:          
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    gt_labels_radical = gt_labels_radical.repeat(3, 1, 1)
                    gt_lengths_radical = gt_lengths_radical.repeat(3)
                flat_gt_labels_radical = self._flatten(gt_labels_radical, gt_lengths_radical)
                flat_pt_logits_radical = self._flatten(pt_logits_radical, gt_lengths_radical)
                

                nll = output.get('nll')
                if nll is not None:
                    loss = self.ce(flat_pt_logits, flat_gt_labels, softmax=False) * weight
                    #!
                    loss2 = self.ce(flat_pt_logits_radical, flat_gt_labels_radical, softmax=False) * weight
                    loss = loss1 + loss2
                    #print('！！！！！get loss1+loss2 nll is not None！！！！！')
                else:        
                    loss1 = self.ce(flat_pt_logits, flat_gt_labels) * weight
                    #!
                    loss2 = self.ce(flat_pt_logits_radical, flat_gt_labels_radical) * weight
                    loss = loss1 + loss2
                    #print('！！！！！get loss1+loss2！！！！！')
        
        
        if record and loss_name is not None: self.losses[f'{loss_name}_loss'] = loss

        return loss

    def forward(self, outputs, *args):
        self.losses = {}
        if isinstance(outputs, (tuple, list)):
            outputs = [self._merge_list(o) for o in outputs]
            
            return sum([self._ce_loss(o, *args) for o in outputs if o['loss_weight'] > 0.])
        else:             
            
            return self._ce_loss(outputs, *args, record=False)


class SoftCrossEntropyLoss(nn.Module):      
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, softmax=True):
        if softmax: log_prob = F.log_softmax(input, dim=-1)
        else: log_prob = torch.log(input)
        
        loss = -(target * log_prob).sum(dim=-1)
        if self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum": return loss.sum()
        else: return loss
