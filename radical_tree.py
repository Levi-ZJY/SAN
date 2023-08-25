class radical_tree:
    def __init__(self,branch):
        self.ifLeaf = None
        self.branch = branch
        self.data = None
        self.lchild = None
        self.rchild = None
        self.mchild = None
    
    def getLchild(self,lchild):
        self.lchild.data = lchild
        
    def getRchild(self,rchild):
        self.rchild.data = rchild


        
structure = ['⿰', '⿱', '⿹', '⿻', '⿶', '⿺', '⿸', '⿵', '⿴', '⿷', '⿲', '⿳']

import re
def get_radical_tree(radical, idx, nowTree):   
    if idx[0] > len(radical)-1:
        return
    
    if radical[idx[0]] in structure:
        nowTree[0].ifLeaf = False
        nowTree[0].data = radical[idx[0]]
        
        if radical[idx[0]] in ['⿲', '⿳']:
            nowTree[0].branch = 3
            nowTree[0].lchild = radical_tree(2)
            nowTree[0].rchild = radical_tree(2)
            nowTree[0].mchild = radical_tree(2)
            lc = nowTree[0].lchild
            rc = nowTree[0].rchild
            mc = nowTree[0].mchild
            idx[0]+=1
            
            nowTree[0] = lc
            get_radical_tree(radical,idx,nowTree)
            
            nowTree[0] = mc
            get_radical_tree(radical,idx,nowTree)
            
            nowTree[0] = rc
            get_radical_tree(radical,idx,nowTree)
            
        else:
            nowTree[0].branch = 2
            nowTree[0].lchild = radical_tree(2)
            nowTree[0].rchild = radical_tree(2)
            lc = nowTree[0].lchild
            rc = nowTree[0].rchild
            idx[0]+=1
            
            nowTree[0] = lc
            get_radical_tree(radical,idx,nowTree)
            
            nowTree[0] = rc
            get_radical_tree(radical,idx,nowTree)
        
    else:
        nowTree[0].ifLeaf = True
        nowTree[0].data = radical[idx[0]]
        idx[0]+=1
        return


    
    
    
def get_tree_weight(Tree, nowWeight, treeWeight):
    
    if Tree.ifLeaf==True:
        treeWeight.append(nowWeight)
        return
    else:
        if Tree.branch==2:
            treeWeight.append(nowWeight/3)
            get_tree_weight(Tree.lchild, nowWeight/3, treeWeight)
            get_tree_weight(Tree.rchild, nowWeight/3, treeWeight)
            
        elif Tree.branch==3:
            treeWeight.append(nowWeight/4)
            get_tree_weight(Tree.lchild, nowWeight/4, treeWeight)
            get_tree_weight(Tree.mchild, nowWeight/4, treeWeight)
            get_tree_weight(Tree.rchild, nowWeight/4, treeWeight)
        else:
            print('???')
            print(Tree.data)
    

    
    
def get_tree_weight_leaves(Tree, treeWeight):
    
    if Tree.ifLeaf==True:
        treeWeight.append(1)
        return
    else:
        if Tree.branch==2:
            treeWeight.append(0)
            get_tree_weight_leaves(Tree.lchild, treeWeight)
            get_tree_weight_leaves(Tree.rchild, treeWeight)
            
        elif Tree.branch==3:
            treeWeight.append(0)
            get_tree_weight_leaves(Tree.lchild, treeWeight)
            get_tree_weight_leaves(Tree.mchild, treeWeight)
            get_tree_weight_leaves(Tree.rchild, treeWeight)
        else:
            print('???')
            print(Tree.data)