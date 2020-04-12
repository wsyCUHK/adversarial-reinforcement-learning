# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 17:30:00 2018

@author: neiltang
"""

'''
implements the Aho-Corasick algorithm
'''

#coding = utf-8

class ACTreeNode():
    
    def __init__(self, edge, is_root = False):
        
        self.is_root = is_root
        
        self.goto_map = dict()
        self.edge = edge
        self.fail_node_idx = 0 #fail to root by default
        self.label = None

      
    def goto(self, c):
        
        child = self.goto_map.get(c, None)
        
        if child is None:
            child = None if not self.is_root else 0
            
        return child
    
    
    def fail(self):
        
        return self.fail_node_idx
    
    
    def output(self):
        
        return self.label
    
    
    def set_label(self, lbl):
        
        self.label = lbl
        
        
    def set_child(self, c, next_idx):
        
        child_idx = self.goto_map.get(c, None)
        
        if child_idx is None:
            child = ACTreeNode(edge = c)
            self.goto_map[c] = next_idx
            
            return child

        else:
            
            return child_idx
    
    
    def set_fail(self, node_idx):
        
        self.fail_node_idx = node_idx
        
        
    def get_children(self):
        
        return self.goto_map.items()
        
        
class ACTree():
    
    def __init__(self):
        
        self.node_list = None
    
    
    def build(self, pattern_list):
        
        self.build_trie(pattern_list)
        
        self.build_fail()
        
        
    def match(self, t):
        
        assert self.node_list is not None, 'ACTree has not been built yet'
        res = list()
        
        node_cur = self.node_list[0]
        
        if t is None:
            return res
        
        for c in t:
            node_next = node_cur.goto(c)
            
            while node_next is None:
                node_cur = self.node_list[node_cur.fail()]
                node_next = node_cur.goto(c)
                
            node_cur = self.node_list[node_next]
            n = node_cur
            
            while not n.is_root:
                
                if n.output() is not None:
                    res.append(n.output())
                    
                n = self.node_list[n.fail()]
                
        return res                
        
        
    def build_trie(self, pattern_list):
        
        self.node_list = [ACTreeNode(edge = None, is_root = True)]
        next_idx = 1
        
        for pattern in pattern_list:
            
            if pattern is None or len(pattern) == 0:
                continue
            
            node = self.node_list[0]
            for c in pattern:
                
                node = node.set_child(c, next_idx)
                
                if isinstance(node, int):
                    
                    node = self.node_list[node]
                    
                else:
                    
                    self.node_list.append(node)
                    next_idx += 1
                
            node.set_label(pattern)
            
            
    def build_fail(self):
        
        queue = list()
        
        for _, node_idx in self.node_list[0].get_children():
            
            self.node_list[node_idx].set_fail(0)
            queue.append(node_idx)
            
        while len(queue) > 0:
            
            node_idx = queue.pop(0)
            node = self.node_list[node_idx]
            
            for edge, child_idx in node.get_children():
                
                child_node = self.node_list[child_idx]
                fail_node = self.node_list[node.fail()]
                fail_next = fail_node.goto(edge)
                while fail_next is None:
                    
                    fail_node = self.node_list[fail_node.fail()]
                    fail_next = fail_node.goto(edge)
                    
                child_node.set_fail(fail_next)
                queue.append(child_idx)
