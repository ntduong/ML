'''
Created on Feb 21, 2013
@author: Administrator
'''
from collections import defaultdict
from math import log

def readDataFromFile(filename='decision_tree_example.txt'):
    with open(filename, 'rt') as f:
        data = []
        for line in f:
            data.append(line.strip().split('\t'))
        
    return data

def uniquecounts(rows):
    results = defaultdict(int)
    for row in rows:
        r = row[len(row)-1]
        results[r] += 1
        
    return results

def gini_impurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1])/total
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2])/total
            imp += p1*p2
            
    return imp

def entropy(rows):
    log2 = lambda x: log(x)/log(2)
    results = uniquecounts(rows)
    ent = 0.0
    total = len(rows)
    for r in results:
        p = float(results[r])/total
        ent -= p*log2(p)
        
    return ent

def divide_set(rows, col, value):
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[col] >= value
    else:
        split_function = lambda row: row[col] == value
        
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return (set1, set2)

class treenode(object):
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.fb = fb
        self.tb = tb
        
def buildtree(rows, score_function=entropy):
    if len(rows) == 0: return treenode()
    current_score = score_function(rows)
    
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_cnt = len(rows[0])-1 # excluding the last column
    for col in range(0, column_cnt):
        col_values = {}
        for row in rows:
            col_values[row[col]] = 1
        for value in col_values:
            (set1, set2) = divide_set(rows, col, value)
            len1 = len(set1)
            total = len(rows)
            p = float(len1)/total
            gain = current_score - p*score_function(set1) - (1-p)*score_function(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)
    if best_gain > 0:
        trueBranch = buildtree(best_sets[0], score_function)
        falseBranch = buildtree(best_sets[1], score_function)
        return treenode(col=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch)
    else:
        return treenode(results=uniquecounts(rows))
    
def print_tree(node, indent=''):
    if node.results != None:
        print str(node.results)
    else:
        print str(node.col) + ':' + str(node.value) + '?'
        print indent + 'T->',
        print_tree(node.tb, indent+' ')
        print indent + 'F->',
        print_tree(node.fb, indent+' ')
        
def getWidth(node):
    if node.tb == None and node.fb == None:
        return 1
    return getWidth(node.fb) + getWidth(node.tb)
        
def getHeight(node):
    if node.tb == None and node.fb == None:
        return 1
    return getHeight(node.tb) + getHeight(node.fb) + 1

from PIL import Image, ImageDraw

def drawNode(draw, node, x, y):
    if node.results == None:
        w1 = getWidth(node.fb)*100
        w2 = getWidth(node.tb)*100
        
        left = x-(w1+w2)/2
        right = x+(w1+w2)/2
        
        draw.text((x-20,y-10),str(node.col)+':'+str(node.value),(0,0,0))
        
        draw.line((x, y, left+w1/2, y+100), fill=(255,0,0))
        draw.line((x, y, right-w2/2, y+100), fill=(255,0,0))
        
        drawNode(draw, node.fb, left+w1/2, y+100)
        drawNode(draw, node.tb, right-w2/2, y+100)
    else:
        txt = ' \n'.join(['%s:%d' %v for v in node.results.items()])
        draw.text((x-20,y), txt, (0,0,0))

def drawTree(node, jpeg='tree.jpg'):
    w = getWidth(node)*100
    h = getHeight(node)*100+120
    
    img = Image.new('RGB', (w,h), (255,255,255))
    draw = ImageDraw.Draw(img)
    
    drawNode(draw, node, w/2, 20)
    img.save(jpeg, 'JPEG')
    
def classify(observation, node):
    if node.results != None:
        return node.results
    else:
        v = observation[node.col]
        branch = None
        if isinstance(v,int) or isinstance(v,float):
            if v >= node.value: branch = node.tb
            else: branch = node.fb
        else:
            if v == node.value: branch = node.tb
            else: branch = node.fb
        return classify(observation, branch) 
    
def prune(node, mingain):
    if node.tb.results == None:
        prune(node.tb, mingain)
    if node.fb.results == None:
        prune(node.fb, mingain)
        
    if node.tb.results != None and node.fb.results != None:
        tb, fb = [], []
        for v, c in node.tb.results.items():
            tb.extend([[v]]*c)
        for v, c in node.fb.results.items():
            fb.extend([[v]]*c)
            
        delta = entropy(tb+fb) - (entropy(tb) + entropy(fb))/2
        
        if delta < mingain:
            node.tb, node.fb = None, None
            node.results = uniquecounts(tb+fb)

def missing_value_classify(observation, node):
    if node.results != None:
        return node.results
    else:
        v = observation[node.col]
        if v == None:
            tr, fr = missing_value_classify(observation, node.tb), missing_value_classify(observation, node.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount)/(tcount+fcount)
            fw = 1-tw
            result = {}
            for k,v in tr.items():
                result[k] = v*tw
            for k,v in fr.items():
                if k not in result: result[k] = 0
                result[k] += v*fw
            return result
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= node.value: branch = node.tb
                else: branch = node.fb
            else:
                if v == node.value: branch = node.tb
                else: branch = node.fb
            return missing_value_classify(observation, branch)
        

if __name__ == '__main__':
    data = readDataFromFile()
    root = buildtree(data)
    print missing_value_classify(['google',None,'yes',None], root)