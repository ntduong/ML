# Weighted random selection
import random
from bisect import bisect

class WeightedRandomSelector(object):
    '''
    Simple implementation of weighted ramdom selection algorithm.
    '''
    def __init__(self, weights):
        self.cumWeights = []
        tmp = 0
        for w in weights:
            tmp += w
            self.cumWeights.append(tmp)
            
    def next(self):
        rnd = random.random() * self.cumWeights[-1]
        return bisect(self.cumWeights, rnd)
        
    def __call__(self):
        return self.next()
    
if __name__ == '__main__':
    weights = [0.1, 0.2, 0.3, 0.4]
    selector = WeightedRandomSelector(weights)
    print selector()
    