# Weighted random selection
import random
from bisect import bisect

def select(i2w):
	items = i2w.keys()
	weights = i2w.values()
	tmp = 0
	csum = []
	for w in weights:
		tmp += w
		csum.append(tmp)
		
	id = bisect(csum, random.random() * csum[-1])
	return items[id]
	
def faster_select(i2w):
	weights = i2w.values()
	rnd = random.random() * sum(weights)
	for i, w in enumerate(weights):
		rnd -= w
		if rnd < 0:
			return i
			
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
	weights = [0.1]
	selector = WeightedRandomSelector(weights)
	print selector()
	