""" For testing recommendation program.
"""


from recommendations import *
from data import critics

itemsim = calculateSimilarItems(critics)
print itemsim