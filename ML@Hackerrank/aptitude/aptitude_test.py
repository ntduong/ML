'''
Created on 2013/11/02
@author: duong
'''

import numpy as np
from scipy.stats import spearmanr
from scipy.stats.stats import kendalltau

def solve_testcase(gpa, ts, method="s"):
    f = spearmanr if method == "s" else kendalltau
    scores = []
    for i, t in enumerate(ts):
        scores.append((f(gpa, t)[0], i+1))
    return sorted(scores, reverse=True)[0][1]

def solve_file(fname="aptitude/input.txt"):
    ans = []
    with open(fname, "r") as fin:
        T = int(fin.readline())
        for _ in xrange(T):
            N = int(fin.readline())
            gpa = map(float, fin.readline().strip().split())
            ts = []
            for _ in xrange(5):
                t = map(float, fin.readline().strip().split())
                ts.append(t)
            
            ans.append(solve_testcase(gpa, ts, method="s"))
    return ans
    
def solve():
    T = int(raw_input())
    for _ in xrange(T):
        N = int(raw_input())
        gpa = map(float, raw_input().strip().split())
        ts = []
        for _ in xrange(1, 5):
            t = map(float, raw_input().strip().split())
            ts.append(t)
            
        print solve_testcase(gpa, ts, method="s")
        
if __name__ == "__main__":
    ans = solve()   
    print ans

    true_ans = []
    with open("aptitude/output.txt", "r") as answers:
        for a in answers:
            true_ans.append(int(a.strip()))
    
    assert len(ans) == len(true_ans), "Something wrong!"
    print "True answers:\n", true_ans
    print "My answers:\n", ans
    print sum(map(lambda x,y: 1 if x==y else 0, ans, true_ans))/float(len(ans))