"""
file       kruskal-wallis.py
 
author  Ernesto P. Adorio, Ph.D.
           UPEPP at Clark Field, Pampanga
"""
def isequalfloats(x,y, ztol= 1.0e-8):
    # returns True if both x and y are within ztol of each other.
    return True if (abs(x-y) <= ztol) else False

def kruskalwallis(D, ignoreties = True, ztol = 1.0e-8):
    """
    D is an array of arrays of values.
    ignoreties is False if tries are to be processed.
    ztol is zero tolerance, used for equalities.
    """
    nclasses = len(D)               # number of groups
    nj       = [float(len(d)) for d in D]  # number of items in each group
    print "nj=",nj
    n        = sum(nj)              # number of values in all groups
    Ranks    = []                   # ranks array.
 
    for i, d in enumerate(D):
        for x in d:
            Ranks.append((x, i))
    Ranks.sort()
    
    for i,r in enumerate(Ranks):
        print i, r
        
    Tj  = [0] * nclasses
    Tj2 = [0] * nclasses
       
    if ignoreties:
        for (i, r) in  enumerate(Ranks):
            j = r[1]
            Tj[j] += (i+1)          # sum of ranks  
            Tj2[j] = (i+1) * (i+1)  # sum of square of ranks.
            
        print Tj
    else:
        i = 0
        while i < n:
            start = i
            end   = i
            ranksum = i+1
            for j in range(i+1, n):
                if not isequalfloats(Ranks[j][0], Ranks[i][0]):
                    end = j-1
                    break
                else:
                    ranksum+= (j+1)
            if start == end:
                avgrank = i + 1
            else:
                avgrank = (start + end)/2.0 + 1.0
                
            print "avgrank = ", avgrank, "start = ",start, "end=", end
             
            for j in range(start, end+1):
                k = Ranks[j][1]
                Tj[k]  += (avgrank)
            i = end + 1
            
        print Tj
        print nj
        print n
 
        sumtj2 = sum([tj*tj/nj[i] for (i,tj) in enumerate(Tj)])
        print sumtj2
        
        KWstatistic = 12.0/(n*(n+1))* sumtj2 -3 * (n+1)
        
        return KWstatistic
 
 
    print "Tj:", Tj
