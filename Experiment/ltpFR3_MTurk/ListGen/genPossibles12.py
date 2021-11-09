import copy
import itertools as it
import json

ll = 12

all_permutations_adjacent = []
for i in range(ll-5):  # Choose first occurrence of adjacent grouping (first adjacent grouping must occur with at least nb*2 spaces remaining in the list)
    for j in range(i+2, ll-3):  # Choose location of second adjacent grouping
        for k in range(j+2, ll-1):  # Choose location of third adjacent grouping
            p = [None for x in range(ll)]
            p[i] = 'a0'
            p[i + 1] = 'a1'
            p[j] = 'b0'
            p[j + 1] = 'b1'
            p[k] = 'c0'
            p[k+1] = 'c1'
            all_permutations_adjacent.append(p)

all_permutations_both = []
for n in range(len(all_permutations_adjacent)):
    for i in range(ll-5):
        if all_permutations_adjacent[n][i] is None:  # The first word in a distant pair must occupy the first open list position
            for j in range(i + 1, ll - 4):  # Choose start location of second distant grouping
                for k in range(j + 1, ll - 3):  # Choose start location of third distant grouping
                    p = copy.deepcopy(all_permutations_adjacent[n])
                    if p[i] is None and p[j] is None and p[k] is None:
                        p[i] = 'd0'
                        p[j] = 'e0'
                        p[k] = 'f0'
                        all_permutations_both.append(p)
            break

valid = []
for n in range(len(all_permutations_both)):
    nones = [i for i, x in enumerate(all_permutations_both[n]) if x is None]
    assignments = [list(zip(nones, p)) for p in it.permutations(['d1', 'e1', 'f1'])]
    for a in assignments:
        p = copy.deepcopy(all_permutations_both[n])
        for ind, item in a:
            p[ind] = item
        if (p.index('d1') - p.index('d0') > 2) and (p.index('e1') - p.index('e0') > 2) and (p.index('f1') - p.index('f0') > 2):
            valid.append(p)

for p in valid:
    print p

print 'Solutions: %i' % len(valid)

with open('valid12.json', 'w') as f:
    json.dump({'3bin-valid12': valid}, f)