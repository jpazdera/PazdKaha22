import copy
import itertools as it
import json

ll = 24

all_permutations_adjacent = []
for i in range(ll-11):  # Choose first occurrence of adjacent grouping (first adjacent grouping must occur with at least nb*2 spaces remaining in the list)
    for j in range(i+2, ll-9):  # Choose location of second adjacent grouping
        for k in range(j+2, ll-7):  # Choose location of third adjacent grouping
            for l in range(k+2, ll-5):  # Choose location of fourth adjacent grouping
                for m in range(l+2, ll-3):  # Choose location of fifth adjacent grouping
                    for n in range(m+2, ll-1):  # Choose location of sixth adjacent grouping
                        p = [None for x in range(ll)]
                        p[i] = 'a0'
                        p[i + 1] = 'a1'
                        p[j] = 'b0'
                        p[j + 1] = 'b1'
                        p[k] = 'c0'
                        p[k+1] = 'c1'
                        p[l] = 'd0'
                        p[l+1] = 'd1'
                        p[m] = 'e0'
                        p[m+1] = 'e1'
                        p[n] = 'f0'
                        p[n+1] = 'f1'
                        all_permutations_adjacent.append(p)

all_permutations_both = []
for n in range(len(all_permutations_adjacent)):
    for i in range(ll-11):
        if all_permutations_adjacent[n][i] is None:  # The first word in a distant pair must occupy the first open list position
            for j in range(i + 1, ll - 9):  # Choose start location of second distant grouping
                for k in range(j + 1, ll - 7):  # Choose start location of third distant grouping
                    for l in range(k + 1, ll - 5):  # Choose start location of fourth distant grouping
                        for m in range(l + 1, ll - 3):  # Choose start location of fifth distant grouping
                            for n in range(m + 1, ll - 1):  # Choose start location of sixth distant grouping
                                p = copy.deepcopy(all_permutations_adjacent[n])
                                if p[i] is None and p[j] is None and p[k] is None:
                                    p[i] = 'g0'
                                    p[j] = 'h0'
                                    p[k] = 'i0'
                                    p[l] = 'j0'
                                    p[m] = 'k0'
                                    p[n] = 'l0'
                                    all_permutations_both.append(p)
            break

valid = []
for n in range(len(all_permutations_both)):
    nones = [i for i, x in enumerate(all_permutations_both[n]) if x is None]
    assignments = [list(zip(nones, p)) for p in it.permutations(['g1', 'h1', 'i1', 'j1', 'k1', 'l1'])]
    for a in assignments:
        p = copy.deepcopy(all_permutations_both[n])
        for ind, item in a:
            p[ind] = item
        if (p.index('g1') - p.index('g0') > 2) and (p.index('h1') - p.index('h0') > 2) and (p.index('i1') - p.index('i0') > 2) and \
                (p.index('j1') - p.index('j0') > 2) and (p.index('k1') - p.index('k0') > 2) and (p.index('l1') - p.index('l0') > 2):
            valid.append(p)

for p in valid:
    print p

print 'Solutions: %i' % len(valid)

with open('valid24.json', 'w') as f:
    json.dump({'3bin-valid24': valid}, f)
