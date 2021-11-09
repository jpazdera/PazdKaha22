import scipy.io
import numpy as np

w2v = scipy.io.loadmat('/Users/jessepazdera/svn/word2vec/w2v.mat')['w2v']
with open('/Users/jessepazdera/PycharmProjects/ListGen/word_IDs.txt') as ind_file:
    ind = [int(x)-1 for x in ind_file.readlines()]

matrix = w2v[ind, :][:, ind]
np.savetxt('/Users/jessepazdera/PycharmProjects/ListGen/w2v_scores_ltpFR3.txt', matrix, fmt='%.4g')
