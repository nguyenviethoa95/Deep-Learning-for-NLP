from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pickle
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import euclidean

#raw_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)
#raw_model.init_sims(replace=True)
#raw_model.save("processed")

# Load pretrain word2vec model
model = KeyedVectors.load('processed',mmap="r")

# calculate the pearson coefficient 
def calculatePearson():
    pre_dist = []
    euclidDist = []

    with open("SimLex-999.txt", "r") as infile:
        for i,line  in enumerate(infile):
            if i != 0:
                pre_dist.append(float(line.split("\t")[3]))
    with open("eucliddistance.txt","rb") as infile:
        for line in infile:
            euclidDist.append(float(line))

    print(pearsonr(pre_dist,euclidDist))

# Calculate euclidean distance and save to a file 
with open("SimLex-999.txt","r") as infile:
     dist = []
     for line in infile:
         line = line.split("\t")
         try:
             vec1 = model.word_vec(str(line[0]))
         except:
             vec1 = np.zeros((300,1))
         try:
             vec2 = model.word_vec(str(line[1]))
         except:
             vec2 = np.zeros((300,1))

         dist.append(euclidean(vec1,vec2))

     with open("eucliddistance.txt","w") as outfile:
         for i in dist:
            outfile.write(str(i)+"\n")




