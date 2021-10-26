"""cluster.py: Processes feature vectors clusters."""

__author__ = "Leo Hatvani"
__copyright__ = "Copyright 2018"
__credits__ = ["Leo Hatvani"]
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Leo Hatvani"
__email__ = "leo@hatvani.org"
__status__ = "Prototype"

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import csv
# from pprint import pprint
# import seaborn as sns
# import matplotlib as mpl
# import copy
# import glob
# import re
# from time import clock

import argparse
import csv
import scipy.sparse.csgraph as csgraph
from sklearn.preprocessing import normalize
import numpy as np
import os
import hdbscan
from itertools import combinations
import random
import skfuzzy
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil, floor, inf

parser = argparse.ArgumentParser()
parser.add_argument('--vectors', default="", help='path to file that contains feature vectors')
parser.add_argument('--dependencies', default="", help='path to file that describes the dependency graph')
parser.add_argument('--method', default="hdbscan", help='method for clustering: hdbscan (default) or fcm')
parser.add_argument('--nclusters', default=45, type=int, help='number of clusters for FCM, ignored for HDBSCAN')
args, unknown = parser.parse_known_args()
fname_dep = getattr(args,"dependencies")
fname_vec = getattr(args,"vectors")
opt_method = getattr(args, "method")
opt_nclusters = getattr(args, "nclusters")

# Load the feature vectors

label2req = {} # Contains the label to requirement dependencies
req2req = {} # Contains the dependencies of requirements to requirements
labels = [] # A list of labels corresponding to values
values = [] # A list of feature vectors
reqs = set()

print("Loading feature vectors from {}".format(fname_vec))
with open(fname_vec, 'r') as csvfile:
    datasetreader = csv.reader(csvfile)
    next(datasetreader) # Skip header
    for row in datasetreader:
        labels.append(row[0])
        values.append(row[1:])
        label2req[row[0]] = set()

# Load the requirement-test-case graph

print("Loading dependencies from {}".format(fname_dep))
with open(fname_dep, 'r') as csvfile:
    dsreader = csv.DictReader(csvfile)
    for row in dsreader:
        if row['Item'][0:2] == "TC":
            assert(row['Item'] in labels)
            label2req[row['Item']].add(row['DependsOn'])
            reqs.add(row['DependsOn'])
        elif row['Item'][0:3] == "REQ":
            if row['Item'] not in req2req:
                req2req[row['Item']] = set()
            req2req[row['Item']].add(row['DependsOn'])
            reqs.add(row['Item'])
            reqs.add(row['DependsOn'])
        else:  
            print("Error, unrecognized data in the input file: {}".format(row['Item']))
            assert(False)

def floyd_warshall(labels, reqs, label2req, req2req):
    retval = {}
    # Constructing the graph as L1 L2 L3 ... LN R1 R2 ...
    offset = len(labels)
    size = len(labels)+len(reqs)
    graph = np.zeros((size,size))
    for i in range(0, len(labels)):
        for j1 in range(0, len(reqs)):
            j = j1 + offset
            assert i!=j
            if reqs[j1] in label2req[labels[i]]:
                graph[i][j] = 1
    for i1 in range(0, len(reqs)):
        i = i1+offset
        for j in range(0, len(labels)):
            assert i!=j
            if reqs[i1] in label2req[labels[j]]:
                graph[i][j] = 1
        for j1 in range(0,len(reqs)):
            j = j1+offset
            if reqs[i1] in req2req[reqs[j1]] or reqs[j1] in req2req[reqs[i1]]:
                graph[i][j] = 1

    sparse_graph = csgraph.csgraph_from_dense(graph)

    dist_matrix = csgraph.floyd_warshall(sparse_graph, directed=False, unweighted=True)
    independent_labels = set()
    for i in range(0, len(labels)):
        independent = True
        for j in range(0, len(labels)):
            if i != j:
                if not np.isinf(dist_matrix[i][j]):
                    retval[(labels[i],labels[j])] = dist_matrix[i][j]
                    independent = False
        if independent:
            independent_labels.add(labels[i])
    return retval, independent_labels

for r in reqs:
    if r not in req2req:
        req2req[r] = set()
label2label_distance, independent_labels = floyd_warshall(labels, list(reqs), label2req, req2req)

print("Found {} independent labels from ground truth".format(len(independent_labels)))

# Normalize everything using L2 normalization 
values = normalize(np.asarray(values), norm='l2')

# Create results directory if it does not exist
if (not os.path.isdir("results/")):
    os.mkdir("results")

# Clustering

if opt_method=='hdbscan':
    clusterer = hdbscan.HDBSCAN().fit(values)
    mylabels = clusterer.labels_
elif opt_method == 'fcm':
    values_rotated = np.rot90(values)
    cntr, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(values_rotated, opt_nclusters, 2, error=0.005, maxiter=1000, init=None)
    mylabels = np.argmax(u, axis=0) 
else:
    print("Clustering method unknown")
    exit()

myindependent = set()
for i in range(0,len(mylabels)):
    if mylabels[i] == -1:
        myindependent.add(labels[i])

print("Found {} independent labels from {}".format(len(myindependent), opt_method))
print("There are {} independent labels in common".format(len(independent_labels.intersection(myindependent))))


assert len(mylabels) == len(labels)
label2cluster = {}
for i in range(0,len(mylabels)):
    label2cluster[labels[i]] = mylabels[i]

label2label_random = {}
for comb in combinations(labels, 2):
    label2label_random[comb] = random.randint(0,1)
    label2label_random[comb[::-1]] = label2label_random[comb]


def write_results(file_name):
    resoutfile = open(file_name,'w')
    resout = csv.writer(resoutfile)
    resout.writerow(['Distance', 
                     opt_method+' Yes, GT Yes', 
                     opt_method+' No, GT No', 
                     opt_method+' Yes, GT No', 
                     opt_method+' No, GT Yes', 
                     'Rnd Yes, GT Yes', 
                     'Rnd No, GT No', 
                     'Rnd Yes, GT No', 
                     'Rnd No, GT Yes'])
    for distance in range(1,23):
        real_matches = [0,0,0,0]
        rand_matches = [0,0,0,0]
        for comb in combinations(labels, 2):
            if label2cluster[comb[0]] == -1 or label2cluster[comb[1]] == -1:
                # Treat any unclustered data as disconnected
                if comb in label2label_distance and label2label_distance[comb]<=distance:
                    real_matches[3] += 1
                else:
                    real_matches[1] += 1                
            else:
                # Else check if it is connected in the ground truth
                if comb in label2label_distance and label2label_distance[comb]<=distance:
                    if label2cluster[comb[0]] == label2cluster[comb[1]]:
                        real_matches[0] += 1
                    else:
                        real_matches[3] += 1
                else:
                    if label2cluster[comb[0]] == label2cluster[comb[1]]:
                        real_matches[2] += 1
                    else:
                        real_matches[1] += 1

            #Random comparison
            if comb in label2label_distance and label2label_distance[comb]<=distance:
                if label2label_random[comb]:
                    rand_matches[0] += 1
                else:
                    rand_matches[3] += 1
            else:
                if label2label_random[comb]:
                    rand_matches[2] += 1
                else:
                    rand_matches[1] += 1
        resout.writerow([distance] + real_matches + rand_matches)

def write_baseline(file_name):
    resoutfile = open(file_name,'w')
    resout = csv.writer(resoutfile)
    resout.writerow(['Distance', 'Dependent', 'Not dependent'])
    for distance in range(0,22):
        real_matches = [0,0]
        for comb in combinations(labels, 2):
            if comb in label2label_distance and label2label_distance[comb]<=distance:
                real_matches[0] += 1
            else:
                real_matches[1] += 1                
        resout.writerow([distance] + real_matches)

def get_tsne(_values):
    tsne = TSNE(n_components=2, random_state=0)
    return tsne.fit_transform(_values)


def my_palette(n):
    retval = []
    for i in range(0,  floor(n/2.0)):
        retval.append(mpl.colors.hsv_to_rgb( ( (1.0/floor(n/2))*i ,1,1) ))
    for i in range(0,  ceil(n/2.0)):
        retval.append(mpl.colors.hsv_to_rgb( ( (1.0/ceil(n/2))*i ,1,0.6) ))
    return retval

def clustered_graph(file_name, labels, label2cluster, X_tsne):
    sns.set_context('paper')
    sns.set_style('white')
    sns.set_color_codes()

    plot_kwds={'alpha':0.75, 's':40, 'linewidths':0}
    fig = plt.figure()
    fig.set_dpi(72)
    fig.set_size_inches(6, 6)
    ax = fig.add_subplot(111)

    sns.despine(top=True, bottom=True, left=True, right=True)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add the grey color at the end (corresponding to -1 cluster index) to denote the unclustered.
    maxcolors = ceil((max(label2cluster.values())+1)/3)
    pal = my_palette(maxcolors) 

    fltrdX = []
    fltrdY = []
    for i in range(0,len(labels)):
        if label2cluster[labels[i]] == -1:
            fltrdX.append(X_tsne[i, 0])
            fltrdY.append(X_tsne[i, 1])
    plt.scatter(fltrdX, fltrdY, c=mpl.colors.rgb2hex( (0.5,0.5,0.5) ), label="$C_u$ $(n={})$".format(len(fltrdX)), **plot_kwds)

    for cluster in range(0,max(label2cluster.values())+1):
        fltrdX = []
        fltrdY = []
        for i in range(0,len(labels)):
            if label2cluster[labels[i]] == cluster:
                fltrdX.append(X_tsne[i, 0])
                fltrdY.append(X_tsne[i, 1])
        marker = 'o'
        if (cluster >= maxcolors):
            marker = '^'
        if (cluster >= 2*maxcolors):
            marker = 'D'

        plt.scatter(fltrdX, fltrdY, c=mpl.colors.rgb2hex(pal[cluster%maxcolors]), marker=marker, label="$C_{{{}}}$ $(n={})$".format(cluster+1, len(fltrdX)), **plot_kwds)

    lgd = plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0., prop = {"size": 6})
    fig.tight_layout()
    plt.savefig(file_name, format="pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()

write_baseline('results/baseline.csv')
write_results('results/results.csv')

X_tsne = get_tsne(values)
clustered_graph('results/clustered.pdf', labels, label2cluster, X_tsne)
