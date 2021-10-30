#*K means -> unsupervised learning.
#*k is the number of clusters.
#Put centroids in random locations.

#pretty complicated process...
#Repeat re-drawing centroids until no changes.

#Very slow.

import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
#*Remember andrew ng's scaling down.
data = scale(digits.data)
y = digits.target

k = 10
#NUmber of clusters.

#*Just calculates: shape just returns dimensions.
samples, features = data.shape

#Don't have to test. 
#*Unsupervised so automatically generates a y-value. 
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))
    
#Can choose random or kmeans++ (initial location of centers are equidistant or whatever)
#max_iter: number of iterations it will camp to. 
clf = KMeans(n_clusters=k, init="random", n_init=10)

bench_k_means(clf, "1", data)

