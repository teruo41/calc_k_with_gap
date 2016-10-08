#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2016 Teruo Tanimoto
# MIT License
#
# calc_k_with_gap
#   A python implementation of an algorithm to finding k for k-means clustering
# 
# Gap statistic defined in
# Tibshirani, Walther, Hastie:
#  Estimating the number of clusters in a data set via the gap statistic
#  J. R. Statist. Soc. B (2001) 63, Part 2, pp 411-423
#
# A part of this file is from gap.py of Mikael Vejdemo-Johansson
# https://gist.github.com/michiexile/5635273

# import packages
from sklearn.cluster import KMeans
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
dst = scipy.spatial.distance.euclidean

def calc_k_with_gap(data, refs=None, nrefs=20, max_k=10):
    """
    Compute the Gap statistic for an nxm dataset in data.
    Find k value for K-Means clustering using gap statistic
    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of data.

    Input:
        data: A (n,m) scipy array. dataset for clustering
        refs: A precomputed set of reference distributions to calculate gap statistic (optional)
        nrefs: The number of reference distibutions. This value is used when refs is None.
        max_k: Maximum k to be searched

    Output:
        k: Founded k (If k == max_k, k is not a solution.)
        gaps: A list of gap statistics. Note that gaps[0] is gap_1.
        sks: A list of sk values. Note that sks[0] is sk_1.
    """
    print("calc_k_with_gap() preparing...")
    shape = data.shape
    if refs==None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))
	
        rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i]*dists+bots
    else:
        rands = refs

    print("calc_k_with_gap() preparing...   Done.")

    gaps = scipy.zeros(max_k)
    sks  = scipy.zeros(max_k)
    for (i,k) in enumerate(range(1, max_k)):
        print("calc_k_with_gap() iteration for i = ", i)
        kmeans_model = KMeans(n_clusters=k).fit(data)
        kmc = kmeans_model.cluster_centers_
        kml = kmeans_model.labels_
        disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(shape[0])])

        refdisps = scipy.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            kmeans_model = KMeans(n_clusters=k).fit(rands[:,:,j])
            kmc = kmeans_model.cluster_centers_
            kml = kmeans_model.labels_
            refdisps[j] = sum([dst(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])

        w_bar = scipy.mean(scipy.log(refdisps))
        sks[i] = scipy.sqrt(scipy.mean((scipy.log(refdisps) - w_bar) ** 2))
        gaps[i] = w_bar - scipy.log(disp)

        if i > 0:
            if gaps[i-1] > gaps[i] - sks[i]:
                break

    return k, gaps, sks
