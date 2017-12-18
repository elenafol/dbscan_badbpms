from __future__ import print_function
import os
import sys
import numpy as np
import pandas
import logging
from scipy import stats
from sklearn import preprocessing
from time import time
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network.multilayer_perceptron import MLPClassifier, MLPRegressor
from sklearn.externals import joblib
from sklearn.cluster import DBSCAN
from sklearn.neighbors import DistanceMetric
from scipy.spatial import distance
import argparse
from mock.mock import inplace

if "win" in sys.platform:
    sys.path.append('\\\\AFS\\cern.ch\\work\\e\\efol\\public\\Beta-Beat.src\\')
else:
    sys.path.append('/afs/cern.ch/work/e/efol/public/Beta-Beat.src/')
from Utilities import tfs_pandas

LOGGER = logging.getLogger(__name__)

#WANTED_COLS_X = ["PK2PK", "CORMS", "NOISE", "TUNEX", "AMPX", "AVG_AMPX", "AVG_MUX", "NATTUNEX", "NATAMPX", "PH_ADV"]
WANTED_COLS_X = ["PK2PK", "TUNEX", "AMPX", "PH_ADV", "PH_ADV_BEAT", "PH_ADV_MDL", "BETX"]


def clustering(file, twiss, eps, minSamples):
    #read bpm data
    bpm_tfs_data = _create_columns(file, twiss)
    #cluster into IR, focusing, defocusing
    regions = bpm_tfs_data[["PH_ADV_MDL", "BETX"]].copy()
    labels = _dbscan_clustering_regions(regions, eps, minSamples)
    
    noise_samples = bpm_tfs_data.iloc[np.where(labels == -1)]
    cluster1_samples = bpm_tfs_data.iloc[np.where(labels == 1)]
    cluster2_samples = bpm_tfs_data. iloc[np.where(labels == 0)]
    print("Number of all bpms")
    print(noise_samples.shape[0])
    print(cluster1_samples.shape[0])
    print(cluster2_samples.shape[0])
    
    #labels = _dbscan_clustering(data_features, eps, minSamples)
    #noise_samples = data_features.iloc[np.where(labels == -1)]
    #cluster1_samples = data_features.iloc[np.where(labels == 1)]
    #cluster2_samples = data_features.iloc[np.where(labels == 0)]
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)
    _plot_beta_function(twiss, regions, labels)
    np.set_printoptions(threshold=np.nan)
    
    
    #_weighted_feature(noise_samples, "PK2PK", 10)
    #_weighted_feature(cluster1_samples, "PK2PK", 10)
    #_weighted_feature(cluster2_samples, "PK2PK", 10)

    
    
    noise_samples_second_itr = noise_samples[["PK2PK", "TUNEX", "AMPX"]].copy()
    cluster1_samples_second_itr = cluster1_samples[["PK2PK", "TUNEX", "AMPX"]].copy()
    cluster2_samples_second_itr = cluster2_samples[["PK2PK", "TUNEX", "AMPX"]].copy()
    
    #ir_data_std_normalized = _get_standard_score_normalization(noise_samples_second_itr)
    #foc_data_std_normalized = _get_standard_score_normalization(cluster1_samples_second_itr)
    #defoc_data_std_normalized = _get_standard_score_normalization(cluster2_samples_second_itr)
    
    #print(np.max(ir_data_std_normalized["PK2PK"]))
    #print(np.max(ir_data_std_normalized["TUNEX"]))
    #print(np.max(ir_data_std_normalized["AMPX"]))
    #print("---------------------------------")
    #print(np.min(ir_data_std_normalized["PK2PK"]))
    #print(np.min(ir_data_std_normalized["TUNEX"]))
    #print(np.min(ir_data_std_normalized["AMPX"]))
    
    
    labels_irs = _dbscan_clustering_noise(noise_samples_second_itr, 0.3, 10)
    labels_foc = _dbscan_clustering_noise(cluster1_samples_second_itr, 0.7, 10)
    labels_defoc = _dbscan_clustering_noise(cluster2_samples_second_itr, 0.7, 10)
    
    
    bad_in_irs = noise_samples.iloc[np.where(labels_irs == -1)]
    bad_in_foc = cluster1_samples.iloc[np.where(labels_foc == -1)]
    bad_in_defoc = cluster2_samples.iloc[np.where(labels_defoc == -1)]
    
    print("Number of bad BPMs not considering the phase advance-----------------")
    print(bad_in_irs.shape[0])
    print(bad_in_foc.shape[0])
    print(bad_in_defoc.shape[0])
    
    for index, row in bad_in_irs.iterrows():
        bad_in_irs.loc[index, "PH_ADV_BEAT"] = bpm_tfs_data.loc[index, "PH_ADV_BEAT"]
    for index, row in bad_in_foc.iterrows():
        bad_in_foc.loc[index, "PH_ADV_BEAT"] = bpm_tfs_data.loc[index, "PH_ADV_BEAT"]
    for index, row in bad_in_defoc.iterrows():
        bad_in_defoc.loc[index, "PH_ADV_BEAT"] = bpm_tfs_data.loc[index, "PH_ADV_BEAT"]

    
    bad_in_irs_ph_checked = bad_in_irs[bad_in_irs["PH_ADV_BEAT"] > 0.022]
    bad_in_foc_ph_checked = bad_in_foc[bad_in_foc["PH_ADV_BEAT"] > 0.022]
    bad_in_defoc_ph_checked = bad_in_defoc[bad_in_defoc["PH_ADV_BEAT"] > 0.022]
    
    
    print("---------Bad BPMs in IRs-----------------")
    print(bad_in_irs_ph_checked.shape[0])
    #print(bad_in_irs.index)
    for index, row in bad_in_irs_ph_checked.iterrows():
        print(index)
        print(bpm_tfs_data.loc[index, "PH_ADV_BEAT"])
        #print(row["PH_ADV_BEAT"])
    print("---------Bad BPMs in focusing region-----------------")
    print(bad_in_foc_ph_checked.shape[0])
    #print(bad_in_foc.index)
    for index, row in bad_in_foc_ph_checked.iterrows():
        print(index)
        print(bpm_tfs_data.loc[index, "PH_ADV_BEAT"])
        #print(row["PH_ADV_BEAT"])
        #_max_distance(bad_in_foc, row)
    print("---------Bad BPMs in defocusing region-----------------")
    print(bad_in_defoc_ph_checked.shape[0])
    #print(bad_in_defoc.index)
    for index, row in bad_in_defoc_ph_checked.iterrows():
        print(index)
        print(bpm_tfs_data.loc[index, "PH_ADV_BEAT"])
        #print(row["PH_ADV_BEAT"])
        #_max_distance(bad_in_defoc, row)


def _weighted_feature(data, feature, weight):
    column_data = data.loc[:, feature]
    weighted_column_data = column_data * weight
    data.loc[:, feature] = weighted_column_data   
    
 
def _max_distance(data_features, data_point):
    distances = []
    #average of data features per column
    for column in data_features.columns:
        column_mean = data_features[column].mean()
        distances.append(distance.mahalanobis(column_mean, data_point[column], np.cov(data_point)))
                         #(column_mean, data_point[column]))
    all_distances = pandas.DataFrame(distances, data_features.columns)
    #print("###Maximum distance###")
    #print(all_distances.max(axis=1))
    #all_distances.transpose()
    indx = all_distances.idxmax()
    #print(all_distances.idxmax())
    print(data_point[indx])
    

    #feature_with_max_dst = all_distances.idxmax(axis=1)
    #print(feature_with_max_dst)
    #print(all_distances[feature_with_max_dst])
    #print('###########') 
    

def _dbscan_clustering_regions(data, eps, minSamples):
    db = DBSCAN(eps=eps, min_samples=minSamples, metric='mahalanobis', metric_params={'V': np.cov(data)}, 
                        algorithm='brute', leaf_size=30, p=None, n_jobs=-1)
    prediction = db.fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    _plotting(data, core_samples_mask, labels, "PH_ADV_MDL", "BETX")
   
    return labels

def _dbscan_clustering_noise(data, eps, minSamples):
    #mahalanobis_metric = DistanceMetric.get_metric('mahalanobis', V=np.cov(data))
    db = DBSCAN(eps=eps, min_samples=minSamples, metric='mahalanobis', metric_params={'V': np.cov(data)}, 
                        algorithm='brute', leaf_size=30, p=None, n_jobs=-1)
    prediction = db.fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    _plotting(data, core_samples_mask, labels, "PK2PK", "TUNEX")
   
    return labels


COLORS = ("blue", "red", "green")

def _plotting(data_features, core_samples_mask, labels, xcolumn, ycolumn):
    print(data_features.columns)
    unique_labels = set(labels)
    for k, col in zip(unique_labels, COLORS[:len(unique_labels)]):
        if k == -1:  # Is noise
            col = "black"
     
        class_member_mask = (labels == k)
        this_class_core_mask = class_member_mask & core_samples_mask
        this_class_non_core_mask = class_member_mask & ~core_samples_mask
     
        core_points = data_features.iloc[this_class_core_mask]
        non_core_points = data_features.iloc[this_class_non_core_mask]
        plt.plot(
            core_points.loc[:, xcolumn],
            core_points.loc[:, ycolumn],
            'o',
            markerfacecolor=col,
            markeredgecolor='k',
            markersize=14,
            label = "Core samples $C_{" + str(k+1) +"}$" if k != -1 else "",
        )
        
        plt.plot(
            non_core_points.loc[:, xcolumn],
            non_core_points.loc[:, ycolumn],
            'o' if k == -1 else 's',
            markerfacecolor=col,
            markeredgecolor='k',
            markersize=6,
            label = "Noise" if k == -1 else "Non core samples $C_{" + str(k+1) +"}$",
        )

    plt.xlabel(xcolumn, fontsize = 25)    
    plt.ylabel(ycolumn,fontsize = 25)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.legend(fontsize = 25)
    plt.show()


def _plot_beta_function(twiss, data_features, labels):
    twiss = tfs_pandas.read_tfs(twiss).set_index("NAME")
    unique_labels = set(labels)
    for k, col in zip(unique_labels, COLORS[:len(unique_labels)]):
        if k == -1:  # Is noise
            col = "black"
     
        class_member_mask = (labels == k)
        class_members = data_features.iloc[class_member_mask].index
        
        plt.plot(
            twiss.loc[class_members, "S"],
            twiss.loc[class_members, "BETX"],
            'o',
            markerfacecolor=col)
    plt.show()


def _create_columns(file, twiss):
    bpm_tfs_data = tfs_pandas.read_tfs(file).set_index("NAME")
    model_tfs_data = tfs_pandas.read_tfs(twiss).set_index("NAME").loc[bpm_tfs_data.index]

    ph_adv_bpm_data = (bpm_tfs_data.loc[:, "AVG_MUX"] - np.roll(bpm_tfs_data.loc[:, "AVG_MUX"], 1)) % 1.
    ph_adv_model = (model_tfs_data.loc[:, "MUX"] - np.roll(model_tfs_data.loc[:, "MUX"], 1)) % 1.
    bpm_tfs_data.loc[:, "PH_ADV_MDL"] = ph_adv_model

    bpm_tfs_data.loc[:, "PH_ADV_BEAT"] = np.abs(ph_adv_model - ph_adv_bpm_data)
    
    bpm_tfs_data.loc[:, "PH_ADV"] = bpm_tfs_data.loc[:, "AVG_MUX"] - np.roll(bpm_tfs_data.loc[:, "AVG_MUX"], 1)
    bpm_tfs_data.loc[:, "BETX"] = model_tfs_data.loc[:, "BETX"]
    return bpm_tfs_data


def _get_scaled_features(bpm_tfs_data):
    for column in bpm_tfs_data.columns:
        if column not in WANTED_COLS_X:
            bpm_tfs_data.drop(column, axis=1, inplace=True)
            continue
        column_data = bpm_tfs_data.loc[:, column]
        norm_column_data = (column_data - column_data.min()) / (column_data.max() - column_data.min())
        bpm_tfs_data.loc[:, column] = norm_column_data
        bpm_tfs_data = bpm_tfs_data.dropna(axis=1)
    
    return bpm_tfs_data

def _get_standard_score_normalization(bpm_tfs_data):
    for column in bpm_tfs_data.columns:
        if column not in WANTED_COLS_X:
            bpm_tfs_data.drop(column, axis=1, inplace=True)
            continue
        column_data = bpm_tfs_data.loc[:, column]
        norm_column_data = ((column_data-np.mean(column_data))/np.std(column_data))
        bpm_tfs_data.loc[:, column] = norm_column_data
        bpm_tfs_data = bpm_tfs_data.dropna(axis=1)
    
    return bpm_tfs_data
    


def _normalize(bpms_data):
    
    return bpms_data

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        dest="file", type=str,
    )
    parser.add_argument(
        "--twiss",
        dest="twiss", type=str,
    )
    parser.add_argument(
        "--eps",
        dest="eps", type=float,
        help="Distance to data points in the neighborhood of a core sample.",
    )
    parser.add_argument(
        "--minSamples",
        dest="minSamples", type=float,
        help="Minimum number to data points in the neighborhood of a core sample.",
    )
    options = parser.parse_args()
    return options.file, options.twiss, options.eps, options.minSamples
    
if __name__ == "__main__":
    _file, _twiss, _eps, _minSamples = _parse_args()
    clustering(_file, _twiss, _eps, _minSamples)