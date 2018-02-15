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
from sklearn.cluster import KMeans

import argparse
#from mock.mock import inplace

if "win" in sys.platform:
    sys.path.append('\\\\AFS\\cern.ch\\work\\e\\efol\\public\\Beta-Beat.src\\')
else:
    sys.path.append('/afs/cern.ch/work/e/efol/public/Beta-Beat.src/')
from Utilities import tfs_pandas
from model.accelerators import lhc

LOGGER = logging.getLogger(__name__)

#WANTED_COLS_X = ["PK2PK", "CORMS", "NOISE", "TUNEX", "AMPX", "AVG_AMPX", "AVG_MUX", "NATTUNEX", "NATAMPX", "PH_ADV"]


WANTED_COLS_X = ["PK2PK", "TUNEX", "AMPX", "PH_ADV", "PH_ADV_BEAT", "PH_ADV_MDL", "BETX"]


def get_common_bad_bpms(files, twiss):
    common_bad_bpms = []
    all_data = []
    files_list = files.split(',')
    for _ in range(3):
        all_data, bad_from_iteration = get_all_bad_bpms(files, twiss)
        print(bad_from_iteration)
        if len(common_bad_bpms) == 0:
            common_bad_bpms = common_bad_bpms + bad_from_iteration
        else:
            common_bad_bpms = list(set(common_bad_bpms).intersection(bad_from_iteration))
    print("---Bad BPMs after 10 interrations---")
    for bpm in common_bad_bpms:
        print(bpm)
    outpath = os.path.abspath(os.path.join(os.path.dirname(files_list[0]), "..", "bad_bpms_clustering.txt"))
    print("Printing bad bpms to: {}".format(outpath))
    with open(outpath, "w") as outfile:
        outfile.write("\n".join(common_bad_bpms))
    _plot_bad_bpms_ph(all_data, common_bad_bpms)
    return common_bad_bpms


def _plot_bad_bpms_ph(all_data, bad_bpm_names):
    bpms_to_plot = all_data.iloc[bad_bpm_names, :]
    bpms_to_plot.PH_ADV_BEAT.plot()
    plt.title("Phase advance beating")
    plt.show()


def get_all_bad_bpms(files, twiss):
    bad_bpms = []
    all_data = []
    files_list = files.split(',')
    for file_in in files_list:
        all_data, bad_in_irs_single_file, bad_in_arcs_single_file = clustering(file_in, twiss, 0.3, 70)
        bad_bpms = bad_bpms + bad_in_arcs_single_file
        bad_bpms = bad_bpms + bad_in_irs_single_file
    bad_bpms = list(set(bad_bpms))
#     outpath = os.path.abspath(os.path.join(os.path.dirname(files_list[0]), "..", "bad_bpms_clustering.txt"))
#     print("Printing bad bpms to: {}".format(outpath))
#     with open(outpath, "w") as outfile:
#         outfile.write("\n".join(bad_bpms))
    return all_data, bad_bpms
    

def clustering(file, twiss, eps, minSamples):
    #read bpm data
    bpm_tfs_data = _create_columns(file, twiss)

    #data_for_clustering = bpm_tfs_data[["TUNEX", "PH_ADV_BEAT"]].copy()
    #print(data_for_clustering.shape[0])
    ir_bpm_data_for_clustering = bpm_tfs_data.iloc[~lhc.Lhc.get_arc_bpms_mask(bpm_tfs_data.index)]
    ir_bpm_data_for_clustering.loc[:, "TUNEX"] = _normalize_parameter(ir_bpm_data_for_clustering.loc[:, "TUNEX"])
    ir_bpm_data_for_clustering.loc[:, "AMPX"] = _normalize_parameter(ir_bpm_data_for_clustering.loc[:, "AMPX"])
    ir_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"] = _normalize_parameter(ir_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"])
    ir_tune_ph = ir_bpm_data_for_clustering[["TUNEX", "PH_ADV_BEAT"]].copy()
    ir_tune_amp = ir_bpm_data_for_clustering[["TUNEX", "AMPX"]].copy()
    ir_ph_amp = ir_bpm_data_for_clustering[["PH_ADV_BEAT", "AMPX"]].copy()
    labels_irs_kmeans_tune_ph = _kmeans_clustering(ir_tune_ph, 3, 100, "PH_ADV_BEAT", "TUNEX")
    labels_irs_kmeans_tune_amp = _kmeans_clustering(ir_tune_amp, 3, 100, "TUNEX", "AMPX")
    labels_irs_kmeans_ph_amp = _kmeans_clustering(ir_ph_amp, 3, 100, "PH_ADV_BEAT", "AMPX" )
    limit_for_good_cluster_ir = ir_bpm_data_for_clustering.shape[0] * 0.1
    bad_in_irs_kmeans = []
    bad_ir_tune_ph = _get_bad_clusters_from_dataset(ir_tune_ph, labels_irs_kmeans_tune_ph, limit_for_good_cluster_ir)
    bad_ir_tune_amp = _get_bad_clusters_from_dataset(ir_tune_amp, labels_irs_kmeans_tune_amp, limit_for_good_cluster_ir)
    bad_ir_ph_amp = _get_bad_clusters_from_dataset(ir_ph_amp, labels_irs_kmeans_ph_amp, limit_for_good_cluster_ir)
#     bad_ir_tune_amp = (ir_bpm_data_for_clustering.loc[bad_ir_tune_amp_all, :]
#                                                 .loc[np.abs(ir_bpm_data_for_clustering.PH_ADV_BEAT) > 0.1, :]
#                                                 .index)
    
#     print("---Bad in the IRs---")
#     print("TUNE - PH")
#     for index in bad_ir_tune_ph:
#         print(index)
#     print("TUNE - AMP")
#     for index in bad_ir_tune_amp:
#         print(index)
#     print("PH - AMP")
#     for index in bad_ir_ph_amp:
#         print(index)
#     
# #     bad_in_irs_kmeans_buffer = list(set(bad_ir_tune_ph).intersection(bad_ir_tune_amp))
# #     bad_in_irs_kmeans = list(set(bad_in_irs_kmeans_buffer).intersection(bad_ir_ph_amp))
    bad_in_irs_kmeans = list(set(list(bad_ir_ph_amp) + list(bad_ir_tune_amp) + list(bad_ir_tune_ph)))
    print("---Bad in the IRS as list---")
    for bpm in bad_in_irs_kmeans:
        print(bpm)
        print(ir_bpm_data_for_clustering.loc[bpm, "PH_ADV_BEAT"])
        print(ir_bpm_data_for_clustering.loc[bpm, "TUNEX"])
        print(ir_bpm_data_for_clustering.loc[bpm, "AMPX"])
    
    
    arc_bpm_data_for_clustering = bpm_tfs_data.iloc[lhc.Lhc.get_arc_bpms_mask(bpm_tfs_data.index)]
    arc_bpm_data_for_clustering.loc[:, "TUNEX"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "TUNEX"])
    arc_bpm_data_for_clustering.loc[:, "AMPX"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "AMPX"])
    arc_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"])
    arc_tune_ph = arc_bpm_data_for_clustering[["TUNEX", "PH_ADV_BEAT"]].copy()
    arc_tune_amp = arc_bpm_data_for_clustering[["TUNEX", "AMPX"]].copy()
    arc_ph_amp = arc_bpm_data_for_clustering[["PH_ADV_BEAT", "AMPX"]].copy()
    #labels_from_arcs = _dbscan_clustering_noise(arc_bpm_data_for_clustering, eps, minSamples)
    labels_arcs_kmeans_tune_ph = _kmeans_clustering(arc_tune_ph, 3, 100, "PH_ADV_BEAT", "TUNEX")
    labels_arcs_kmeans_tune_amp = _kmeans_clustering(arc_tune_amp, 3, 100, "TUNEX", "AMPX")
    labels_arcs_kmeans_ph_amp = _kmeans_clustering(arc_ph_amp, 3, 100, "PH_ADV_BEAT", "AMPX" )
    limit_for_good_cluster_arc = arc_bpm_data_for_clustering.shape[0] * 0.10
    bad_in_arcs_kmeans = []
    bad_arc_tune_ph = _get_bad_clusters_from_dataset(arc_tune_ph, labels_arcs_kmeans_tune_ph, limit_for_good_cluster_arc)
    bad_arc_tune_amp = _get_bad_clusters_from_dataset(arc_tune_amp, labels_arcs_kmeans_tune_amp, limit_for_good_cluster_arc)
    #TODO: filter for bpms where(arc_bpm_data_for_clustering.PH_ADV_BEAT > 0.02)
#     bad_arc_tune_amp = (arc_bpm_data_for_clustering.loc[bad_arc_tune_amp_all, :]
#                                                    .loc[np.abs(arc_bpm_data_for_clustering.PH_ADV_BEAT) > 0.1, :]
#                                                    .index)
    bad_arc_ph_amp = _get_bad_clusters_from_dataset(arc_ph_amp, labels_arcs_kmeans_ph_amp, limit_for_good_cluster_arc)

#     print("---Bad in the arcs---")
#     print("TUNE - PH")
#     for index in bad_arc_tune_ph:
#         print(index)
#     print("TUNE - AMP")
#     for index in bad_arc_tune_amp:
#         print(index)
#     print("PH - AMP")
#     for index in bad_arc_ph_amp:
#         print(index)
#
    bad_in_arcs_kmeans = list(set(list(bad_arc_tune_ph) + list(bad_arc_ph_amp) + list(bad_arc_tune_amp)))
    print("---Bad in the arcs as list---")
    for bpm in bad_in_arcs_kmeans:
        print(bpm)
        print(arc_bpm_data_for_clustering.loc[bpm, "PH_ADV_BEAT"])
        print(arc_bpm_data_for_clustering.loc[bpm, "TUNEX"])
        print(arc_bpm_data_for_clustering.loc[bpm, "AMPX"])

    #ir_data_std_normalized = _get_standard_score_normalization(noise_samples_second_itr)
    #foc_data_std_normalized = _get_standard_score_normalization(cluster1_samples_second_itr)
    #defoc_data_std_normalized = _get_standard_score_normalization(cluster2_samples_second_itr)

    return bpm_tfs_data, bad_in_irs_kmeans, bad_in_arcs_kmeans

    #For dbscan
    #bad_in_arcs = data_for_clustering.iloc[np.where(labels_from_arcs == -1)]


def _get_bad_clusters_from_dataset(data, labels, limit):
    bad_bpms = []
    unique_labels = set(labels)
    for label in unique_labels:
        count_bpms_in_cluster = bpms_with_label(data, labels, label)
        if count_bpms_in_cluster < limit:
            bad_bpms.append(data.iloc[np.where(labels == label)].index)
    if len(bad_bpms) == 0:
        return []
    return bad_bpms[0]


def bpms_with_label(data, labels, label):
    bpms_with_label = data.iloc[np.where(labels == label)]
    return bpms_with_label.shape[0]

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


def _kmeans_clustering(data, n_clusters, n_init, x, y):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, max_iter=n_init, tol=0.01, algorithm='auto')
    kmeans.fit(data)
    labels = kmeans.labels_
    plot_kmeans(data, labels, x, y)
    return labels



COLORS = ("blue", "red", "green", "yellow", "pink", "black", "orange")

def plot_kmeans(data, labels, x, y):
    unique_labels = set(labels)
    for k, col in zip(unique_labels, COLORS[:len(unique_labels)]):
        class_member_mask = (labels == k)
        data_points = data.iloc[class_member_mask]
        plt.plot(
            data_points.loc[:, x],
            data_points.loc[:, y],
            'o',
            markerfacecolor=col,
            markeredgecolor='k',
            markersize=14,
        )
    plt.xlabel(x, fontsize = 25)    
    plt.ylabel(y,fontsize = 25)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.show()


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


def plot_compare_phase_advances(ph_adv_bpm_data):
    getphase_data = tfs_pandas.read_tfs('/afs/cern.ch/work/e/efol/public/betabeatGui/temp/2017-12-18/LHCB1/Results/notcleaned_measurementNotVeryGood/getphasex.out').set_index("NAME").loc[ph_adv_bpm_data.index]
    getllm_phase_column = getphase_data.loc[:, "PHASEX"]
    getllm_phase_column.plot()
    ph_adv_bpm_data.plot()
    plt.legend()
    plt.show()
    


def _create_columns(file, twiss):
    bpm_tfs_data = tfs_pandas.read_tfs(file).set_index("NAME")
    model_tfs_data = tfs_pandas.read_tfs(twiss).set_index("NAME").loc[bpm_tfs_data.index]

    ph_adv_bpm_data = (bpm_tfs_data.loc[:, "MUX"] - np.roll(bpm_tfs_data.loc[:, "MUX"], 1)) % 1.
    ph_adv_model = (model_tfs_data.loc[:, "MUX"] - np.roll(model_tfs_data.loc[:, "MUX"], 1)) % 1.

    bpm_tfs_data.loc[:, "PH_ADV_MDL"] = ph_adv_model

    bpm_tfs_data.loc[:, "PH_ADV_BEAT"] = (ph_adv_bpm_data - ph_adv_model)
    bpm_tfs_data.PH_ADV_BEAT[bpm_tfs_data.PH_ADV_BEAT > 0.5] = 1 - bpm_tfs_data.PH_ADV_BEAT[bpm_tfs_data.PH_ADV_BEAT > 0.5]

    bpm_tfs_data.loc[:, "PH_ADV"] = ph_adv_bpm_data
    bpm_tfs_data.loc[:, "BETX"] = model_tfs_data.loc[:, "BETX"]
    return bpm_tfs_data


def _normalize_parameter(column_data):
    return (column_data - column_data.min()) / (column_data.max() - column_data.min())


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
    


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        dest="files", type=str,
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
    return options.files, options.twiss, options.eps, options.minSamples
    
if __name__ == "__main__":
    _files, _twiss, _eps, _minSamples = _parse_args()
    #get_all_bad_bpms(_files, _twiss)
    get_common_bad_bpms(_files, _twiss)
    #clustering(_file, _twiss, _eps, _minSamples)
