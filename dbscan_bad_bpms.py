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
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation

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
"""
python dbscan_bad_bpms.py --alg=dbscan --files=/home/efol/work/public/halfInteger_afterCorr/Beam1@Turn@2017_11_28@10_20_48_638/Beam1@Turn@2017_11_28@10_20_48_638.sdds.linx,/home/efol/work/public/halfInteger_afterCorr/Beam1@Turn@2017_11_28@10_28_07_290/Beam1@Turn@2017_11_28@10_28_07_290.sdds.linx,/home/efol/work/public/halfInteger_afterCorr/Beam1@Turn@2017_11_28@10_34_07_907/Beam1@Turn@2017_11_28@10_34_07_907.sdds.linx --twiss=/home/efol/work/public/halfInteger_afterCorr/model_B1_41_43/twiss_adt.dat --eps=0.7 --minSamples=70 --nClusters=6 --nInit=100

"""

WANTED_COLS_X = ["PK2PK", "TUNEX", "AMPX", "PH_ADV", "PH_ADV_BEAT", "PH_ADV_MDL", "BETX"]


def get_bad_bmps_from_kmeans(files_list, twiss, n_clusters, n_init):
    common_bad_bpms_files = []
    all_bad_bpms_files = []
    for file_in in files_list:
        common_bpms_from_iterrations = []
        for _ in range(3):
            bad_from_iteration = kmeans_iterration(file_in, twiss, n_clusters, n_init)
            all_bad_bpms_files = all_bad_bpms_files + bad_from_iteration
            if len(common_bpms_from_iterrations) == 0:
                common_bpms_from_iterrations = common_bpms_from_iterrations + bad_from_iteration
            else:
                common_bpms_from_iterrations = list(set(common_bpms_from_iterrations).intersection(bad_from_iteration))
        if len(common_bad_bpms_files) == 0:
                common_bad_bpms_files = common_bad_bpms_files + common_bpms_from_iterrations
        else:
            common_bad_bpms_files = list(set(common_bad_bpms_files).intersection(common_bpms_from_iterrations))
    all_bad_bpms_files = set(list(all_bad_bpms_files))
    return all_bad_bpms_files, common_bad_bpms_files
    
            
    # print("---Bad BPMs after 10 interrations---")
    # for bpm in common_bad_bpms:
        # print(bpm)
    # outpath = os.path.abspath(os.path.join(os.path.dirname(files_list[0]), "..", "bad_bpms_clustering.txt"))
    # print("Printing bad bpms to: {}".format(outpath))
    # with open(outpath, "w") as outfile:
        # outfile.write("\n".join(common_bad_bpms))
    # _plot_bad_bpms_ph(all_data, common_bad_bpms)


def separate_by_plane(files_list):
    files_x = []
    files_y = []
    for file_in in files_list:
        if "x" in os.path.abspath(file_in):
            files_x.append(file_in)
        elif "y" in os.path.abspath(file_in):
            files_y.append(file_in)
        else:
            print("Given file is not a measurement!")
    return files_x, files_y



""""
linx,liny files from all measurements (TODO: pass measurements dir, not files)
TODO: plane independent: first separate into x,y and then in irs and arcs
collect all IRs, all arcs
in case of dbscan - get labels and iloc
in case of kmeans - get bad bpms
plotting: each file separetely: plot phase_adv_beat, tune, amp - 3 plots with bpm names in x-axis

return bpms which are bad in all files (in each method)
return all bad bpms, without duplicates
"""

"""
add to options? eps_arc, eps_ir, minSamp_ir, minSamp_arc -> mandatory if --dbscan
numClusters, n_iter -> mandatory if --kmeans
options for --isolForest?
"""
def get_bad_bpms(algorithm, files, twiss, eps, minSamples, n_clusters, n_init):
    print(algorithm)
    bad_bpms = []
    files_list = files.split(',')
    files_x, files_y = separate_by_plane(files_list)
    common_bad_from_dbscan = []
    all_bad_from_dbscan =[]
    if algorithm == "dbscan":
        all_bad_from_dbscan_x, common_bad_from_dbscan_x = get_bad_bmps_from_dbscan(files_x, twiss, eps, minSamples, "x")
        #all_bad_from_dbscan_y, common_bad_from_dbscan_y = get_bad_bmps_from_dbscan(files_y, twiss, eps, minSamples, "y")
        print("---DBSCAN: all bad BPMS in horizontal---")
        for bpm in all_bad_from_dbscan_x:
            print(bpm)
        print("---DBSCAN: common bad BPMS in horizontal---")
        for bpm in common_bad_from_dbscan_x:
            print(bpm)
#         print("---DBSCAN: all bad BPMS in vertical---")
#         for bpm in all_bad_from_dbscan_y:
#             print(bpm)
#         print("---DBSCAN: all bad BPMS in vertical---")
#         for bpm in common_bad_from_dbscan_y:
#             print(bpm)
    if algorithm == "kmeans":
        all_bad_from_kmeans_x, common_bad_from_kmeans_x = get_bad_bmps_from_kmeans(files_x, twiss, n_clusters, n_init)
        #Not implemeted for vertical plane yet
        #all_bad_from_kmeans_y, common_bad_from_kmeans_y = get_bad_bmps_from_kmeans(files_y, twiss, n_clusters, n_init)

#     outpath = os.path.abspath(os.path.join(os.path.dirname(files_list[0]), "..", "bad_bpms_clustering.txt"))
#     print("Printing bad bpms to: {}".format(outpath))
#     with open(outpath, "w") as outfile:
#         outfile.write("\n".join(bad_bpms))
    
    
def get_bad_bmps_from_dbscan(files, twiss, eps, minSamples, plane):
    common_bad_from_dbscan = []
    all_bad_from_dbscan =[]
    for file_in in files:
        bpm_tfs_data = _create_columns(file_in, twiss)
        needed_columns_x = bpm_tfs_data[["TUNEX", "PH_ADV_BEAT", "AMPX"]].copy()
        #needed_columns_y = bpm_tfs_data[["TUNEY", "PH_ADV_BEAT", "AMPY"]].copy()
        if plane == "x":
            needed_columns = needed_columns_x
#         elif plane == "y":
#             needed_columns = needed_columns_y
        arc_bpm_data_for_clustering, ir_bpm_data_for_clustering = get_data_for_clustering(needed_columns)
        print("IRs BPMS:")
        for bpm in ir_bpm_data_for_clustering.index:
            print(bpm)
        print("Arcs/IRs")
        print(len(arc_bpm_data_for_clustering), len(ir_bpm_data_for_clustering))
        labels_from_arcs = _dbscan_clustering_noise(arc_bpm_data_for_clustering, eps, minSamples)
        bad_in_arcs_from_file = arc_bpm_data_for_clustering.iloc[np.where(labels_from_arcs == -1)].index
        labels_from_irs = _dbscan_clustering_noise(ir_bpm_data_for_clustering, 0.6, 190)
        bad_in_irs_from_file = ir_bpm_data_for_clustering.iloc[np.where(labels_from_irs == -1)].index
        print("---Bad BPMs from single file---")
        for bpm in list(bad_in_arcs_from_file)+list(bad_in_irs_from_file):
            print(bpm)
        all_bad_from_dbscan = all_bad_from_dbscan + list(bad_in_arcs_from_file) + list(bad_in_irs_from_file)
        if len(common_bad_from_dbscan) == 0:
            common_bad_from_dbscan = common_bad_from_dbscan + list(bad_in_arcs_from_file) + list(bad_in_irs_from_file)
        else:
            common_bad_from_dbscan = (list(set(common_bad_from_dbscan).
                                        intersection(bad_in_arcs_from_file).
                                        intersection(bad_in_irs_from_file)))


    all_bad_from_dbscan = set(list(all_bad_from_dbscan))
    print("Number of bad BPMs from all files")
    print(len(all_bad_from_dbscan))
    return all_bad_from_dbscan, common_bad_from_dbscan


def kmeans_iterration(file_in, twiss, n_clusters, n_init):
    bpm_tfs_data = _create_columns(file_in, twiss)
    arc_bpm_data_for_clustering, ir_bpm_data_for_clustering = get_data_for_clustering(bpm_tfs_data)
    ir_tune_ph = ir_bpm_data_for_clustering[["TUNEX", "PH_ADV_BEAT"]].copy()
    ir_tune_amp = ir_bpm_data_for_clustering[["TUNEX", "AMPX"]].copy()
    ir_ph_amp = ir_bpm_data_for_clustering[["PH_ADV_BEAT", "AMPX"]].copy()
    labels_irs_kmeans_tune_ph = _kmeans_clustering(ir_tune_ph, n_clusters, n_init, "PH_ADV_BEAT", "TUNEX")
    labels_irs_kmeans_tune_amp = _kmeans_clustering(ir_tune_amp, n_clusters, n_init, "TUNEX", "AMPX")
    labels_irs_kmeans_ph_amp = _kmeans_clustering(ir_ph_amp, n_clusters, n_init, "PH_ADV_BEAT", "AMPX" )
    limit_for_good_cluster_ir = ir_bpm_data_for_clustering.shape[0] * 0.1
    bad_in_irs_kmeans = []
    bad_ir_tune_ph = _get_bad_clusters_from_dataset(ir_tune_ph, labels_irs_kmeans_tune_ph, limit_for_good_cluster_ir)
    bad_ir_tune_amp = _get_bad_clusters_from_dataset(ir_tune_amp, labels_irs_kmeans_tune_amp, limit_for_good_cluster_ir)
    bad_ir_ph_amp = _get_bad_clusters_from_dataset(ir_ph_amp, labels_irs_kmeans_ph_amp, limit_for_good_cluster_ir)
    bad_in_irs_kmeans = list(set(list(bad_ir_ph_amp) + list(bad_ir_tune_amp) + list(bad_ir_tune_ph)))
    arc_bpm_data_for_clustering.loc[:, "TUNEX"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "TUNEX"])
    arc_bpm_data_for_clustering.loc[:, "AMPX"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "AMPX"])
    arc_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"])
    arc_tune_ph = arc_bpm_data_for_clustering[["TUNEX", "PH_ADV_BEAT"]].copy()
    arc_tune_amp = arc_bpm_data_for_clustering[["TUNEX", "AMPX"]].copy()
    arc_ph_amp = arc_bpm_data_for_clustering[["PH_ADV_BEAT", "AMPX"]].copy()
    labels_arcs_kmeans_tune_ph = _kmeans_clustering(arc_tune_ph, n_clusters, n_init, "PH_ADV_BEAT", "TUNEX")
    labels_arcs_kmeans_tune_amp = _kmeans_clustering(arc_tune_amp, n_clusters, n_init, "TUNEX", "AMPX")
    labels_arcs_kmeans_ph_amp = _kmeans_clustering(arc_ph_amp, n_clusters, n_init, "PH_ADV_BEAT", "AMPX" )
    limit_for_good_cluster_arc = arc_bpm_data_for_clustering.shape[0] * 0.10
    bad_in_arcs_kmeans = []
    bad_arc_tune_ph = _get_bad_clusters_from_dataset(arc_tune_ph, labels_arcs_kmeans_tune_ph, limit_for_good_cluster_arc)
    bad_arc_tune_amp = _get_bad_clusters_from_dataset(arc_tune_amp, labels_arcs_kmeans_tune_amp, limit_for_good_cluster_arc)
    bad_arc_ph_amp = _get_bad_clusters_from_dataset(arc_ph_amp, labels_arcs_kmeans_ph_amp, limit_for_good_cluster_arc)
    bad_in_arcs_kmeans = list(set(list(bad_arc_tune_ph) + list(bad_arc_ph_amp) + list(bad_arc_tune_amp)))
    return list(bad_in_irs_kmeans + bad_in_arcs_kmeans)
        
    
def define_centroids_x(twiss, dataframe, n_clusters):
    ph_in_tune_max = dataframe.loc[dataframe.TUNEX == 1].PH_ADV_BEAT
    tune_in_ph_max = dataframe.loc[dataframe.PH_ADV_BEAT == 1].TUNEX
    ph_in_tune_min = dataframe.loc[dataframe.TUNEX == 0].PH_ADV_BEAT
    tune_in_ph_min = dataframe.loc[dataframe.PH_ADV_BEAT == 0].TUNEX
    dataframe_remove_min_max = dataframe[dataframe.TUNEX != 1 | dataframe.TUNEX != 0 | dataframe.PH_ADV_BEAT != 1 | dataframe.PH_ADV_BEAT != 0]
    tune_avg = np.mean(dataframe_remove_min_max.TUNEX)
    ph_in_tune_avg = dataframe.loc[dataframe.TUNEX == tune_avg].PH_ADV_BEAT
    ph_avg = np.mean(dataframe_remove_min_max.PH_ADV_BEAT)
    tune_in_ph_avg = dataframe.loc[dataframe.PH_ADV_BEAT == ph_avg].TUNEX
    tune_ph_centroids = ([[1,ph_in_tune_max],[0,ph_in_tune_min],
                            [tune_in_ph_max,1],[tune_in_ph_min,0],
                            [tune_avg, ph_in_tune_avg],[tune_in_ph_avg, ph_avg]])
    return tune_ph_centroids

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

    
    # print("---Bad in the IRS as list---")
    # for bpm in bad_in_irs_kmeans:
        # print(bpm)
        # print(ir_bpm_data_for_clustering.loc[bpm, "PH_ADV_BEAT"])
        # print(ir_bpm_data_for_clustering.loc[bpm, "TUNEX"])
        # print(ir_bpm_data_for_clustering.loc[bpm, "AMPX"])
    
#     bad_arc_tune_amp = (arc_bpm_data_for_clustering.loc[bad_arc_tune_amp_all, :]
#                                                    .loc[np.abs(arc_bpm_data_for_clustering.PH_ADV_BEAT) > 0.1, :]
#                                                    .index)

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
    # print("---Bad in the arcs as list---")
    # for bpm in bad_in_arcs_kmeans:
        # print(bpm)
        # print(arc_bpm_data_for_clustering.loc[bpm, "PH_ADV_BEAT"])
        # print(arc_bpm_data_for_clustering.loc[bpm, "TUNEX"])
        # print(arc_bpm_data_for_clustering.loc[bpm, "AMPX"])

    
def get_data_for_clustering(bpm_tfs_data):
    ir_bpm_data_for_clustering = bpm_tfs_data.iloc[~lhc.Lhc.get_arc_bpms_mask(bpm_tfs_data.index)]
    ir_bpm_data_for_clustering.loc[:, "TUNEX"] = _normalize_parameter(ir_bpm_data_for_clustering.loc[:, "TUNEX"])
    ir_bpm_data_for_clustering.loc[:, "AMPX"] = _normalize_parameter(ir_bpm_data_for_clustering.loc[:, "AMPX"])
    ir_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"] = _normalize_parameter(ir_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"])
    arc_bpm_data_for_clustering = bpm_tfs_data.iloc[lhc.Lhc.get_arc_bpms_mask(bpm_tfs_data.index)] 
    arc_bpm_data_for_clustering.loc[:, "TUNEX"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "TUNEX"])
    arc_bpm_data_for_clustering.loc[:, "AMPX"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "AMPX"])
    arc_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"])
    return arc_bpm_data_for_clustering, ir_bpm_data_for_clustering, 


def _plot_bad_bpms_ph(all_data, bad_bpm_names):
    bpms_to_plot = all_data.iloc[bad_bpm_names, :]
    bpms_to_plot.PH_ADV_BEAT.plot()
    plt.title("Phase advance beating")
    plt.show()
    

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
    db = DBSCAN(eps=eps, min_samples=minSamples, metric='euclidean', 
                        algorithm='brute', leaf_size=30, p=None, n_jobs=-1)
    prediction = db.fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    _plotting(data, core_samples_mask, labels, "PH_ADV_BEAT", "TUNEX")
    multid_plotting(data, core_samples_mask, labels)
    return labels


def _kmeans_clustering(data, n_clusters, n_init, x, y):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, max_iter=n_init, tol=0.01, algorithm='auto')
    kmeans.fit(data)
    labels = kmeans.labels_
    #plot_kmeans(data, labels, x, y)
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


def multid_plotting(data_features, core_samples_mask, labels):
    plt.figure()
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    unique_labels = set(labels)
    for l, col in zip(unique_labels, COLORS):
        if l == -1:
            col = "red"
        class_member_mask = (labels == l)
        this_class_core_mask = class_member_mask & core_samples_mask
        this_class_non_core_mask = class_member_mask & ~core_samples_mask
        core_points = data_features.iloc[this_class_core_mask]
        non_core_points = data_features.iloc[this_class_non_core_mask]
        ax.plot3D(core_points.loc[:, "TUNEX"], core_points.loc[:, "AMPX"], core_points.loc[:, "PH_ADV_BEAT"], 'o', markerfacecolor=col,
                      markeredgecolor='k', markersize=14, label = "Core samples" if l != -1 else "")
        ax.plot3D(non_core_points.loc[:, "TUNEX"], non_core_points.loc[:, "AMPX"], non_core_points.loc[:, "PH_ADV_BEAT"], "^" if l == -1 else 's', markerfacecolor=col,
                      markeredgecolor='k', markersize=14 if l == -1 else 6, label = "Noise" if l == -1 else "Non core samples")
        #ax.plot3D(data_features[labels == l, 0], data_features[labels == l, 1], data_features[labels == l, 2],
        #               'o', color=plt.cm.jet(np.float(l) / np.max(labels + 1)) if l != -1 else 'black', label = "Core samples $C_{" + str(l+1) +"}$" if l != -1 else "Noise")
    ax.set_xlabel('Tune', fontsize = 25, linespacing=3.2)
    ax.set_ylabel('Amplitude', fontsize = 25, linespacing=3.2)
    ax.set_zlabel('Phase advance beating', fontsize = 25, linespacing=3.2)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='z', labelsize=15)

    plt.legend(fontsize = 25)
    plt.show()


def _plotting(data_features, core_samples_mask, labels, xcolumn, ycolumn):
    print(data_features.columns)
    unique_labels = set(labels)
    for k, col in zip(unique_labels, COLORS[:len(unique_labels)]):
        if k == -1:  # Is noise
            col = "red"

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
            label = "Core samples" if k != -1 else "",
        )
        plt.plot(
            non_core_points.loc[:, xcolumn],
            non_core_points.loc[:, ycolumn],
            "^" if k == -1 else 's',
            markerfacecolor=col,
            markeredgecolor='k',
            markersize=14 if k == -1 else 6,
            label = "Noise" if k == -1 else "Non core samples",
        )

    plt.xlabel('Phase advance beating', fontsize = 25)    
    plt.ylabel('Tune',fontsize = 25)
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
    

#TODO: --dbscan, --kmeans should be true/false options, depending on chosen algorithm -> additional mandatory options
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
        "--alg",
        dest="algorithm", type=str,
    )
    parser.add_argument(
        "--eps",
        dest="eps", type=float,
        help="DBSCAN - Distance to data points in the neighborhood of a core sample.",
    )
    parser.add_argument(
        "--minSamples",
        dest="minSamples", type=float,
        help="DBSCAN - Minimum number to data points in the neighborhood of a core sample.",
    )
    parser.add_argument(
        "--nClusters",
        dest="n_clusters", type=int,
    )
    parser.add_argument(
        "--nInit",
        dest="n_init", type=int,
    )
    options = parser.parse_args()
    return options.algorithm, options.files, options.twiss, options.eps, options.minSamples, options.n_clusters, options.n_init
    
if __name__ == "__main__":
    _algorithm, _files, _twiss, _eps, _minSamples, _n_clusters, _n_init = _parse_args()
    get_bad_bpms(_algorithm, _files, _twiss, _eps, _minSamples, _n_clusters, _n_init)
    #get_all_bad_bpms(_files, _twiss)
    #get_common_bad_bpms_kmeans(_files, _twiss)
    #clustering(_file, _twiss, _eps, _minSamples)
