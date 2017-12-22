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
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.stats import randint as sp_randint
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network.multilayer_perceptron import MLPClassifier, MLPRegressor
from sklearn.externals import joblib
from sklearn.cluster import DBSCAN
from sklearn.neighbors import DistanceMetric
from scipy.spatial import distance
import argparse
from mock.mock import inplace
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

if "win" in sys.platform:
    sys.path.append('\\\\AFS\\cern.ch\\work\\e\\efol\\public\\Beta-Beat.src\\')
else:
    sys.path.append('/afs/cern.ch/work/e/efol/public/Beta-Beat.src/')
from Utilities import tfs_pandas
from model.accelerators import lhc

LOGGER = logging.getLogger(__name__)

#WANTED_COLS_X = ["PK2PK", "CORMS", "NOISE", "TUNEX", "AMPX", "AVG_AMPX", "AVG_MUX", "NATTUNEX", "NATAMPX", "PH_ADV"]
WANTED_COLS_X = ["PK2PK", "TUNEX", "AMPX", "PH_ADV", "PH_ADV_BEAT", "PH_ADV_MDL", "BETX"]


def clustering(file, twiss, eps, minSamples):
    #read bpm data
    bpm_tfs_data = _create_columns(file, twiss)
    #cluster into IR, focusing, defocusing

    bpm_tfs_data_copy = bpm_tfs_data[["PK2PK"]].copy()
    arc_bpm_data = bpm_tfs_data_copy.iloc[lhc.Lhc.get_arc_bpms_mask(bpm_tfs_data_copy.index)] 
    
    
    
    arc_bpm_data_from_scaler = StandardScaler().fit_transform(arc_bpm_data.values)
    scaled_arc_bpm_data = pandas.DataFrame(arc_bpm_data_from_scaler, index=arc_bpm_data.index, columns=arc_bpm_data.columns)
    
    isolationForest = IsolationForest(max_samples='auto', random_state=None)
    isolationForest.fit(arc_bpm_data)
    scores_pred = isolationForest.decision_function(arc_bpm_data)
    prediction = isolationForest.predict(arc_bpm_data)
    threshold = stats.scoreatpercentile(scores_pred, 10)
    
    xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
    n_inliers = int((1. - 0.1) * 284)
    n_outliers = 28
    ground_truth = np.ones(284, dtype=int)
    ground_truth[-n_outliers:] = -1
    Z = isolationForest.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.plot(
            scaled_arc_bpm_data["PK2PK"].values,
            'o',
            markerfacecolor="blue")
    
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='orange')
    plt.show()
    
    #print(arc_bpm_data.shape[0])
    
    #labels_from_arcs = _dbscan_clustering_noise(twiss, scaled_arc_bpm_data, eps, minSamples)
    #bad_in_arcs = arc_bpm_data.iloc[np.where(labels_from_arcs == -1)]
    #good_in_arcs = arc_bpm_data.iloc[np.where(labels_from_arcs != -1)]
    
    #print("Number of bad BPMs in the arcs")
    #print(bad_in_arcs.shape[0])
    
    #for index, row in bad_in_arcs.iterrows():
    #    print(index)
    #    print(scaled_arc_bpm_data.loc[index, "AMPX"])
	#_max_distance(arc_bpm_data_for_clustering,row)
	




def _weighted_feature(data, feature, weight):
    column_data = data.loc[:, feature]
    weighted_column_data = column_data * weight
    data.loc[:, feature] = weighted_column_data   
    
 
def _max_distance(data_features, data_point):
    distances = []
    #average of data features per column
    for column in data_features.columns:
        column_mean = data_features[column].mean()
        distances.append(distance.mahalanobis(column_mean, data_point[column], np.linalg.inv(np.cov(data_features))))
                         #(column_mean, data_point[column]))
    all_distances = pandas.DataFrame(distances, data_features.columns)
    #print("###Maximum distance###")
    #print(all_distances.max(axis=1))
    #all_distances.transpose()
    indx = all_distances.idxmax()
    #print(all_distances.idxmax())
    print(data_point[indx])
    print(all_distances.max)
    

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

def _dbscan_clustering_noise(twiss, data, eps, minSamples):
    #mahalanobis_metric = DistanceMetric.get_metric('mahalanobis', V=np.cov(data))
    db = DBSCAN(eps=eps, min_samples=minSamples, metric='euclidean',
                        algorithm='auto', leaf_size=30, p=None, n_jobs=-1)
    prediction = db.fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    _plotting(data, core_samples_mask, labels, "PK2PK", "TUNEX")
    _plot_beta_function(twiss, data, labels)
    multid_plotting(data, core_samples_mask, labels)
   
    return labels


COLORS = ("blue", "red", "green", "yellow", "darkviolet", "cyan", "maroon")

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


def multid_plotting(data_features, core_samples_mask, labels):
    plt.figure()
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    unique_labels = set(labels)
    for l, col in zip(unique_labels, COLORS):
        if l == -1:
            col = "black"
        class_member_mask = (labels == l)
        this_class_core_mask = class_member_mask & core_samples_mask
        this_class_non_core_mask = class_member_mask & ~core_samples_mask
     
        core_points = data_features.iloc[this_class_core_mask]
        non_core_points = data_features.iloc[this_class_non_core_mask]
        ax.plot3D(core_points.loc[:, "TUNEX"], core_points.loc[:, "AMPX"], core_points.loc[:, "PH_ADV_BEAT"], 'o', markerfacecolor=col,
                      markeredgecolor='k', markersize=14, label = "Core samples $C_{" + str(l+1) +"}$" if l != -1 else "")
              
        ax.plot3D(non_core_points.loc[:, "TUNEX"], non_core_points.loc[:, "AMPX"], non_core_points.loc[:, "PH_ADV_BEAT"], 'o' if l == -1 else 's', markerfacecolor=col,
                      markeredgecolor='k', markersize=6, label = "Noise" if l == -1 else "Non core samples $C_{" + str(l+1) +"}$")
             
        #ax.plot3D(data_features[labels == l, 0], data_features[labels == l, 1], data_features[labels == l, 2],
        #               'o', color=plt.cm.jet(np.float(l) / np.max(labels + 1)) if l != -1 else 'black', label = "Core samples $C_{" + str(l+1) +"}$" if l != -1 else "Noise")
    ax.set_xlabel('Tune', fontsize = 25, linespacing=3.2)
    ax.set_ylabel('Amplitude', fontsize = 25, linespacing=3.2)
    ax.set_zlabel('Phase beating', fontsize = 25, linespacing=3.2)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='z', labelsize=15)

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

    ph_adv_bpm_data = (np.roll(bpm_tfs_data.loc[:, "MUX"], -1) - bpm_tfs_data.loc[:, "MUX"])
    ph_adv_bpm_data[ph_adv_bpm_data < 0.] += 1.
    
    ph_adv_model = (model_tfs_data.loc[:, "MUX"] - np.roll(model_tfs_data.loc[:, "MUX"], 1)) % 1.
    bpm_tfs_data.loc[:, "PH_ADV_MDL"] = ph_adv_model

    bpm_tfs_data.loc[:, "PH_ADV_BEAT"] = np.abs(ph_adv_model - ph_adv_bpm_data)
    
    bpm_tfs_data.loc[:, "PH_ADV"] = ph_adv_bpm_data
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
