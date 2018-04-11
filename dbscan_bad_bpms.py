from __future__ import print_function
import os
import sys
import numpy as np
import pandas
import logging
from scipy import stats
from time import time
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
from sklearn.externals import joblib
from sklearn.cluster import DBSCAN
from sklearn.neighbors import DistanceMetric
from scipy.spatial import distance
from sklearn.cluster import KMeans
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
import shutil

import argparse
#from mock.mock import inplace

if "win" in sys.platform:
    sys.path.append('\\\\AFS\\cern.ch\\work\\e\\efol\\public\\Beta-Beat.src\\')
else:
    sys.path.append('/afs/cern.ch/work/e/efol//public/master-beta-beat-commissioning/Beta-Beat.src')
from utils import tfs_pandas
from model.accelerators import lhc

LOGGER = logging.getLogger(__name__)

#WANTED_COLS_X = ["PK2PK", "CORMS", "NOISE", "TUNEX", "AMPX", "AVG_AMPX", "AVG_MUX", "NATTUNEX", "NATAMPX", "PH_ADV"]
"""
python dbscan_bad_bpms.py --alg=dbscan --files=/home/efol/work/public/halfInteger_afterCorr/Beam1@Turn@2017_11_28@10_20_48_638/Beam1@Turn@2017_11_28@10_20_48_638.sdds.linx,/home/efol/work/public/halfInteger_afterCorr/Beam1@Turn@2017_11_28@10_28_07_290/Beam1@Turn@2017_11_28@10_28_07_290.sdds.linx,/home/efol/work/public/halfInteger_afterCorr/Beam1@Turn@2017_11_28@10_34_07_907/Beam1@Turn@2017_11_28@10_34_07_907.sdds.linx --twiss=/home/efol/work/public/halfInteger_afterCorr/model_B1_41_43/twiss_adt.dat --eps=0.7 --minSamples=70 --nClusters=6 --nInit=100

"""

WANTED_COLS_X = ["PK2PK", "TUNEX", "AMPX", "PH_ADV", "PH_ADV_BEAT", "PH_ADV_MDL", "BETX"]



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
plotting: each file separetely: plot phase_adv_beat, tune, amp - 3 plots with bpm names in x-axis

return bpms which are bad in all files (in each method)
return all bad bpms, without duplicates
"""

"""
add to options? eps_arc, eps_ir, minSamp_ir, minSamp_arc -> mandatory if --dbscan
options for --isolForest?
"""
def get_bad_bpms(algorithm, files, twiss, eps, minSamples):
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
        print("---DBSCAN: common bad BPMS in horozontal---")
        for bpm in common_bad_from_dbscan_x:
            print(bpm)
#         print("---DBSCAN: all bad BPMS in vertical---")
#         for bpm in all_bad_from_dbscan_y:
#             print(bpm)
#         print("---DBSCAN: all bad BPMS in vertical---")
#         for bpm in common_bad_from_dbscan_y:
#             print(bpm)
    if algorithm == "lof":
        common_bad_bpms_from_lof_x, all_bad_from_lof_x = get_bad_bpms_from_lof(files_x, twiss, "x")
#         print("---LOF: all bad BPMS in horizontal---")
#         for index in all_bad_from_lof_x.index:
#             print(index)
    if algorithm == "forest":
        common_bad_bpms_from_forest_x, all_bad_from_forest_x = get_bad_bpms_from_forest(files_x, twiss, "x")
        print("---Random Forest: all bad BPMS in horizontal---")
        for index in all_bad_from_forest_x.index:
            print(index)
#     outpath = os.path.abspath(os.path.join(os.path.dirname(files_list[0]), "..", "bad_bpms_clustering.txt"))
#     print("Printing bad bpms to: {}".format(outpath))
#     with open(outpath, "w") as outfile:
#         outfile.write("\n".join(bad_bpms))


def remove_bpms_from_file(path, bad_bpm_names):
    #copy and rename original file
    src_dir = os.path.abspath(os.path.join(path, os.pardir))
    filename = os.path.basename(path)
    new_filename = os.path.join(src_dir, filename + ".notcleaned")
    os.rename(path, new_filename)
    #take the content of renamed file, remove bpms and write new file with the name of original file
    original_file_tfs = tfs_pandas.read_tfs(new_filename).set_index("NAME", drop=False)
    original_file_tfs = original_file_tfs.loc[~original_file_tfs.index.isin(bad_bpm_names)]
    tfs_pandas.write_tfs(path, original_file_tfs)


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
#         print("IRs BPMS:")
#         for bpm in ir_bpm_data_for_clustering.index:
#             print(bpm)
#         print("Arcs/IRs")
#         print(len(arc_bpm_data_for_clustering), len(ir_bpm_data_for_clustering))
        labels_from_arcs = _dbscan_clustering_noise(arc_bpm_data_for_clustering, eps, minSamples)
        bad_in_arcs_from_file = arc_bpm_data_for_clustering.iloc[np.where(labels_from_arcs == -1)].index
        labels_from_irs = _dbscan_clustering_noise(ir_bpm_data_for_clustering, 0.6, 190)
        bad_in_irs_from_file = ir_bpm_data_for_clustering.iloc[np.where(labels_from_irs == -1)].index
        print("---Bad BPMs from single file---")
        for bpm in list(bad_in_arcs_from_file)+list(bad_in_irs_from_file):
            print(bpm)
        #Question:Will overwrite every time?
        all_bad_from_dbscan = all_bad_from_dbscan + list(bad_in_arcs_from_file) + list(bad_in_irs_from_file)
        #import pdb; pdb.set_trace()
        if len(common_bad_from_dbscan) == 0:
            common_bad_from_dbscan = common_bad_from_dbscan + list(bad_in_arcs_from_file) + list(bad_in_irs_from_file)
        else:
            common_bad_from_dbscan = (list(set(common_bad_from_dbscan).
                                        intersection(bad_in_arcs_from_file).
                                        intersection(bad_in_irs_from_file)))
        remove_bpms_from_file(file_in, list(bad_in_arcs_from_file)+list(bad_in_irs_from_file))

    all_bad_from_dbscan = set(list(all_bad_from_dbscan))
    print("Number of bad BPMs from all files")
    print(len(all_bad_from_dbscan))
    return all_bad_from_dbscan, common_bad_from_dbscan



#run lof fit_predict, return labels, show plots with outliers(arcs/IRs) for each file
def get_bad_bpms_from_lof(files_x, twiss, plane):
    common_bad_bpms_lof = []
    all_bad_from_lof = []
    all_bad_arcs = []
    all_bad_irs = []
    for file_in in files_x:
        bpm_tfs_data = _create_columns(file_in, twiss)
        needed_columns_x = bpm_tfs_data[["TUNEX", "PH_ADV_BEAT", "AMPX"]].copy()
        #needed_columns_y = bpm_tfs_data[["TUNEY", "PH_ADV_BEAT", "AMPY"]].copy()
        if plane == "x":
            needed_columns = needed_columns_x
#         elif plane == "y":
#             needed_columns = needed_columns_y        
        arc_bpm_data_for_clustering, ir_bpm_data_for_clustering = get_data_for_clustering(needed_columns)
        lof_arcs = LocalOutlierFactor(n_neighbors=38, metric='minkowski', p=2, contamination=0.05)
        lof_irs = LocalOutlierFactor(n_neighbors=27, metric='minkowski', p=2, contamination=0.05)
        labels_from_arcs = lof_arcs.fit_predict(arc_bpm_data_for_clustering)
        labels_from_irs = lof_irs.fit_predict(ir_bpm_data_for_clustering)
        bad_in_arcs_from_file = arc_bpm_data_for_clustering.iloc[np.where(labels_from_arcs == -1)]
        bad_in_irs_from_file = ir_bpm_data_for_clustering.iloc[np.where(labels_from_irs == -1)]
        good_in_arcs_from_file = arc_bpm_data_for_clustering.iloc[np.where(labels_from_arcs != -1)]
        good_in_irs_from_file = ir_bpm_data_for_clustering.iloc[np.where(labels_from_irs != -1)]
#         print("---------------------------------------------------------------------------------")
#         print(os.path.abspath(file_in))
        bad_bpms_in_file = bad_in_arcs_from_file + bad_in_irs_from_file

        #plot_two_dim(good_in_arcs_from_file, bad_in_arcs_from_file, good_in_irs_from_file, bad_in_irs_from_file,"TUNEX", "PH_ADV_BEAT")
        #plot_two_dim(good_in_arcs_from_file, bad_in_arcs_from_file, good_in_irs_from_file, bad_in_irs_from_file,"AMPX", "PH_ADV_BEAT")
        if len(all_bad_arcs) == 0:
            all_bad_arcs = bad_in_arcs_from_file
        else:
            all_bad_arcs = all_bad_arcs + bad_in_arcs_from_file
        if len(all_bad_irs) == 0:
            all_bad_irs = bad_in_irs_from_file
        else:
            all_bad_irs = all_bad_irs + bad_in_irs_from_file
        if len(common_bad_bpms_lof) == 0:
            common_bad_bpms_lof = list(bad_in_arcs_from_file) + list(bad_in_irs_from_file)
        else:
            common_bad_bpms_lof = (list(set(common_bad_bpms_lof).
                                          intersection(bad_in_arcs_from_file).
                                          intersection(bad_in_irs_from_file)))
    all_bad_from_lof = all_bad_arcs + all_bad_irs
    good_in_arcs = arc_bpm_data_for_clustering[~arc_bpm_data_for_clustering.isin(all_bad_arcs)]
    good_in_irs = ir_bpm_data_for_clustering[~ir_bpm_data_for_clustering.isin(all_bad_irs)]
    print(len(np.unique(all_bad_from_lof.index)))
    for index in np.unique(all_bad_from_lof.index):
        print_significant_feature(index, arc_bpm_data_for_clustering, ir_bpm_data_for_clustering, good_in_arcs, good_in_irs)
    #check_bpm("BPM.12R1.B1", arc_bpm_data_for_clustering, ir_bpm_data_for_clustering, good_in_arcs, good_in_irs)
#     for file_in in files_x:
#         remove_bpms_from_file(file_in, np.unique(all_bad_from_lof.index))
    return common_bad_bpms_lof, all_bad_from_lof


def check_bpm(index, arc_bpm_data_for_clustering, ir_bpm_data_for_clustering, good_in_arcs_from_file, good_in_irs_from_file):
    if index in arc_bpm_data_for_clustering:
        print("Avearages")
        print(np.mean(good_in_arcs_from_file.TUNEX))
        print(np.mean(good_in_arcs_from_file.PH_ADV_BEAT))
        print(np.mean(good_in_arcs_from_file.AMPX))
        print(index)
        print(arc_bpm_data_for_clustering.loc[index, "TUNEX"])
        print(arc_bpm_data_for_clustering.loc[index, "PH_ADV_BEAT"])
        print(arc_bpm_data_for_clustering.loc[index, "AMPX"])
    else:
        print("Avearages")
        print(np.mean(good_in_irs_from_file.TUNEX))
        print(np.mean(good_in_irs_from_file.PH_ADV_BEAT))
        print(np.mean(good_in_irs_from_file.AMPX))
        print(index)
        print(ir_bpm_data_for_clustering.loc[index, "TUNEX"])
        print(ir_bpm_data_for_clustering.loc[index, "PH_ADV_BEAT"])
        print(ir_bpm_data_for_clustering.loc[index, "AMPX"])


def print_significant_feature(index, all_arcs, all_irs, good_in_arcs, good_in_irs):
    if index in all_arcs.index:
        print(index)
        max_dist = max(abs(all_arcs.loc[index, "TUNEX"]-np.mean(good_in_arcs.TUNEX)), abs(all_arcs.loc[index, "PH_ADV_BEAT"]-np.mean(good_in_arcs.PH_ADV_BEAT)),abs(all_arcs.loc[index, "AMPX"]-np.mean(good_in_arcs.AMPX)))
        if max_dist == abs(all_arcs.loc[index, "TUNEX"]-np.mean(good_in_arcs.TUNEX)):
            print("Significant feature: Tune")
            print("mean: " + str(np.mean(good_in_arcs.TUNEX)) + " file: " + str(all_arcs.loc[index, "TUNEX"]))
        elif max_dist == abs(all_arcs.loc[index, "PH_ADV_BEAT"]-np.mean(good_in_arcs.PH_ADV_BEAT)):
            print("Significant feature: Phase advance beating")
            print("mean: " + str(np.mean(good_in_arcs.PH_ADV_BEAT)) + " file: " + str(all_arcs.loc[index, "PH_ADV_BEAT"]))
        else:
            print("Significant feature: Amplitude")
            print("mean: " + str(np.mean(good_in_arcs.AMPX)) + " file: " + str(all_arcs.loc[index, "AMPX"]))
    elif index in all_irs.index:
        print(index)
        max_dist = max(abs(all_irs.loc[index, "TUNEX"]-np.mean(good_in_irs.TUNEX)), abs(all_irs.loc[index, "PH_ADV_BEAT"]-np.mean(good_in_irs.PH_ADV_BEAT)),abs(all_irs.loc[index, "AMPX"]-np.mean(good_in_irs.AMPX)))
        if max_dist == abs(all_irs.loc[index, "TUNEX"]-np.mean(good_in_irs.TUNEX)):
            print("Significant feature: Tune")
            print("mean: " + str(np.mean(good_in_irs.TUNEX)) + " file: " + str(all_irs.loc[index, "TUNEX"]))
        elif max_dist == abs(all_irs.loc[index, "PH_ADV_BEAT"]-np.mean(good_in_irs.PH_ADV_BEAT)):
            print("Significant feature: Phase advance beating")
            print("mean: " + str(np.mean(good_in_irs.PH_ADV_BEAT)) + " file: " + str(all_irs.loc[index, "PH_ADV_BEAT"]))
        else:
            print("Significant feature: Amplitude")
            print("mean: " + str(np.mean(good_in_irs.AMPX)) + " file: " + str(all_irs.loc[index, "AMPX"]))


def get_bad_bpms_from_forest(files_x, twiss, plane):
    common_bad_bpms_forest = []
    all_bad_from_forest = []
    all_bad_arcs = []
    all_bad_irs = []
    for file_in in files_x:
        bpm_tfs_data = _create_columns(file_in, twiss)
        needed_columns_x = bpm_tfs_data[["S","TUNEX", "PH_ADV_BEAT", "AMPX"]].copy()
        #needed_columns_y = bpm_tfs_data[["TUNEY", "PH_ADV_BEAT", "AMPY"]].copy()
        if plane == "x":
            needed_columns = needed_columns_x
#         elif plane == "y":
#             needed_columns = needed_columns_y
        arc_bpm_data_for_clustering, ir_bpm_data_for_clustering = get_data_for_clustering(needed_columns)
        forest_arcs = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, max_features=1.0, bootstrap=False)
        forest_irs = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, max_features=1.0, bootstrap=False)
        forest_arcs.fit(arc_bpm_data_for_clustering)
        forest_irs.fit(ir_bpm_data_for_clustering)
        labels_from_arcs = forest_arcs.predict(arc_bpm_data_for_clustering)
        labels_from_irs = forest_irs.predict(ir_bpm_data_for_clustering)
        bad_in_arcs_from_file = arc_bpm_data_for_clustering.iloc[np.where(labels_from_arcs == -1)]
        bad_in_irs_from_file = ir_bpm_data_for_clustering.iloc[np.where(labels_from_irs == -1)]
        good_in_arcs_from_file = arc_bpm_data_for_clustering.iloc[np.where(labels_from_arcs != -1)]
        good_in_irs_from_file = ir_bpm_data_for_clustering.iloc[np.where(labels_from_irs != -1)]

        bad_bpms_in_file = bad_in_arcs_from_file + bad_in_irs_from_file
        plot_two_dim(good_in_arcs_from_file, bad_in_arcs_from_file, good_in_irs_from_file, bad_in_irs_from_file,"TUNEX", "PH_ADV_BEAT")
        plot_two_dim(good_in_arcs_from_file, bad_in_arcs_from_file, good_in_irs_from_file, bad_in_irs_from_file,"AMPX", "PH_ADV_BEAT")
        multid_plotting(good_in_arcs_from_file, bad_in_arcs_from_file, os.path.basename(file_in) + ":   ARSc")
        multid_plotting(good_in_irs_from_file, bad_in_irs_from_file, os.path.basename(file_in) + ":   IRs")
        if len(all_bad_arcs) == 0:
            all_bad_arcs = bad_in_arcs_from_file
        else:
            all_bad_arcs = all_bad_arcs + bad_in_arcs_from_file
        if len(all_bad_irs) == 0:
            all_bad_irs = bad_in_irs_from_file
        else:
            all_bad_irs = all_bad_irs + bad_in_irs_from_file
        if len(common_bad_bpms_forest) == 0:
            common_bad_bpms_forest = list(bad_in_arcs_from_file) + list(bad_in_irs_from_file)
        else:
            common_bad_bpms_forest = (list(set(common_bad_bpms_forest).
                                          intersection(bad_in_arcs_from_file).
                                          intersection(bad_in_irs_from_file)))
    all_bad_from_forest = all_bad_arcs + all_bad_irs
    good_in_arcs = arc_bpm_data_for_clustering[~arc_bpm_data_for_clustering.isin(all_bad_arcs)]
    good_in_irs = ir_bpm_data_for_clustering[~ir_bpm_data_for_clustering.isin(all_bad_irs)]
    print(len(np.unique(all_bad_from_forest.index)))
    for index in np.unique(all_bad_from_forest.index):
        print_significant_feature(index, arc_bpm_data_for_clustering, ir_bpm_data_for_clustering, good_in_arcs, good_in_irs)
    print("Checking potentially wrong remodev BPMs.....")
    check_bpm("BPM.30R2.B1", arc_bpm_data_for_clustering, ir_bpm_data_for_clustering, good_in_arcs, good_in_irs)
    check_bpm("BPM.31R2.B1", arc_bpm_data_for_clustering, ir_bpm_data_for_clustering, good_in_arcs, good_in_irs)
#     for file_in in files_x:
#         remove_bpms_from_file(file_in, np.unique(all_bad_from_forest.index))
    return common_bad_bpms_forest, all_bad_from_forest


def plot_two_dim(good_in_arcs_from_file, bad_in_arcs_from_file, good_in_irs_from_file, bad_in_irs_from_file, col1, col2):
    plt.plot(
        good_in_arcs_from_file.loc[:, col1],
        good_in_arcs_from_file.loc[:, col2],
        'o',
        markerfacecolor="black",
        markeredgecolor='black',
        markersize=10,
        label = "Good",
    )
    plt.plot(
        bad_in_arcs_from_file.loc[:, col1],
        bad_in_arcs_from_file.loc[:, col2],
        '^',
        markerfacecolor="red",
        markeredgecolor='red',
        markersize=10,
        label = "Bad",
    )
    for index in bad_in_arcs_from_file.index:
        plt.annotate(index, xy=(bad_in_arcs_from_file.loc[index, col1], bad_in_arcs_from_file.loc[index, col2]))
    plt.xlabel(col1, fontsize = 25)
    plt.ylabel(col2,fontsize = 25)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.legend(fontsize = 25)
    plt.title("arcs")
    plt.show()

    plt.plot(
        good_in_irs_from_file.loc[:, col1],
        good_in_irs_from_file.loc[:, col2],
        'o',
        markerfacecolor="black",
        markeredgecolor='black',
        markersize=10,
        label = "Good",
    )
    plt.plot(
        bad_in_irs_from_file.loc[:, col1],
        bad_in_irs_from_file.loc[:, col2],
        '^',
        markerfacecolor="red",
        markeredgecolor='red',
        markersize=10,
        label = "Bad",
    )
    for index in bad_in_irs_from_file.index:
        plt.annotate(index, xy=(bad_in_irs_from_file.loc[index, col1], bad_in_irs_from_file.loc[index, col2]))
    plt.xlabel(col1, fontsize = 25)
    plt.ylabel(col2,fontsize = 25)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.legend(fontsize = 25)
    plt.title("IRS")
    plt.show()


def multid_plotting(good_bpms, bad_bpms, title):
    plt.figure()
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.plot3D(good_bpms.loc[:, "TUNEX"], good_bpms.loc[:, "AMPX"], good_bpms.loc[:, "PH_ADV_BEAT"], 'o', markerfacecolor="black",
                      markeredgecolor='black', markersize=14)
    ax.plot3D(bad_bpms.loc[:, "TUNEX"], bad_bpms.loc[:, "AMPX"], bad_bpms.loc[:, "PH_ADV_BEAT"], '^', markerfacecolor="red",
                      markeredgecolor='black', markersize=14)
    for index in bad_bpms.index:
        ax.text(bad_bpms.loc[index, "TUNEX"], bad_bpms.loc[index, "AMPX"], bad_bpms.loc[index, "PH_ADV_BEAT"], index)
    ax.set_title(title)
    ax.set_xlabel('Tune', fontsize = 25, linespacing=3.2)
    ax.set_ylabel('Amplitude', fontsize = 25, linespacing=3.2)
    ax.set_zlabel('Phase advance beating', fontsize = 25, linespacing=3.2)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='z', labelsize=15)

    plt.legend(fontsize = 25)
    plt.show()


def multid_plotting_dbscan(data_features, core_samples_mask, labels):
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


#plotting for dbscan
COLORS = ("blue", "red", "green", "yellow", "pink", "black", "orange")
def _plotting_dbscan(data_features, core_samples_mask, labels, xcolumn, ycolumn):
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
            markerfacecolor="None",
            markeredgecolor='black',
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

def get_data_for_clustering(bpm_tfs_data):
    ir_bpm_data_for_clustering = bpm_tfs_data.iloc[~lhc.Lhc.get_arc_bpms_mask(bpm_tfs_data.index)]
    ir_bpm_data_for_clustering.loc[:, "TUNEX"] = _normalize_parameter(ir_bpm_data_for_clustering.loc[:, "TUNEX"])
    ir_bpm_data_for_clustering.loc[:, "AMPX"] = _normalize_parameter(ir_bpm_data_for_clustering.loc[:, "AMPX"])
    ir_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"] = _normalize_parameter(ir_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"])
    ir_bpm_data_for_clustering.loc[:, "S"] = _normalize_parameter(ir_bpm_data_for_clustering.loc[:, "S"])*4
    arc_bpm_data_for_clustering = bpm_tfs_data.iloc[lhc.Lhc.get_arc_bpms_mask(bpm_tfs_data.index)] 
    arc_bpm_data_for_clustering.loc[:, "TUNEX"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "TUNEX"])
    arc_bpm_data_for_clustering.loc[:, "AMPX"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "AMPX"])
    arc_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "PH_ADV_BEAT"])
    arc_bpm_data_for_clustering.loc[:, "S"] = _normalize_parameter(arc_bpm_data_for_clustering.loc[:, "S"])*4
    return arc_bpm_data_for_clustering, ir_bpm_data_for_clustering


def _weighted_feature(data, feature, weight):
    column_data = data.loc[:, feature]
    weighted_column_data = column_data * weight
    data.loc[:, feature] = weighted_column_data


def _dbscan_clustering_noise(data, eps, minSamples):
    #mahalanobis_metric = DistanceMetric.get_metric('mahalanobis', V=np.cov(data))
    db = DBSCAN(eps=eps, min_samples=minSamples, metric='euclidean', 
                        algorithm='brute', leaf_size=30, p=None, n_jobs=-1)
    prediction = db.fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    _plotting_dbscan(data, core_samples_mask, labels, "PH_ADV_BEAT", "TUNEX")
    multid_plotting_dbscan(data, core_samples_mask, labels)
    return labels


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
    options = parser.parse_args()
    return options.algorithm, options.files, options.twiss, options.eps, options.minSamples
    
if __name__ == "__main__":
    _algorithm, _files, _twiss, _eps, _minSamples = _parse_args()
    get_bad_bpms(_algorithm, _files, _twiss, _eps, _minSamples)
    #get_all_bad_bpms(_files, _twiss)
    #get_common_bad_bpms_kmeans(_files, _twiss)
    #clustering(_file, _twiss, _eps, _minSamples)
