from __future__ import print_function
import os
import sys
import numpy as np
import pandas
import logging
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from time import time
from scipy.stats import randint as sp_randint
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network.multilayer_perceptron import MLPClassifier, MLPRegressor
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from autoencoder import Autoencoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import mpl_toolkits.mplot3d.axes3d as p3


if "win" in sys.platform:
    sys.path.append('\\\\AFS\\cern.ch\\work\\e\\efol\\public\\Beta-Beat.src\\')
else:
    sys.path.append('/afs/cern.ch/work/e/efol/public/Beta-Beat.src/')
from Utilities import tfs_pandas
import generate_data
import bpm_statistic

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
LOGGER = logging.getLogger(__name__)
MEASDIR = '/user/slops/data/LHC_DATA/OP_DATA/Betabeat'


def main(samples_file=None, targets_file=None):
    logging.basicConfig()
        
    if samples_file is None and targets_file is None:
        all_samples = []
        all_targets = []
        
        datesDirs = bpm_statistic.getDatesDirs(MEASDIR)
        allMeasDir = []
        for dateDir in datesDirs:
            for measDir in bpm_statistic.getMeasurementDirs(dateDir):
                if('2017' in dateDir):
                    allMeasDir.append(measDir)
         
        allTbTDirs = []
        for measDir in allMeasDir:
            for tbtDir in bpm_statistic.getTbTDataDirs(measDir):
                if ((os.path.isfile(os.path.join(tbtDir, os.path.basename(tbtDir) + '.sdds.raw')) and 
                 os.path.isfile(os.path.join(tbtDir, os.path.basename(tbtDir) + '..sdds.clean'))) or 
                (os.path.isfile(os.path.join(tbtDir, os.path.basename(tbtDir) + '_0.sdds')) and 
                 os.path.isfile(os.path.join(tbtDir, os.path.basename(tbtDir) + '_0.sdds.new')))):
                    if len(allTbTDirs) < 51:
                        allTbTDirs.append(tbtDir)
        
        
        print('Number of directories:', len(allTbTDirs))
        for tbtDir in allTbTDirs:
            print('Read directory: ', tbtDir)
            output = bpm_statistic.get_labeled_measurement(tbtDir)
            if output is None:
                continue
            (raw_bpm_names_x, raw_matrix_x, is_good_list_x,
             raw_bpm_names_y, raw_matrix_y, is_good_list_y) = output
            all_samples.append(raw_matrix_x)
            all_targets.extend(is_good_list_x)
            all_samples.append(raw_matrix_y)
            all_targets.extend(is_good_list_y)
        
        all_targets = np.array(all_targets)
        all_samples = np.vstack(all_samples)
        np.save('/afs/cern.ch/work/e/efol/public/ML/NN_scikit/bpms_samples.npy', all_samples)
        np.save('/afs/cern.ch/work/e/efol/public/ML/NN_scikit/bpms_targets.npy', all_targets)
    
    else:
        all_samples = np.load(samples_file)
        all_targets = np.load(targets_file)
        
    print('Shape all_samples:', np.shape(all_samples))

    all_targets = np.array(all_targets, dtype=np.bool)
    all_samples, all_targets = _split_samples_and_normalize(all_samples, all_targets)
    
    test_samples = []
    test_target = []
    (test_bpm_names_x, test_matrix_x, test_is_good_list_x,
     test_bpm_names_y, test_matrix_y, test_is_good_list_y) = bpm_statistic.get_labeled_measurement('/afs/cern.ch/work/e/efol/public/ML/NN_scikit/Beam1@Turn@2017_05_19@12_03_53_268')
    
    test_samples.append(test_matrix_x)
    test_samples.append(test_matrix_y)
    test_samples = np.vstack(test_samples)
    
    #In target there are only good bpms
    test_target.extend(test_is_good_list_x)
    test_target.extend(test_is_good_list_y)
    test_target = np.array(test_target, dtype=np.bool)
    
    features = _build_samples_with_features(test_samples)
    #_kmeans_pca_plotting(features)
    _dbscan_clustering(test_samples, test_target, features)
    #_agglomerative_clustering(features)
    
    
    test_samples, test_target = _split_samples_and_normalize(test_samples, test_target)
     
#    print('Shape of splitted all_samples:', np.shape(all_samples))
     
#     mlp_single = MLPRegressor(hidden_layer_sizes=(330, 110, 330), activation='relu', solver='adam', alpha=1e-5,
#                         batch_size='auto', learning_rate='adaptive', learning_rate_init=0.0001, power_t=0.5,
#                         max_iter=200, shuffle=True, random_state=None, tol=0.001, verbose=True, warm_start=False,
#                         momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
#                         beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#      
#     random_forest = RandomForestRegressor(n_estimators=2, criterion='mse', max_depth=5, 
#                                           min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
#                                           max_features= "sqrt", max_leaf_nodes=None, min_impurity_split=1e-07, 
#                                           bootstrap=True, oob_score=False, n_jobs=16, random_state=None, 
#                                           verbose=2, warm_start=False)
#     mlp = MultiOutputRegressor(mlp_single, n_jobs=-1)
#     ridge_regressor = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True,
#                             max_iter=None, tol=0.001, solver='auto', random_state=None)
#      
#     svm = OneClassSVM(kernel='rbf', degree=3, gamma='auto', coef0=0.0, 
#                       tol=0.001, nu=0.5, shrinking=True, cache_size=200, 
#                       verbose=True, max_iter=-1, random_state=None)
     
#     au = Autoencoder((2200, 330, 110))
#     prediction = au.fit_and_predict(all_samples, test_samples)
      
  
    #the samples only with bpms classified as good by SVD Clean 
#    all_good_samples = all_samples[all_targets, :]
#     
#     predictor = ridge_regressor
# 
#     predictor.fit(all_good_samples, all_good_samples)
#     joblib.dump(predictor, '/afs/cern.ch/work/e/efol/public/ML/NN_scikit/mlp.pkl')
#      mlp = joblib.load('/afs/cern.ch/work/e/efol/public/ML/NN_scikit/mlp.pkl')
#     
#     prediction_mlp = predictor.predict(test_samples)
# 
#     np.set_printoptions(threshold=np.nan)
#     #RMS for each BPM
#     print("Shape of test samples:", test_samples.shape)
#     print("Shape of prediction:", prediction_mlp.shape)
     # losses = mean_squared_error(test_samples, prediction_mlp, multioutput='raw_values')
#     losses = np.average((test_samples - prediction) ** 2, axis=1)
#     print(losses)
      
#     print("Shape of losses:", losses.shape)
#     #Average RMS for good BPMs
#     #print(np.sqrt(np.mean(np.square(losses[test_target]))))
#     #Average RMS for bad BPMs
#     #print(np.sqrt(np.mean(np.square( losses[np.logical_not(test_target)] ))))
#     
     #plot losses
#     plt.hist(losses, label='all', bins=100)
#     plt.hist(losses[test_target], label="good", bins=100)
#     plt.hist(losses[np.logical_not(test_target)], label="bad", bins=100)
#     plt.xlabel("Losses",fontsize = 25)
#     plt.xticks(fontsize = 25)
#     plt.yticks(fontsize = 25)
#     plt.legend(fontsize = 25)
#     plt.show()
#     plt.cla()
       
#     all_test_bpms_names = test_bpm_names_x + test_bpm_names_y
#     all_test_bpms_names = np.array(3 * all_test_bpms_names)
#        
#     all_test_bpms_names = all_test_bpms_names[np.argsort(losses)][::-1][:10]
#     ordered_test_target = test_target[np.argsort(losses)][::-1][:10]
#     for index, sample in enumerate(test_samples[np.argsort(losses)][::-1][:10]):
#         print("Is", all_test_bpms_names[index], "SVD good?", ordered_test_target[index])
#         plt.plot(np.array(range(len(sample))), sample)
#         plt.xlabel("Turns", fontsize=25)
#         plt.ylabel("Beam position [mm]", fontsize=25)
#         plt.xticks(fontsize = 25)
#         plt.yticks(fontsize = 25)   
#         plt.show()
#         plt.cla()


def _build_samples_with_features(samples):
    orbits = np.mean(samples, axis=1)
    orbits = np.abs(orbits) / np.max(np.abs(orbits))
    min = np.min(samples, axis=1)
    max = np.max(samples, axis=1)
    pk2pk = max - min
    pk2pk = pk2pk / np.max(np.abs(pk2pk))
    tune = np.argmax(np.abs(np.fft.rfft((samples.T - np.mean(samples, axis=1)).T)), axis=1) / float(samples.shape[1]) * 2.
    #features = np.column_stack((orbits, pk2pk, tune))
    features = np.column_stack((pk2pk, tune))
    return features

def _agglomerative_clustering(features):
    for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
        model = AgglomerativeClustering(n_clusters=2,
                                        linkage="average", affinity=metric)
        model.fit(features)
        plt.figure()
        plt.axes([0, 0, 1, 1])
        for l, c in zip(np.arange(model.n_clusters), 'rgbk'):
            plt.plot(features[model.labels_ == l].T, c=c, alpha=.5)
    #     plt.axis('tight')
    #     plt.axis('off')
        plt.suptitle("AgglomerativeClustering(affinity=%s)" % metric, size=20)


plt.show()

    
def _dbscan_clustering(samples, is_good, features):
    for index, metric in enumerate(["euclidean"]):
        db = DBSCAN(eps=0.075, min_samples=100, metric=metric, 
                        algorithm='auto', leaf_size=30, p=None, n_jobs=-1)
        prediction = db.fit(features)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        np.set_printoptions(threshold=np.nan)
        
        noise_samples = samples[np.where(labels == -1)]
        noise_is_good = is_good[np.where(labels == -1)]
        for i, sample in enumerate(noise_samples):
            print(features[i])
            print("Is SVD good?", noise_is_good[i], "Tune:", features[i][1] )
            plt.plot(range(len(sample)), sample)
            plt.xlabel("Turns", fontsize=25)
            plt.ylabel("Beam position [mm]", fontsize=25)
            plt.xticks(fontsize = 25)
            plt.yticks(fontsize = 25)   
            plt.show()
            plt.cla()
    
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
#         plt.figure()
#         fig = plt.figure()
#         ax = p3.Axes3D(fig)
#         #ax.view_init(7, -80)
#         for l, col in zip(unique_labels, colors):
# #             if l == -1:
# #                 col = [0, 0, 0, 1]
# #             class_member_mask = (labels == l)
# #             xy = features[class_member_mask & core_samples_mask]
# #             ax.plot3D(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
# #                      markeredgecolor='l', markersize=14, label = "Core samples $C_{" + str(l+1) +"}$" if l != -1 else "")
# #                      #markeredgecolor='k', markersize=14, label = "Core samples" if k != -1 else "")
# #              
# #             xy = features[class_member_mask & ~core_samples_mask]
# #             ax.plot3D(xy[:, 0], xy[:, 1], xy[:, 2], 'o' if l == -1 else 's', markerfacecolor=tuple(col),
# #                      markeredgecolor='l', markersize=6, label = "Noise" if l == -1 else "Non core samples $C_{" + str(l+1) +"}$")
#             
#             ax.plot3D(features[labels == l, 0], features[labels == l, 1], features[labels == l, 2],
#                       'o', color=plt.cm.jet(np.float(l) / np.max(labels + 1)) if l != -1 else 'black', label = "Core samples $C_{" + str(l+1) +"}$" if l != -1 else "Noise")
#         ax.set_xlabel('Orbit [mm]', fontsize = 25, linespacing=3.2)
#         ax.set_ylabel('Peak to peak [mm]', fontsize = 25, linespacing=3.2)
#         ax.set_zlabel('Tune', fontsize = 25, linespacing=3.2)
#         ax.tick_params(axis='x', labelsize=15)
#         ax.tick_params(axis='y', labelsize=15)
#         ax.tick_params(axis='z', labelsize=15)
#        
#         #ax.xaxis.set_rotate_label(False)
#         #ax.yaxis.set_rotate_label(False)
#         #ax.zaxis.set_rotate_label(False)
#         plt.legend(fontsize = 25)
#         plt.title('Data points in 3D-Feature space', fontsize = 25)
#         plt.show()
        
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
         
            class_member_mask = (labels == k)
         
            xy = features[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14, label = "Core samples $C_{" + str(k+1) +"}$" if k != -1 else "")
                     #markeredgecolor='k', markersize=14, label = "Core samples" if k != -1 else "")
             
            xy = features[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 1], xy[:, 0], 'o' if k == -1 else 's', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6, label = "Noise" if k == -1 else "Non core samples $C_{" + str(k+1) +"}$")
                     #markeredgecolor='k', markersize=6, label = "Noise" if k == -1 else "Non core samples")
        plt.xlabel("Peak to peak",fontsize = 25)
        plt.ylabel("Tune", fontsize = 25)    
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.legend(fontsize = 25)
        #plt.suptitle("%s metrics" % metric, size=25)
     
    #plt.title('Estimated number of clusters: %d' % n_clusters_, fontsize=25)
    plt.show()
    
    
    
    

def _kmeans_pca_plotting(features):
    n_samples, n_features = features.shape
    #reduced_data = PCA(n_components=2).fit_transform(features)
    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10, n_jobs=-1)
    kmeans.fit(features)
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    a_min, a_max = features[:, 2].min() - 1, features[:, 2].max() + 1
    b_min, b_max = features[:, 3].min() - 1, features[:, 3].max() + 1
    c_min, c_max = features[:, 4].min() - 1, features[:, 4].max() + 1
    xx, yy, aa, bb, cc = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(a_min, a_max, h), np.arange(b_min, b_max, h), np.arange(c_min, c_max, h))
    
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel(), aa.ravel(),bb.ravel(),cc.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max(), aa.min(), aa.max(), bb.min(), bb.max(), cc.min(), cc.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    
    plt.plot(features[:, 0], features[:, 1], features[:, 2], features[:, 3], features[:, 4], 'k.', markersize=6)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.legend(fontsize = 25)
    plt.show()

def _split_samples_and_normalize(samples, targets):
    # Split
    samples1 = samples[:, :2200]
    samples2 = samples[:, 2200:4400]
    samples3 = samples[:, 4400:6600]
   
    samples = np.vstack((samples1, samples2, samples3))
    
    # Remove average (orbit)
    samples = (samples.transpose() - np.mean(samples, axis=1)).transpose()
    
    # Normalize amplitude
    samples_pk2pk = np.max(samples, axis=1) - np.min(samples, axis=1)
    samples_pk2pk[samples_pk2pk < 1e-16] = 1.
    samples = (samples.transpose() / samples_pk2pk).transpose()
    
    # Shift to [0, 1] range
    samples = (samples + np.ones(samples.shape))/2
    
    list_targets = list(targets)
    targets = np.array(3 * list_targets)
    return samples, targets
        
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        main()
    else:
        main(*sys.argv[1:])

