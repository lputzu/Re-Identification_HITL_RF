from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
#from quadprog import solve_qp
from sklearn.decomposition import PCA
import numpy as np
import sys, os, re
from scipy.spatial import distance
from passive_aggressive import PAIR2

#convert distances to scores [0,1]
def distance2score(distances):
    return 1/(1+distances)


#convert scores to distances [0,Inf)
def score2distance(score):
    #epsilon = sys.float_info.epsilon
    epsilon = np.finfo( type(1.) ).eps
    if len(np.shape(score))>1:
        score = np.squeeze(score)
    minscore = np.min(score)
    maxscore = np.max(score)
    if minscore < 0 or maxscore > 1:
        maxscore = np.max(score)
        score = (score-minscore)/(maxscore-minscore)
    score = score + epsilon
    distances = (1/score)
    return distances


class Ranker:
    def __init__(self, gallery, cams, method, separate_camera_set, datasetPath, eng=None):
        self.gallery = gallery #gallery images featureSet
        self.cams = cams #gallery images cameras
        self.method = method # RF method used for re-ranking
        self.separate_camera_set = separate_camera_set #separate gallery sets
        self.path = datasetPath # path for dataset features
        self.eng = eng # MATLAB engine (only for EMR)
        self.w = None
        self.score = None
        self.RFscore = None
        self.query = None

    def get_separate_rank(self, distance, rank, cam):
        if self.separate_camera_set: # Filter out samples from same camera
            sameCameraSet = self.cams == cam
            otherCameraSet = self.cams != cam
            distance[sameCameraSet] = np.max(distance)
            rank = np.hstack((rank[otherCameraSet], rank[sameCameraSet] ))
        return (distance, rank)

    def get_weight(self):
        str = re.findall(r'\d+', self.method)[0]
        wb = int(str)*0.1
        wa = 1-wb
        return (str,wa, wb)

    # ALL the below re-ranking methods take the same INPUTS and return the same OUTPUT
    # INPUT
    #   query   : query image features
    #   cam     : query image camera
    #   positive: gallery indexes of positive samples
    #   negative: gallery indexes of negative samples
    # OUTPUT
    #   distances  : gallery image distances
    #   rank : index of the sorted (by distance) gallery element
    def get_dist_rank(self, query, cam, positive, negative):
        #print(positive)
        #print(negative)
        if not negative and not positive: #firt round
            distances = euclidean_distances(np.expand_dims(query, axis=0), self.gallery)
            self.score = distance2score(distances)
            rank = np.argsort(distances)
            return self.get_separate_rank(distances[0], rank[0], cam)
        else:
            if self.method is "query_shift":
                print("query_shift")
                (distances, rank) = self.query_shift(query, positive, negative)
            elif "query_shift_learn" in self.method:
                (distances, rank) = self.query_shift_learn(query, positive, negative)
            elif self.method is "relevance_score":
                print("relevance_score")
                (distances, rank) = self.relevance_score(query, self.gallery, positive, negative)
            elif self.method is "MRS":
                print("mean_relevance_score")
                (distances, rank) = self.mean_rs(query, self.gallery, positive, negative)
            elif "PA" in self.method:
                print("PA")
                (distances, rank) = self.PAcall(query, positive, negative)
            else:
                raise ValueError('Wrong re-ranking method')
            return self.get_separate_rank(distances, rank, cam)

    def update_query(self, query, positive, negative):#ROCCHIO
        if negative: #if negative exist
            neg = 0.35*np.mean(self.gallery[negative, :], axis=0)
        else:
            neg = 0
        if positive:#if positive exist
            pos = 0.65*np.mean(self.gallery[positive, :], axis=0)
        else:
            pos = 0
        new_query = np.expand_dims(query, axis=0) + pos - neg
        self.query = new_query
        self.w = new_query

    def query_shift(self, query, positive, negative):#ROCCHIO
        self.update_query(query, positive, negative)
        distances = euclidean_distances(self.w, self.gallery)
        rank = np.argsort(distances)
        return (distances[0], rank[0])

    def query_shift_learn(self, query, positive, negative):#ROCCHIO learned
        (c, a, b)= self.get_weight()
        print("query_shift_learn",c)
        if self.w is None:
            (distances, rank) = self.query_shift(query, positive, negative)
        else:
            (distances, rank) = self.query_shift(self.w*b+query, positive, negative)
        self.w = np.squeeze(self.w) - query
        return (distances, rank)

    def relevance_score(self, query, gallery, positive, negative, weight=None):#GIACINTO
        if not positive: #if positive is empty compute distances from the query
            positive_features = (np.expand_dims(query,axis=0))
        else:
            positive_features = np.concatenate((np.expand_dims(query,axis=0),gallery[positive, :]))
        # calculate the distances between positive and gallery
        if weight is None:
            distance_positive = euclidean_distances(gallery, positive_features)
        else:
            distance_positive = pairwise_distances(gallery, positive_features, metric='minkowski', w=weight)
        # extract for each image the closest positive image distances
        closest_positive = distance_positive.min(axis=1)
        if not negative:#if negative is empty use a big distance value
            closest_negative = np.ones(np.shape(gallery)[0])*(np.max(closest_positive)+1000)
        else:
            negative_features = gallery[negative, :]
            # calculate the distances between negative and gallery
            if weight is None:
                distance_negative = euclidean_distances(gallery, negative_features)
            else:
                distance_negative = pairwise_distances(gallery, negative_features, metric='minkowski', w=weight)
            # extract for each image the closest negative image distances
            closest_negative = distance_negative.min(axis=1)
        #print("closest neg ", min(closest_negative), max(closest_negative))
        #print("closest pos ", min(closest_positive), max(closest_positive))
        # calculate the final score for each image
        scores = np.divide(closest_negative, (closest_negative+closest_positive))
        if "relevance_score_learn" in self.method:
            self.RFscore = scores
        rank = np.argsort(-scores)
        distances = score2distance(scores)
        return (distances, rank)

    def mean_rs(self, query, gallery, positive, negative):
        mean_pos = np.mean(np.vstack((query, gallery[positive, :])), axis=0)
        dist_pos = euclidean_distances(np.expand_dims(mean_pos, axis=0), gallery)[0]
        if negative: #if negative exist
            mean_neg = np.mean(gallery[negative, :], axis=0)
            dist_neg = euclidean_distances(np.expand_dims(mean_neg, axis=0), gallery)[0]
            scores = np.divide(dist_neg, (dist_neg+dist_pos))
            if "mean_rs_learn" in self.method:
                self.RFscore = scores
            rank = np.argsort(-scores)
            distances = score2distance(scores)
        else:
            distances = dist_pos
            if "mean_rs_learn" in self.method:
                self.RFscore = distance2score(distances)
            rank = np.argsort(distances)
        return (distances, rank)

    def PAcall(self, query, positive, negative):#Call to Passive Aggressive
        (c, a, b)= self.get_weight()
        print("PA",c)
        modelIL = PAIR2(w=self.w)
        model = PAIR2()
        pos_feature = np.vstack((query, self.gallery[positive, :]))
        neg_feature = np.vstack((-query, self.gallery[negative, :]))
        self.w = modelIL.fit(pos_feature,  neg_feature)
        w = model.fit(pos_feature,  neg_feature)
        #w = (w/sum(w))*a+(self.w/sum(self.w))*b
        w = w*a+self.w*b
        scores = np.sum(np.multiply(self.gallery, w), axis=1)

        distances = score2distance(scores)
        rank = np.argsort(distances)
        return (distances, rank)
