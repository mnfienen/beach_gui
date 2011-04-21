import pls_modeling as p
import svm_modeling as s
import logistic_modeling as l

from matplotlib import mlab
from copy import deepcopy

import numpy as np

class Model:
    '''A wrapper containing multiple candidate models and the tools for combining them for inference.'''
    
    def __init__(self, data_dictionary, model_target, **args):
        '''Create a meta-model object'''
    
        #Check to see if a threshold has been specified in the function's arguments
        try: self.threshold = args['threshold']
        except KeyError: self.threshold=2.3711   # if there is no 'threshold' key, then use the default (2.3711)
 
    
        self.pls_model = p.Model(data_dictionary, model_target, threshold=self.threshold)
        self.svm_model = s.Model(data_dictionary, model_target, threshold=self.threshold)
        self.logistic_model = l.Model(data_dictionary, model_target, weights='integer', threshold=self.threshold)
        
        self.model_dict = deepcopy(data_dictionary)
        self.model_target = model_target
        
    def Predict(self, data_dictionary):
        pls_predictions = self.pls_model.Predict_Exceedances(data_dictionary)
        svm_predictions = self.svm_model.Predict_Exceedances(data_dictionary)
        logistic_predictions = self.logistic_model.Predict_Exceedances(data_dictionary)
        
        exceedances = np.array(  data_dictionary[self.model_target] >= self.threshold, dtype=int)
        
        return np.vstack( (exceedances, pls_predictions, svm_predictions, logistic_predictions) )
        
    def Aggregate(self, data_dictionary):
        #Combine the submodels to predict exceedances.
        svm_predictions = self.svm_model.Predict_Probability(data_dictionary)
        pls_predictions = self.pls_model.Predict_Probability(data_dictionary)
        logistic_predictions = self.logistic_model.Predict(data_dictionary)
        
        pls_precision = 1/self.pls_model.Extract('RMSEP')
        svm_precision = 3
        logistic_precision=3

        prob = np.zeros( shape=(len(svm_predictions), 2) )
        
        initial_probability = sum( np.array(self.model_dict[self.model_target] >= self.threshold, dtype=float) ) / len(self.model_dict[self.model_target])
        prob[:,0] = 1-initial_probability
        prob[:,1] = initial_probability
        
        prob[:,0] = prob[:,0] + pls_precision*(1-pls_predictions)
        prob[:,1] = prob[:,1] + pls_precision*(pls_predictions)
        
        prob[:,0] = prob[:,0] + svm_precision*(1-svm_predictions)
        prob[:,1] = prob[:,1] + svm_precision*(svm_predictions)
        
        prob[:,0] = prob[:,0] + logistic_precision*(1-logistic_predictions)
        prob[:,1] = prob[:,1] + logistic_precision*(logistic_predictions)
        
        prob = array(prob)
        
        return prob
        
        

        