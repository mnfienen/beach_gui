import timeseries_pkg.timeseries as timeseries
import pls
import numpy as np
from matplotlib import mlab
import utils
import copy

import random

from matplotlib import dates
from random import choice
from scipy import stats

import matplotlib.mlab as mlab
import matplotlib.pyplot as pyplot


def Validate(predictions, actual, decision_threshold):

    raw = list()

    for k in range(len(predictions)):
        t_pos = int(predictions[k] >= decision_threshold and actual[k] >= 2.3711)
        t_neg = int(predictions[k] < decision_threshold and actual[k] < 2.3711)
        f_pos = int(predictions[k] >= decision_threshold and actual[k] < 2.3711)
        f_neg = int(predictions[k] < decision_threshold and actual[k] >= 2.3711)
        raw.append([t_pos, t_neg, f_pos, f_neg])
    
    raw = np.array(raw)

    #pyplot.scatter(x=actual, y=predictions)
    
    return raw


class AR1():
    '''represents a timeseries model'''
    
    def __init__(self, data_dictionary, model_target, second_stage='pls', **args):
        #Initialize the time series model
        data = self.data = copy.copy(data_dictionary)
        target = self.model_target = model_target
        Y = self.Y = copy.copy(data[target])
        self.second_stage = second_stage
        
        t_series = self.timeseries = timeseries.Wrapper(dict=data_dictionary, **args)
        
        ar_args = dict()
        ar_args['na'] = 1
        ar_args['target'] = 'LogEC'
        ar_args['nb'] = []
        
        self.ar_model = t_series.ARX(ar_args)
        
        #Now replace the output variable in the data_dictionary with the AR residuals.
        data.pop(target)
        data['ar_residuals'] = self.ar_model.residual
        
        self.julian = np.vstack((self.ar_model.fitted, self.data['julian'])).transpose()
        self.Model_Residuals(second_stage, **args)
        self.Get_Fitted(**args)
        '''
        self.Threshold(**args)
        '''    
    
            
    def Model_Residuals(self, second_stage, **args):
        #Make the model that will use the inputs to predict the residuals of the AR model.
        
        if second_stage.lower() == 'pls':
            self.residual_model = pls.Model(self.data, model_target='ar_residuals', AR_part=self.julian, **args)
            
        if second_stage.lower() == 'pls_split':
            self.residual_model = pls.Model_Wrapper(self.data, model_target='ar_residuals', AR_part=self.julian, **args)
            self.residual_model.Generate_Models(**args)        
                
    def Get_Fitted(self, **args):
        #Combine the outputs of the two models.
        
        mask = np.ones( self.ar_model.residual.shape[0], dtype=bool )
        mask[ mlab.find(np.isnan(self.ar_model.residual)) ] = False
        
        self.actual = self.Y[mask]
        self.residual = self.residual_model.residual
        self.fitted = self.residual_model.fitted
        
        
    def Threshold(self, **args):
        #Use the sum of ar_model and residual_model to set the decision threshold.
        
        if 'specificity_limit' in args: specificity_limit = args['specificity_limit']
        else: specificity_limit = 0.92
        
        try:
            non_exceedances = self.fitted[mlab.find(self.actual < 2.3711)]
            self.threshold = utils.Quantile(non_exceedances, specificity_limit)
        except IndexError:
            self.threshold = 2.3711
        
        
    def Predict(self, validation_dictionary, **args):
        #Use the time series, then the residual model to predict exceedances
        
        validation_series = timeseries.Wrapper(dict=validation_dictionary, **args).series[0]
        ar_prediction = self.ar_model.Predict(validation_series)
        
        residual_prediction = self.residual_model.Predict(validation_dictionary)
        
        if self.second_stage.lower() == 'pls':
            residual_prediction = residual_prediction[:,self.residual_model.ncomp-1]
            
        return ar_prediction + residual_prediction
        
        
    def Validate(self, validation_dictionary, **args):
        #Use the time series, then the residual model to predict exceedances
        
        validation_series = timeseries.Wrapper(dict=validation_dictionary, **args).series[0]
        ar_prediction = self.ar_model.Predict(validation_series)
        
        performance = self.residual_model.Validate(validation_dictionary, AR_part=ar_prediction)
            
        return performance
        
        
    