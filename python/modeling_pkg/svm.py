import numpy as np

import svm
from svmc import C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
from svmc import LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED

import utils

from matplotlib import dates
from random import choice
from copy import deepcopy

import matplotlib.mlab as mlab
import matplotlib.pyplot as pyplot











def Validate(model, data_dictionary):
    outcome = model.model_target
    ncomp = model.ncomp - 1

    predictions = model.Predict(data_dictionary)
    predictions = predictions[:, ncomp]
    actual = data_dictionary[outcome]

    raw = list()

    for k in range(len(predictions)):
        t_pos = int(predictions[k] >= model.threshold and actual[k] >= 2.3711)
        t_neg = int(predictions[k] < model.threshold and actual[k] < 2.3711)
        f_pos = int(predictions[k] >= model.threshold and actual[k] < 2.3711)
        f_neg = int(predictions[k] < model.threshold and actual[k] >= 2.3711)
        raw.append([t_pos, t_neg, f_pos, f_neg])
    
    raw = np.array(raw)

    #pyplot.scatter(x=actual, y=predictions)
    
    return raw









class Model: 
    '''represents an SVM model'''

    def __init__(self, data_dictionary, model_target, kernel=LINEAR, cv_segments=10, **args):
        #Create an SVM model object
    
        #Check to see if a threshold has been specified in the function's arguments
        try: self.threshold = args['threshold']
        except KeyError: self.threshold=2.3711   # if there is no 'threshold' key, then use the default (2.3711)
        
        #Store some object data
        model_dict = deepcopy(data_dictionary)
        self.model_target = model_target
        self.folds = cv_segments
               
        #Label the exceedances in the training set.
        model_dict[model_target] = self.Assign_Labels(model_dict[model_target])
        
        #Extract the training labels and training set
        self.training_labels = model_dict.pop(model_target)
        self.training_set = np.transpose(model_dict.values())
        self.headers = model_dict.keys()
                
        #Scale the covariates to [-1,1]
        self.Scale_Covariates()
        
        #Generate an SVM model.
        self.svm_problem = svm.svm_problem(self.training_labels, self.training_set)
        self.svm_params = {'kernel_type' : kernel, 'weight_label' : [0,1], 'weight' : [10,1]}
        self.model=svm.svm_model(self.svm_problem, svm.svm_parameter(**self.svm_params))
        
        #Use cross-validation to find the best number of components in the model.
        self.Select_Linear_Model(-5, 10)
        
        #Rebuild the model, calculating the probabilities of class membership
        self.svm_params['probability']=1
        self.model=svm.svm_model(self.svm_problem, svm.svm_parameter(**self.svm_params))


    def Scale_Covariates(self):
        #Establish the range and offset vectors
        self.covariate_offsets = np.zeros( len(self.headers) )
        self.covariate_ranges = np.zeros( len(self.headers) )
        
        #Scale each covariate to [-1,1] and remember the transformation applied
        for i in range( len(self.headers) ):
            self.covariate_ranges[i] = max(self.training_set[:,i]) - min(self.training_set[:,i])
            self.covariate_offsets[i] = min(self.training_set[:,i]) + self.covariate_ranges[i]/2
            self.training_set[:,i] = (self.training_set[:,i]-self.covariate_offsets[i]) * 2./self.covariate_ranges[i]
                      
           
    def Assign_Labels(self, raw):
        #Label observations as above or below the threshold.
        raw = np.array(raw >= self.threshold, dtype=float)
        return raw


    def Predict(self, data_dictionary):
        predictors = np.zeros(shape=( np.array(data_dictionary.values()).shape[1], len(self.headers) ))
        
        for i in range( len(self.headers) ):
            predictors[:,i] = (data_dictionary[self.headers[i]]-self.covariate_offsets[i]) * 2./self.covariate_ranges[i]
            
        #Generate the predictons
        predictions = []
        for i in range( len(predictors) ):
            predictions.append( self.model.predict( predictors[i] ) )
            
        return np.array(predictions, dtype=int)
        
    def Predict_Exceedances(self, data_dictionary):
        return self.Predict(data_dictionary)
        
    def Predict_Probability(self, data_dictionary):
        predictors = np.zeros(shape=( np.array(data_dictionary.values()).shape[1], len(self.headers) ))
        
        for i in range( len(self.headers) ):
            predictors[:,i] = (data_dictionary[self.headers[i]]-self.covariate_offsets[i]) * 2./self.covariate_ranges[i]
            
        #Generate the predictons
        predictions = []
        for i in range( len(predictors) ):
            predictions.append( self.model.predict_probability( predictors[i] )[1][1] )
            
        
        
        return np.array(predictions)

    
    def Select_Model(self, C_min=-10, C_steps=11,  gamma_min=-15, gamma_steps=16):
        #Search for the model parameters that give the smallest CV error
        (C, gamma) = self.__Search__(C_min, C_steps, gamma_min, gamma_steps, 1, 1)
        #(C, gamma) = self.__Search__(np.log2(C)-5, 100, np.log2(gamma)-5, 100, 0.1, 0.1)
        (C, gamma) = self.__Search__(np.log2(C)-1, 100, np.log2(gamma)-1, 100, 0.02, 0.02)
        #(C, gamma) = self.__Search__(np.log2(C)-0.5, 100, np.log2(gamma)-0.5, 100, 0.01, 0.01)
        
        self.svm_params['C'] = C
        self.svm_params['gamma'] = gamma
        
        self.model = svm.svm_model(self.svm_problem, svm.svm_parameter(**self.svm_params))
    
    def Select_Linear_Model(self, C_min=-10, C_steps=11):
        #Search for the model parameters that give the smallest CV error
        C = self.__Linear_Search__(C_min, C_steps, 1)
        C = self.__Linear_Search__(np.log2(C)-2, 50, 0.08)
        C = self.__Linear_Search__(np.log2(C)-0.5, 50, 0.02)
        
        self.svm_params['C'] = C
        
        self.model = svm.svm_model(self.svm_problem, svm.svm_parameter(**self.svm_params))
    
    
    def __Search__(self, C_min, C_steps,  gamma_min, gamma_steps, C_step_by=1., gamma_step_by=1.):
        #Utility function used by Parameter_Search() to find the best parameters
        param_grid = np.array( [[ (C,gamma) for C in 2**(np.arange(C_steps, dtype=float)*C_step_by+C_min)] for gamma in 2**(np.arange(gamma_steps, dtype=float)*gamma_step_by+gamma_min)] )
        error_grid = np.zeros( shape=param_grid.shape[0:2] )
        
        
        for row in range( param_grid.shape[0] ):
            for col in range( param_grid.shape[1] ):
                self.svm_params['C'] = float( param_grid[row,col,0] )
                self.svm_params['gamma'] = float( param_grid[row,col,1] )
                
                CV_predictions = svm.cross_validation(self.svm_problem, svm.svm_parameter(**self.svm_params), self.folds)
                
                error = sum(abs(CV_predictions-self.training_labels))/len(self.training_labels)
                error_grid[row,col] = error

        best = mlab.find(error_grid == error_grid.flatten().min())
        row = best // C_steps
        col = best % C_steps

        (C, gamma) = param_grid[row, col][0].flatten()

        return (C, gamma)
        
        
    def __Linear_Search__(self, C_min, C_steps, C_step_by=1.):
        #Utility function used by Parameter_Search() to find the best parameters
        param_grid = np.array( [ C for C in 2**(np.arange(C_steps, dtype=float)*C_step_by+C_min) ] )
        error_grid = np.zeros( len(param_grid) )
        
        
        for i in range( len(param_grid) ):
            self.svm_params['C'] = float( param_grid[i] )
            
            CV_predictions = svm.cross_validation(self.svm_problem, svm.svm_parameter(**self.svm_params), self.folds)
            
            error = sum(abs(CV_predictions-self.training_labels))/len(self.training_labels)
            error_grid[i] = error

        best = mlab.find(error_grid == error_grid.flatten().min())

        C = param_grid[best][0]

        return C