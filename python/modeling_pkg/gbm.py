import numpy as np
import rpy2
import rpy2.robjects as rob
import rpy2.robjects.numpy2ri
import utils

from matplotlib import dates
from random import choice
from copy import deepcopy

import matplotlib.mlab as mlab
import matplotlib.pyplot as pyplot


#Fire up the interface to R
r = rob.r
r.library("gbm")








class Model: 
    '''represents a gbm (tree with boosting) model generated in R'''

    def __init__(self, data_dictionary, model_target, **args):
        #Create a logistic model object
    
        #Check to see if a threshold has been specified in the function's arguments
        try: self.regulatory_threshold = args['threshold']
        except KeyError: self.regulatory_threshold=2.3711   # if there is no 'threshold' key, then use the default (2.3711)

        try: self.iterations = args['iterations']
        except KeyError: self.iterations=500   # if there is no 'iterations' key, then use the default (400)

        #Cost: two values - the first is the cost of a false positive, the second is the cost of a false negative.
        try: self.cost = args['cost']
        except KeyError: self.cost=[1,1]   # if there is no 'cost' key, then use the default [1,1] (=equal weight)
        self.specificity = self.cost[1]      

        #depth: how many branches should be allowed per decision tree?
        try: self.depth = args['depth']
        except KeyError: self.depth = 1   # if there is no 'depth' key, then use the default 1 (decision stumps)  

        #shrinkage: learning rate parameter
        try: self.shrinkage = args['shrinkage']
        except KeyError: self.shrinkage = 0.01   # if there is no 'shrinkage' key, then use the default 0.01

        #Store some object data
        self.data_dictionary = deepcopy(data_dictionary)
        self.model_target = model_target
                
        #Check to see if a weighting method has been specified in the function's arguments
        try:
            #integer (discrete) weighting
            if str(args['weights']).lower()[0] in ['d', 'i']: 
                self.weights = self.Assign_Weights(method=1)
                
            #float (continuous) weighting
            elif str(args['weights']).lower()[0] in ['f']: 
                self.weights = self.Assign_Weights(method=2)
                
            #cost-based weighting
            elif str(args['weights']).lower()[0] in ['c']: 
                self.weights = self.Assign_Weights(method=3)

            #cost-based weighting, and down-weight the observations near the threshold
            elif str(args['weights']).lower()[0] in ['b']: 
                self.weights = self.Assign_Weights(method=4)

            else: self.weights = self.Assign_Weights(method=0) 
                
        #If there is no 'weights' key, set all weights to one.
        except KeyError: 
            self.weights = self.Assign_Weights(method=0) 
        
        #Label the exceedances in the training set.
        self.data_dictionary[model_target] = self.Assign_Labels(self.data_dictionary[model_target])
        
        #Get the data into R 
        self.data_frame = utils.Dictionary_to_R(self.data_dictionary)

        #Generate a logistic regression model in R.
        self.formula = r['as.formula'](self.model_target + '~.')
        self.gbm_params = {'formula' : self.formula, \
            'distribution' : 'bernoulli', \
            'data' : self.data_frame, \
            'weights' : self.weights, \
            'interaction.depth' : self.depth, \
            'shrinkage' : self.shrinkage, \
            'n.trees' : self.iterations }
        self.model=r.gbm(**self.gbm_params)
        
        #Select model components and a decision threshold
        #self.Select_Model()
        #self.Threshold(self.specificity)


    def Assign_Weights(self, method=0):
        #Weight the observations in the training set based on their distance from the threshold.
        deviation = (self.data_dictionary[self.model_target]-self.regulatory_threshold)/np.std(self.data_dictionary[self.model_target])
        
        #Integer weighting: weight is the observation's rounded-up whole number of standard deviations from the threshold.
        if method == 1: 
            weights = np.ones( len(deviation) )
            breaks = range( int( np.floor(min(deviation)) ), int( np.ceil(max(deviation)) ) )

            for i in breaks:
                #find all observations that meet the upper and lower criteria, separately
                first_slice = mlab.find(deviation > i)
                second_slice = mlab.find(deviation < i+1)
                
                #now find all the observations that meet both criteria simultaneously
                rows = filter( lambda x: x in first_slice, second_slice )
                
                #Decide how many times to replicate each slice of data
                if i<=0:
                    replicates = 0
                else:
                    replicates = 2*i
                    
                weights[rows] = replicates + 1
                
        #Continuous weighting: weight is the observation's distance (in standard deviations) from the threshold.      
        elif method == 2:
            weights = abs(deviation)

        #put more weight on exceedances
        elif method == 3:
            #initialize all weights to one.
            weights = np.ones( len(deviation) )

            #apply weight to the exceedances
            rows = mlab.find( deviation > 0 )
            weights[rows] = self.cost[1]

            #apply weight to the non-exceedances
            rows = mlab.find( deviation <= 0 )
            weights[rows] = self.cost[0]

        #put more weight on exceedances AND downweight near the threshold
        elif method == 4:
            #initialize all weights to one.
            weights = np.ones( len(deviation) )

            #apply weight to the exceedances
            rows = mlab.find( deviation > 0 )
            weights[rows] = self.cost[1]

            #apply weight to the non-exceedances
            rows = mlab.find( deviation <= 0 )
            weights[rows] = self.cost[0]

            #apply weight to the non-exceedances
            rows = mlab.find( abs(deviation) <= 0.25 )
            weights[rows] = 0

        #No weights: all weights are one.
        else: weights = np.ones( len(deviation) )
            
        return weights
            
            
            
    def Assign_Labels(self, raw):
        #Label observations as above or below the threshold.
        raw = np.array(raw >= self.regulatory_threshold, dtype=int)
        return raw
        

    def Extract(self, model_part, **args):
        try: container = args['extract_from']
        except KeyError: container = self.model
        
        #use R's coef function to extract the model coefficients
        if model_part == 'coef':
            part = r.coef(self.model, intercept=True)
        
        #otherwise, go to the data structure itself
        else:
            names = list(container.names)
            index = names.index(model_part)
            part = container[index]
            
        return part


    def Predict(self, data_dictionary):
        data_frame = utils.Dictionary_to_R(data_dictionary)
        prediction_params = {'object': self.model, 'newdata': data_frame, 'n.trees': self.iterations }
        prediction = r.predict(**prediction_params)

        #Translate the R output to a type that can be navigated in Python
        prediction = np.array(prediction).squeeze()
        
        return prediction
        

    def Validate(self, data_dictionary):
        predictions = self.Predict(data_dictionary)
        actual = data_dictionary[self.model_target]

        p = predictions

        raw = list()
    
        for k in range(len(predictions)):
            t_pos = int(predictions[k] >= 0 and actual[k] >= self.regulatory_threshold)
            t_neg = int(predictions[k] <  0 and actual[k] < self.regulatory_threshold)
            f_pos = int(predictions[k] >= 0 and actual[k] < self.regulatory_threshold)
            f_neg = int(predictions[k] <  0 and actual[k] >= self.regulatory_threshold)
            raw.append([t_pos, t_neg, f_pos, f_neg])
        
        raw = np.array(raw)
        
        return raw


    def Plot(self, **plotargs ):
        try:
            ncomp = plotargs['ncomp']
            if type(ncomp)==str: plotargs['ncomp']=self.ncomp
                
        except KeyError: pass
        
        r['''dev.new''']()
        r.plot(self.model, **plotargs)

