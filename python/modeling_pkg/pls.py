import numpy as np
import rpy2
import rpy2.robjects as rob
import rpy2.robjects.numpy2ri
import utils
import random
import copy

from matplotlib import dates
from random import choice
#from scipy import stats


import matplotlib.mlab as mlab
import matplotlib.pyplot as pyplot


#Fire up the interface to R
r = rob.r
r.library("pls")






#Generate a PLS model
def Create_Model(data_dictionary, outcome):
    data_frame = Dictionary_to_R(data_dictionary)
    formula = r['as.formula'](outcome + '~.')
    pls_params = {'formula' : formula, \
        'data' : data_frame, \
        'validation' : 'CV' }
    pls_model = Model(model_target=outcome, r_model_struct=r.plsr(**pls_params))
    return pls_model











def Validate(model, data_dictionary):
    outcome = model.target
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
    '''represents a PLS model generated in R'''

    def __init__(self, data_dictionary, model_target, cv_segments=10, **args):
        #Check to see if a threshold has been specified in the function's arguments
        try: self.threshold = args['threshold']
        except KeyError: self.threshold=2.3711   # if there is no 'threshold' key, then use the default (2.3711)

        self.target = model_target
        if 'AR_part' in args: self.AR_part = args['AR_part']
        if 'specificity' in args: specificity=args['specificity']
        else: specificity=0.9

        #Get the data into R 
        self.data_frame = utils.Dictionary_to_R(data_dictionary)
        self.data_dictionary = data_dictionary
        
        #Generate a PLS model in R.
        self.formula = r['as.formula'](self.target + '~.')
        self.pls_params = {'formula' : self.formula, \
            'data' : self.data_frame, \
            'validation' : 'LOO', \
            'x' : True }
        self.model=r.plsr(**self.pls_params)

        #Use cross-validation to find the best number of components in the model.
        self.Get_Actual()
        self.Cross_Validation(**args)
        self.Get_Fitted()
        
        #Establish a decision threshold
        self.Threshold(specificity)


    def Extract(self, model_part, **args):
        try: container = args['extract_from']
        except KeyError: container = self.model
        
        #use R's coef function to extract the model coefficients
        if model_part == 'coef':
            part = r.coef(self.model, ncomp=self.ncomp, intercept=True)
            
        #use R's MSEP function to estimate the variance.
        elif model_part == 'MSEP':
            part = np.array( r.MSEP(self.model)[0] )
            part = part[1,0,self.ncomp]
            
        #use R's RMSEP function to estimate the standard error.
        elif model_part == 'RMSEP':
            part = np.array( r.RMSEP(self.model)[0] )
            part = part[1,0,self.ncomp]
        
        #otherwise, go to the data structure itself
        else:
            names = list(container.names)
            index = names.index(model_part)
            part = container[index]
            
        return part


    def Predict(self, data_dictionary, **args):
        data_frame = utils.Dictionary_to_R(data_dictionary)
        prediction_params = {'object': self.model, 'newdata': data_frame }
        prediction = r.predict(**prediction_params)

        #Translate the R output to a type that can be navigated in Python
        prediction = np.array(prediction).squeeze()
        return prediction
        
        
    def Predict_Exceedances(self, data_dictionary):
        prediction = self.Predict(data_dictionary)
        return np.array(prediction[:,self.ncomp-1] >= self.threshold, dtype=int)
        
        
    def Predict_Probability(self, data_dictionary):
        prediction = self.Predict(data_dictionary)[:,self.ncomp-1]
        se = self.Extract('RMSEP')
        exceedance_probability = 1-r.pnorm( self.threshold-prediction/se )
        
        return exceedance_probability

        
    def Cross_Validation(self, cv_method=0, **args):
        #select ncomp by the requested CV method
        
        validation = self.Extract('validation')
       
        #method 0: select the fewest components with PRESS within 1 stdev of the least PRESS (by the bootstrap)
        if cv_method == 0: #Use the bootstrap to find the standard deviation of the MSEP
            cv = np.array( self.Extract('pred', extract_from=validation) ).squeeze()
            PRESS = map(lambda x: sum((cv[:,x]-self.actual)**2), range(cv.shape[1]))
            ncomp = np.argmin(PRESS)
            
            cv_squared_error = (cv[:,ncomp]-self.actual)**2
            sample_space = xrange(cv.shape[0])
            
            PRESS_stdev = list()
            
            #Cache random number generator and int's constructor for a speed boost
            _random, _int = random.random, int
            
            for i in np.arange(100):
                PRESS_bootstrap = list()
                
                for j in np.arange(100):
                    PRESS_bootstrap.append( sum([cv_squared_error[_int(_random()*cv.shape[0])] for i in sample_space]) )
                    
                PRESS_stdev.append( np.std(PRESS_bootstrap) )
                
            med_stdev = np.median(PRESS_stdev)
            
            #Maximum allowable PRESS is the minimum plus one standard deviation
            good_ncomp = mlab.find( PRESS < min(PRESS) + med_stdev )
            self.ncomp = np.min(good_ncomp) + 1
            
            
            
        #method 1: select the fewest components w/ PRESS less than the minimum plus a 4% of the range
        if cv_method==1:
            #PRESS stands for predicted error sum of squares
            PRESS0 = list(self.Extract('PRESS0', extract_from=validation))
            PRESS = list(self.Extract('PRESS', extract_from=validation))
    
            #the range is the difference between the greatest and least PRESS values
            PRESS_range = abs(PRESS0 - np.min(PRESS))
            
            #Maximum allowable PRESS is the minimum plus a fraction of the range.
            max_CV_error = np.min(PRESS) + PRESS_range/25
            good_ncomp = mlab.find(PRESS < max_CV_error)
    
            #choose the most parsimonious model that satisfies that criterion
            self.ncomp = np.min(good_ncomp) + 1
        

    def Threshold(self, specificity=0.92):
        self.specificity = specificity
    
        if not hasattr(self, 'actual'):
            self.Get_Actual()
        
        if not hasattr(self, 'fitted'):
            self.Get_Fitted()

        #Decision threshold is the [specificity] quantile of...
        #...the fitted values for non-exceedances in the training set.
        try:
            non_exceedances = self.fitted[mlab.find(self.actual < 2.3711)]
            self.threshold = utils.Quantile(non_exceedances, specificity)

        #This error should only happen if somehow there are no non-exceedances in the training data.
        except IndexError: self.threshold = 2.3711


    def Validate(self, validation_dict):
        target = self.target
        ncomp = self.ncomp - 1
    
        predictions = self.Predict(validation_dict)
        predictions = predictions[:, ncomp]
        actual = validation_dict[target]
    
        raw = list()
    
        for k in range(len(predictions)):
            t_pos = int(predictions[k] >= self.threshold and actual[k] >= 2.3711)
            t_neg = int(predictions[k] < self.threshold and actual[k] < 2.3711)
            f_pos = int(predictions[k] >= self.threshold and actual[k] < 2.3711)
            f_neg = int(predictions[k] < self.threshold and actual[k] >= 2.3711)
            raw.append([t_pos, t_neg, f_pos, f_neg])
        
        raw = np.array(raw)
        return raw


    def Get_Actual(self):
        #Get the fitted counts from the model.
        fitted_values = np.array(self.Extract('fitted.values'))
        fitted_values = np.squeeze(fitted_values)[:,0]
        
        #If this is the second stage of an AR model, then incorporate the AR predictions.
        if hasattr(self, 'AR_part'):
            mask = np.ones( self.AR_part.shape[0], dtype=bool )
            nan_rows = mlab.find( np.isnan(self.AR_part[:,0]) )
            mask[ nan_rows ] = False
            fitted_values += self.AR_part[mask,0]

        #Recover the actual counts by adding the residuals to the fitted counts.
        residual_values = np.array(self.Extract('residuals'))
        residual_values = np.squeeze(residual_values)[:,0]
        
        self.actual = np.array( fitted_values + residual_values ).squeeze()
        
        
    def Get_Fitted(self, **params):
        try: ncomp = params['ncomp']
        except KeyError:
            try: ncomp = self.ncomp
            except AttributeError: ncomp=1
            
        #Get the fitted counts from the model so we can compare them to the actual counts.
        fitted_values = np.array(self.Extract('fitted.values'))
        fitted_values = np.squeeze(fitted_values)[:,self.ncomp-1]
        
        #If this is the second stage of an AR model, then incorporate the AR predictions.
        if hasattr(self, 'AR_part'):
            mask = np.ones( self.AR_part.shape[0], dtype=bool )
            nan_rows = mlab.find( np.isnan(self.AR_part[:,0]) )
            mask[ nan_rows ] = False
            fitted_values += self.AR_part[mask,0]
        
        self.fitted = fitted_values
        self.residual = self.actual-self.fitted
        
        
    def Get_Influence(self):
        #Get the model terms from R's model object
        terms = self.Extract('terms')
        terms = str(terms)
        
        #Get the covariate names
        self.names = self.data_dictionary.keys()
        self.names.remove(self.target)

        #Now get the model coefficients from R.
        coefficients = np.array( self.Extract('coef') )
        coefficients = coefficients.flatten()
        
        #Get the standard deviations (from the data_dictionary) and package the influence in a dictionary.
        raw_influence = list()
        
        for i in range( len(self.names) ):
            standard_deviation = np.std( self.data_dictionary[self.names[i]] )
            raw_influence.append( abs(standard_deviation * coefficients[i+1]) )

            
        self.influence = dict( zip(raw_influence/np.sum(raw_influence), self.names) )
            
            
    def Count(self):
        #Count the number of true positives, true negatives, false positives, and false negatives.
        self.Get_Actual()
        self.Get_Fitted()
        
        #initialize counts to zero:
        t_pos = 0
        t_neg = 0
        f_pos = 0
        f_neg = 0
        
        for obs in range( len(self.fitted) ):
            if self.fitted[obs] >= self.threshold:
                if self.actual[obs] >= 2.3711: t_pos += 1
                else: f_pos += 1
            else:
                if self.actual[obs] >= 2.3711: f_neg += 1
                else: t_neg += 1
        
        return [t_pos, t_neg, f_pos, f_neg]
        
        
    def Plot(self, **plotargs ):
        try:
            ncomp = plotargs['ncomp']
            if type(ncomp)==str: plotargs['ncomp']=self.ncomp
                
        except KeyError: pass
        
        r['''dev.new''']()
        r.plot(self.model, **plotargs)





class Model_Wrapper:
    '''Contains several models that, together, cover the possible prediction space'''

    def __init__(self, data, model_target, **args):
        self.target = model_target
        self.model_frame = data
        self.model_data = np.array(data.values()).transpose()
        self.headers = data.keys()
        if 'AR_part' in args: self.AR_part = copy.copy(args['AR_part'])



    def Split(self, wedge=None, breakpoint=None, **args):
        finished = False
        i=0
        self.models = list()
        self.wedge = wedge
        self.breakpoint = breakpoint

        #Decide how many submodels we're making.
        if breakpoint is None:
            breaks=0
        else:
            breakpoint = np.array( np.sort(breakpoint), ndmin=1 )
            breaks = len(breakpoint) 
    
        #Make the submodels
        while not finished:
            if i>0: lower_bound = breakpoint[i-1]
            else: lower_bound = -np.infty
            if i<breaks:
                upper_bound = breakpoint[i]
            else:
                upper_bound = np.infty
                finished = True

            #Find which rows of data lie in this division.
            upper_rows = mlab.find( self.model_data[:,self.headers.index(wedge)] > lower_bound )
            lower_rows = mlab.find( self.model_data[:,self.headers.index(wedge)] <= upper_bound )

            #Now create a data dictionary with the data for this division.
            rows = filter( lambda x: x in upper_rows, lower_rows )
            submodel_data = self.model_data[rows,:]
            submodel_frame = dict(zip(self.headers, np.transpose(submodel_data)))
            if hasattr(self, 'AR_part'):
                args['AR_part'] = self.AR_part[rows,:]
                

            #Generate this submodel and add it to the list.
            self.f = submodel_frame
            submodel = Model(submodel_frame, self.target, **args)
            self.models.append(submodel)
            i+=1



    def Generate_Models(self, specificity=0.9, breakpoint='', breaks=0, balance_method=1, wedge='julian', **args):
        self.specificity = specificity
        self.wedge = wedge

        #Default is not to tune the split date. This will be overridden if the breakpoint is not specified.
        tune = False

        if not breakpoint:
            wedge_values = np.unique( self.model_frame[wedge] )
            med = np.median(wedge_values)
            breakpoint = med
            tune = True
        else: pass

        if breaks>0:
            self.Split(wedge, breakpoint, **args)
            self.Assign_Thresholds(**args)
            
            if tune:
                self.Tune_Split(balance_method=balance_method, **args)
            
        else:
            self.Split(wedge, **args)
            self.Assign_Thresholds(**args)
            
        self.Get_Actual()



    def Tune_Split(self, balance_method=1, **args):
        self.imbalance = list()
        possible_breaks = np.unique( self.model_frame[self.wedge] )
        
        #Sweep through the possible break points and find the imbalance at each
        for breakpoint in possible_breaks[10:-10]:
            self.Split(wedge=self.wedge, breakpoint = breakpoint, **args)
            self.Assign_Thresholds(**args)
            
            self.imbalance.append( [self.breakpoint, self.Imbalance(method=balance_method)] )
                
        
        #select the break point with the minimal imbalance
        self.imbalance = np.array(self.imbalance)
        optimal_split = np.argmin( self.imbalance[:,1] )
        breakpoint = self.imbalance[optimal_split,0]
        
        #split on the optimal break point and refit the model.
        self.Split(wedge=self.wedge, breakpoint=self.breakpoint)
        self.Assign_Thresholds(**args)
        
    
    
    def Assign_Thresholds(self, **args):
        #Assign the decision thresholds
        
        try: method=args['threshold_method']
        except KeyError: method=1
        
        try: self.specificity=args['specificity']
        except KeyError: pass
        
        if method == 0: self.Threshold_on_Proportions()
        elif method == 1: self.Threshold_on_Counts()
            
        
        
    def Threshold_on_Counts(self):
        counts = list()
        fitted_sorted = list()
        submodels = range( len(self.models) )
        winner = 0 #Begin with the first submodel

        for model in self.models:
            #Count the number of data points in the model and sort the fitted values.
            t_pos = 0.
            f_pos = 0.
            [f_neg, t_neg] = self.Initial_Counts(model)
            fit_order = list( np.argsort(model.fitted) )

            counts.append([t_pos, t_neg, f_pos, f_neg])      
            fitted_sorted.append(fit_order)

        counts = np.array(counts)
        [sensitivity, specificity] = self.Combined_Accuracy(counts)
        specificity_limit = self.specificity

        #Descend through the fitted counts to find the best decision threshold.
        while(specificity > specificity_limit):
            try:

                index = fitted_sorted[winner].pop()
                self.models[winner].threshold = self.models[winner].fitted[index]

                if (self.models[winner].actual[index] >= 2.3711): #A hit: continue to play the 'winner'
                    counts[winner,0] += 1. #t_pos
                    counts[winner,3] -= 1. #f_neg

                if (self.models[winner].actual[index] < 2.3711): #A miss: find a new 'winner'
                    counts[winner,2] += 1. #f_pos
                    counts[winner,1] -= 1. #t_neg

                    #Select a new 'winner', then repair the list of submodels.
                    last = winner
                    submodels.remove(last)
                    try: winner = choice( submodels )
                    except IndexError: winner=last
                    submodels.append(last)

                #Update the combined sensitivity, specificity.     
                [sensitivity, specificity] = self.Combined_Accuracy(counts)
            
            except IndexError:
                submodels.remove(winner)
                winner = choice( submodels )

        self.thresholding_counts = counts


    def Threshold_on_Proportions(self):
        counts = list()
        submodels = range( len(self.models) )

        for model in self.models:
            #Set the threshold of each submodel using the overall specificity limit.
            model.Threshold( self.specificity )

            counts.append(model.Count())

        counts = np.array(counts)
        self.thresholding_counts = counts


    def Initial_Counts(self, model):
        f_neg = 0.
        t_neg = 0.

        #Count the true number of exceedances, non-exceedances. Stored in the model.
        for k in range(len(model.fitted)):
            if (model.actual[k] >= 2.3711): f_neg += 1
            if (model.actual[k] < 2.3711): t_neg += 1

        return [f_neg, t_neg]



    def Combined_Accuracy(self, counts):
        t_pos = np.sum(counts[:,0])
        t_neg = np.sum(counts[:,1])
        f_pos = np.sum(counts[:,2])
        f_neg = np.sum(counts[:,3])

        try:
            specificity = t_neg/(t_neg + f_pos)
        except ZeroDivisionError:
            specificity = 0.

        try:
            sensitivity = t_pos/(t_pos + f_neg)
        except ZeroDivisionError:
            sensitivity = 1.

        return [sensitivity, specificity]



    def Imbalance(self, method=1):
        submodels = len(self.models)
        SS_model = 0
        SS_tot = 0
        errors = 0
        
        for i in range(submodels):
            if method == 0 : SS_tot += self.thresholding_counts[i,3]**2
            elif method == 1 : SS_tot += ( float(self.thresholding_counts[i,3]) / sum(self.thresholding_counts[i,:]))**2
            elif method == 2 : SS_tot += ( float(self.thresholding_counts[i,3]) / (self.thresholding_counts[i,0] + self.thresholding_counts[i,3]))**2
            
            if method == 0 : SS_model += self.thresholding_counts[i,3]
            elif method == 1 : SS_model += float(self.thresholding_counts[i,3]) / sum(self.thresholding_counts[i,:])
            elif method == 2 : SS_model += float(self.thresholding_counts[i,3]) / (self.thresholding_counts[i,0] + self.thresholding_counts[i,3])
            
            errors += self.thresholding_counts[i,2] + self.thresholding_counts[i,3]

        if method == 3 : return errors
        
        else:        
            SS_model = (SS_model**2)/submodels
            return SS_tot - SS_model
        
    
    def Predict(self, validation_frame):
        finished = False
        predictions = list()
        i=0

        validation_array = np.array(validation_frame.values()).transpose()
        validation_headers = validation_frame.keys()
    
        #Decide how many submodels we're making.
        if self.breakpoint is None:
            breaks=0
        else:
            self.breakpoint = np.array( np.sort(self.breakpoint), ndmin=1 )
            breaks = len(self.breakpoint) 
    
        #Make the submodels
        while not finished:
            if i>0: lower_bound = self.breakpoint[i-1]
            else: lower_bound = -np.infty

            if i<breaks:
                upper_bound = self.breakpoint[i]
            else:
                upper_bound = np.infty
                finished = True

            #Find which rows of data lie in this division.
            upper_rows = mlab.find( validation_array[:,validation_headers.index(self.wedge)] > lower_bound )
            lower_rows = mlab.find( validation_array[:,validation_headers.index(self.wedge)] <= upper_bound )

            #Now create a data dictionary with the data for this division.
            rows = filter( lambda x: x in upper_rows, lower_rows )
            submodel_data = validation_array[rows,:]
            submodel_frame = dict(zip(validation_headers, np.transpose(submodel_data)))

            #Make predictions on the split models  
            subseason_predictions = self.models[i].Predict(submodel_frame)[:, self.models[i].ncomp-1 ]
            predictions.extend(subseason_predictions)
            
            #Next submodel
            i += 1
            
        predictions = np.array(predictions).squeeze()
        return predictions
    
    
    
    
    def Validate(self, validation_frame, **args):
        finished = False
        self.predictions = list()
        self.prediction_residuals = list()
        i=0
        raw = list()
        validation_array = np.array(validation_frame.values()).transpose()
        validation_headers = validation_frame.keys()
           
        #Decide how many submodels we're making.
        if self.breakpoint is None:
            breaks=0
        else:
            self.breakpoint = np.array( np.sort(self.breakpoint), ndmin=1 )
            breaks = len(self.breakpoint)
    
        #Make the submodels
        while not finished:
            if i>0: lower_bound = self.breakpoint[i-1]
            else: lower_bound = -np.infty

            if i<breaks:
                upper_bound = self.breakpoint[i]
            else:
                upper_bound = np.infty
                finished = True

            #Find which rows of data lie in this division.
            upper_rows = mlab.find( validation_array[:,validation_headers.index(self.wedge)] > lower_bound )
            lower_rows = mlab.find( validation_array[:,validation_headers.index(self.wedge)] <= upper_bound )

            #Now create a data dictionary with the data for this division.
            rows = filter( lambda x: x in upper_rows, lower_rows )

            #If this CV fold has nothing on one side of the split, then return a row of zeros
            if len(rows)>0:
                submodel_data = validation_array[rows,:]
                submodel_frame = dict(zip(validation_headers, np.transpose(submodel_data)))
                #if hasattr(args, 'AR_part'): args['AR_part'] = AR_part[rows]
    
                #Make predictions on the split models  
                predictions = self.models[i].Predict(submodel_frame)[:, self.models[i].ncomp-1 ]
                if 'AR_part' in args:
                    predictions += args['AR_part'][rows]
                
                actual = submodel_frame[self.target]
                self.actual = actual
                self.i = i
                self.predictions.extend(predictions)
                residuals = actual - predictions
                self.prediction_residuals.extend(residuals)
                
                for k in range(len(predictions)):
                    self.k = k
                    t_pos = int(predictions[k] >= self.models[i].threshold and actual[k] >= 2.3711)
                    t_neg = int(predictions[k] <  self.models[i].threshold and actual[k] < 2.3711)
                    f_pos = int(predictions[k] >= self.models[i].threshold and actual[k] < 2.3711)
                    f_neg = int(predictions[k] <  self.models[i].threshold and actual[k] >= 2.3711)
                    raw.append([t_pos, t_neg, f_pos, f_neg])
        
                i+=1
                raw = np.array(raw)

            else:
                raw = np.array([[0,0,0,0]])

        return raw
        
        
    def Get_Actual(self):
        self.actual = list()
        self.fitted = list()
        self.residual = list()
        
        for m in self.models:
            self.actual.extend(m.actual)
            self.fitted.extend(m.fitted)
            self.residual.extend(m.residual)
            
            

    def Plot(self, **plotargs):
        for model in self.models:
            model.Plot(**plotargs)
