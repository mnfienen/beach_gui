import sys
sys.path.insert(0,'../../python')

#Import the PLS modeling classes
from modeling_pkg import pls, logistic, gbm, pls_parallel
import utils

import numpy as np
import matplotlib.mlab as mlab
import copy

column_names = ['year', 'threshold', 'balance', 'specificity', 'split date', 'true pos', 'true neg', 'false pos', 'false neg', 'total']
boosting_iterations = 2000

def Test(infile, target, method, outfile='', year=0, NA_flag=99999999, **args):
    '''This is the main function in the script. It uses the PLS modeling classes to build a predictive model.'''
    
    [headers, data] = utils.Read_CSV(infile, NA_flag)
    
    if year==0:
        validate = False
        years = np.unique( data[:,headers.index('year')] )
    else: years = np.array( utils.flatten([year]) )
    
    summary = []
    
    for year in years:
        '''Create the model (training) and validation dictionaries'''
        [model_data, validation_data] = utils.Partition_Data(data, headers, year)

        model_dict = dict( zip(headers, np.transpose(model_data)) )
        [model_dict, factor_levels] = utils.Factorize(model_dict, target='year', return_levels=True)

        validation_dict = dict( zip(headers, np.transpose(validation_data)) )
        validation_dict = utils.Factorize(validation_dict, target='year', levels=factor_levels)

        data_dict = dict( zip(headers, np.transpose(data)) )
        data_dict = utils.Factorize(data_dict, target='year')

        if method.lower()=='pls':
            '''Find the best model for the specified year with the specified parameters'''
            s = PLS_Models(model_dict, validation_dict, target, year=year, **args)
            for ss in s: summary.append( ss )

        elif method.lower()=='logistic':
            '''Find the best model for the specified year with the specified parameters'''
            s = Logistic_Model(model_dict, validation_dict, target, year=year, **args)
            for ss in s: summary.append( ss )

        elif method.lower()=='boosting':
            '''Find the best model for the specified year with the specified parameters'''
            s = GBM_Model(model_dict, validation_dict, target, year=year, **args)
            for ss in s: summary.append( ss )

        
    result = np.array(summary)[0]
    columns = ['year', 'threshold', 'balance', 'specificity', 'split date', 'true pos', 'true neg', 'false pos', 'false neg', 'total']    

    '''for r in np.array(summary)[1:]:
        result = np.vstack((result, r))

    
    if outfile: #If an output file has been specified, write the results there
        utils.Write_CSV(result, columns, outfile)'''

    
    return [columns, summary, data_dict]




    
    
    
def PLS_Models(model_dict, validation_dict, target, **args):
    '''This section of code will create and test the prospective models'''
    
    '''Pick the model building parameters out of args'''
    try: break_flag = args['break_flag']   #Decide whether or not we are going to test models that include a midseason split
    except KeyError: break_flag = 2 

    try: limits=utils.flatten([args['specificity']])
    except KeyError: limits = np.arange(11.)/100 + 0.85     #Default: test specificity limits from 0.85 to 0.95 
    
    try: threshold=utils.flatten([args['threshold']])
    except KeyError: threshold=[0,1]                          #Default: threshold by counts

    if break_flag != 1: model=pls.Model( model_dict, model_target=target.lower() )
    if break_flag != 0: mw=pls.Model_Wrapper( data=model_dict, model_target=target.lower() )

    results = list()

    #Test models w/ midseason split
    for spec_lim in limits:
        for threshold_method in threshold:
            if threshold_method==0: balance_method=1 #
            else: balance_method=0
            
            if break_flag != 0:
                '''mw.Generate_Models(breaks=1, specificity=spec_lim, wedge='julian', threshold_method=threshold_method, balance_method=balance_method)
                imbalance = mw.imbalance
                split_index = mlab.find(mw.imbalance[:,1] == np.min(mw.imbalance[:,1]))'''
                
                imbalance = pls_parallel.Tune_Split(mw, specificity=spec_lim, wedge='julian', threshold_method=threshold_method, balance_method=balance_method)
                split_index = mlab.find(imbalance[:,1] == np.min(imbalance[:,1]))
                
                for split in imbalance[split_index,0]:
                    mw.Split(wedge='julian', breakpoint=split)
                    mw.Assign_Thresholds(threshold_method=threshold_method, specificity=spec_lim)
                    
                    summary = Summarize(mw, validation_dict, **args)
                    summary.insert( 1, balance_method)
                    summary.insert( 1, threshold_method)
                    
                    results.append( summary )
              
    #Test models w/o midseason split
    if break_flag != 1:
        for spec_lim in limits:
            model.Threshold(specificity=spec_lim)
            
            summary = Summarize(model, validation_dict, **args)
            summary.insert(1, np.nan)
            summary.insert(1, np.nan)
                
            results.append( summary )
            
            
    return results
    
    
    

    
def Logistic_Model(model_dict, validation_dict, target, **args):
    '''This section of code will create and test the prospective models'''
    
    '''Pick the model building parameters out of args'''
    try: weights = list( args['weights'] )   #Logistic regression, weighted away from the threshold.
    except KeyError: weights = ['discrete']

    try: limits=utils.flatten([args['specificity']])
    except KeyError: limits = np.arange(11.)/100 + 0.85     #Default: test specificity limits from 0.85 to 0.95 

    results = list()

    #Test models w/ midseason split
    for weight in weights:
        for limit in limits:
        
            l=logistic.Model(model_dict, target,  specificity=limit, weights=weight)
                
            summary = Summarize(l, validation_dict, **args)
            summary.insert( 1, weight)
            summary.insert( 1, np.nan)
            
            results.append( summary )

    return results


def GBM_Model(model_dict, validation_dict, target, **args):
    '''This section of code will create and test the prospective models'''
    
    '''Pick the model building parameters out of args'''
    try: weights = list( args['weights'] )   #Gradient boosting, with exceedances given more weight.
    except KeyError: weights = ['both']

    try: costs = list( args['specificity'] )   #Gradient boosting, with exceedances given more weight.
    except KeyError: costs = [1]

    costs = [ [1,i] for i in costs ]

    results = list()

    #Test models w/ midseason split
    for weight in weights:
        for cost in costs:
        
            l=gbm.Model(model_dict, target,  cost=cost, weights=weight, iterations=boosting_iterations)
                
            summary = Summarize(l, validation_dict, **args)
            summary.insert( 1, weight)
            summary.insert( 1, np.nan)
            
            results.append( summary )

    return results

    

def Make_Model(infile, target, params, year=''):
    '''Create a PLS model with the specified params and return it'''

    
    params = dict( zip(column_names, params) )

    if year: params['year']=year #If a year is specified in the call to Make_Model(), then produce a model for prediction in that year.

    [headers, data] = utils.Read_CSV(infile)

    [model_data, validation_data] = utils.Partition_Data(data, headers, params['year'])
    model_dict = dict( zip(headers, np.transpose(model_data)) )
    validation_dict = dict( zip(headers, np.transpose(validation_data)) )

    if np.isnan(params['split date']):
        model = pls.Model(model_dict, target.lower(), specificity=params['specificity'])
    else:
        model = pls.Model_Wrapper(model_dict, target.lower())
        model.Generate_Models(breaks=1, specificity=params['specificity'], balance_method=params['balance'], breakpoint=params['split date'], threshold_method=params['threshold'])

    return model











def Summarize(model, validation_dict, **args):
    #Summarize the prediction results
    raw = model.Validate(validation_dict)
    
    if hasattr(model, 'breakpoint'): split = float( model.breakpoint )
    else: split = np.nan

    spec_lim = float( model.specificity )
    
    if 'fold' in args: year = float( args['fold'] )
    elif 'year' in args: year = float( args['year'] )
    else: year = 'na'
    
    tp = float( sum(raw[:,0]) )  #True positives
    tn = float( sum(raw[:,1]) )  #True negatives
    fp = float( sum(raw[:,2]) )  #False positives
    fn = float( sum(raw[:,3]) )  #False negatives
    total = tp+tn+fp+fn
    
    return [year, spec_lim, split, tp, tn, fp, fn, total]




def Produce(params, method, data_dict, target):    
    params = dict( zip(column_names, params) )

    if method=='pls':
        if np.isnan(params['split date']):
            model=pls.Model(data_dict, target, specificity=params['specificity'])

        else:
            model = pls.Model_Wrapper(data_dict, target)
            model.Generate_Models(breakpoint=params['split date'], balance_method=params['balance'], threshold_method=params['threshold'], specificity=params['specificity'] )


    elif method=='logistic':
        model = logistic.Model(data_dict, target, weights=params['balance'], specificity=params['specificity'])


    elif method=='boosting':
        model = gbm.Model(data_dict, target, weights=params['balance'], cost=[1,params['specificity']], iterations=boosting_iterations)

    return model
    
    
    
    
    
    
    
    
    
    
    
def Test_CV(infile, target, method, outfile='', folds=10, NA_flag=99999999, **args):
    '''This is the main function in the script. It uses the PLS modeling classes to build a predictive model.'''
    
    [headers, data] = utils.Read_CSV(infile, NA_flag)
    [headers, data] = utils.Factorize(data, target='year', headers=headers)
    
    '''if folds==0:
        validate = False
        years = np.unique( data[:,headers('year')] )
    else: years = np.array( utils.flatten([year]) )'''
    
    summary = []
    
    '''Create the model (training) and validation dictionaries'''
    fold = utils.CV_Partition(data, folds)
    data_dict = dict( zip(headers, np.transpose(data)) )
    folds = np.arange(folds)+1
    
    result_dict = dict()
    
    for f in folds:
        model_data = data[fold!=f,:]
        validation_data = data[fold==f,:]

        model_dict = dict( zip(headers, np.transpose(model_data)) )
        validation_dict = dict( zip(headers, np.transpose(validation_data)) )

        if method.lower()=='pls':
            '''Find the best model for the specified fold with the specified parameters'''
            s = PLS_Models(model_dict, validation_dict, target, fold=f, **args)
            for ss in s: summary.append( ss )

        elif method.lower()=='logistic':
            '''Find the best model for the specified fold with the specified parameters'''
            s = Logistic_Model(model_dict, validation_dict, target, fold=f, **args)
            for ss in s: summary.append( ss )

        elif method.lower()=='boosting':
            '''Find the best model for the specified fold with the specified parameters'''
            s = GBM_Model(model_dict, validation_dict, target, fold=f, **args)
            for ss in s: summary.append( ss )

        result_dict[f] = s
      
    combined = result_dict[max(folds)]
    for f in folds[:-1][::-1]:
        new_combined = list()
        for combo_row in combined:
            for row in result_dict[f]:
                if ( (row[1]==combo_row[1] or (np.isnan(row[1]) and np.isnan(combo_row[1]))) and (row[2]==combo_row[2] or (np.isnan(row[2]) and np.isnan(combo_row[2]))) and (row[3]==combo_row[3] or (np.isnan(row[3]) and np.isnan(combo_row[3]))) ):
                    new_row = copy.copy(combo_row)
                    new_row[5] += row[5]
                    new_row[6] += row[6]
                    new_row[7] += row[7]
                    new_row[8] += row[8]
                    new_row[9] += row[9]
                    if not np.isnan(row[4]): new_row[4] = str(row[4]) + ', ' + str( new_row[4] )
                    new_row[0] = 'na'
                    new_combined.append( new_row )
                    
        combined = new_combined
    
    #result = np.array(combined)
    columns = ['fold', 'threshold', 'balance', 'specificity', 'split date', 'true pos', 'true neg', 'false pos', 'false neg', 'total']    

    return [columns, combined, data_dict]