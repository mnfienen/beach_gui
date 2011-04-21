import pp
import numpy as np
import pls
import threading
import copy


class Imbalance_List(object):
    
    def __init__(self):
        self.imbalance = list()
        self.thread_lock = threading.Lock()

    
    def Append(self, row):
        self.thread_lock.acquire()
        self.imbalance.append( row )
        self.thread_lock.release()





def Tune_Split(model, specificity, balance_method=1, threshold_method=1, wedge='julian'):

    # tuple of all parallel python servers to connect with
    ppservers = ()
    #ppservers = ("localhost",)

    #Establish the job server that will handle the parallel, async jobs
    pp_server = pp.Server(ppservers=ppservers)
    pp_jobs = list()
    
    #List the ways that we'll try splitting the data
    params=[]
    for breakpoint in np.unique( model.model_frame[wedge] )[10:-10]:
        params.append( (copy.copy(model), balance_method, threshold_method, specificity, wedge, breakpoint) )

    #Compute the imbalance for each breakpoint in parallel, asynchronously:
    imbalance = Imbalance_List()
    for item in params:
        pp_server.submit(Tuning, item, callback=imbalance.Append, modules=())
        
    #Wait for the jobs to return.
    pp_server.wait()
    
    #select the break point with the minimal imbalance
    imbalance = np.array(imbalance.imbalance)
    return imbalance
    '''optimal_split = np.argmin( imbalance[:,1] )
    
    model.breakpoint = imbalance[optimal_split,0]
    
    
    #split on the optimal break point and refit the model.
    model.Split(wedge=model.wedge, breakpoint=model.breakpoint)
    model.Assign_Thresholds(**args)'''

    
def Tuning(model, balance_method, threshold_method, specificity, wedge, breakpoint):

    model.Generate_Models(specificity, breakpoint, breaks=1, wedge=wedge, threshold_method=threshold_method)
    imbalance = model.Imbalance(balance_method)
    
    return [breakpoint, imbalance]

