print 'Cross-validating by real bootstrap...'

            cv = np.array( self.Extract('pred', extract_from=validation) ).squeeze()
            PRESS = map(lambda x: sum((cv[:,x]-self.actual)**2), range(cv.shape[1]))
            ncomp = np.argmin(PRESS)
            
            #cv_squared_error = (cv[:,ncomp]-self.actual)**2
            cv = np.array( cv[:,np.arange(ncomp+1)], ndmin=2 )
            '''if ncomp==1: cv = cv.transpose()'''
            
            cv_squared_error = (cv - np.array(self.actual, ndmin=2).transpose())**2
            
            sample_space = xrange(cv.shape[0])
            comps = np.arange(cv.shape[1])
            
            PRESS_stdev = list()
            PRESS_med = list()
            
            #Cache random number generator and int's constructor for a speed boost
            _random, _int = random.random, int
            
            for i in np.arange(100):
                PRESS_bootstrap = list()
                
                for j in np.arange(100):
                    components = list()
                    
                    for k in comps:
                        components.append( sum([cv_squared_error[_int(_random()*cv.shape[0]),k] for i in sample_space]) )
                        
                    PRESS_bootstrap.append(components)
                
                PRESS_bootstrap = np.array(PRESS_bootstrap, ndmin=2)
                
                med = [np.median(PRESS_bootstrap[:,m]) for m in comps]
                stdev = [np.std(PRESS_bootstrap[:,m]) for m in comps]
                
                PRESS_stdev.append( stdev )
                PRESS_med.append( med )
                
            PRESS_stdev = np.array(PRESS_stdev, ndmin=2)
            PRESS_med = np.array(PRESS_med, ndmin=2)
            
            stdev = [np.median(PRESS_stdev[:,n]) for n in comps]
            med = [np.median(PRESS_med[:,n]) for n in comps]
            
            
            #Maximum allowable PRESS is the minimum plus one standard deviation
            min = np.argmin(med)
            good_ncomp = mlab.find( med < med[min] + stdev[min] )
            self.ncomp = np.min(good_ncomp) + 1
            print 'done!'