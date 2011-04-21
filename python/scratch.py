from timeseries_pkg import timeseries, timestamp
import numpy as np
import matplotlib.mlab as mlab

def Get_Input(data_source, **args):
    #Read a time series from the source and timestamp each observation.
    
    if 'NA_flag' in args:
        NA_flag = args['NA_flag']
    else: NA_flag = -1.23e25
    
    if 'dayfirst' in args:
        dayfirst = args['dayfirst']
    else: dayfirst = False
    
    if 'yearfirst' in args:
        yearfirst = args['yearfirst']
    else: yearfirst = False
    
    #Establish the data source and read the headers
    source = Source(data_source)
    headers = source.headers
    print headers
    
    #Now use the headers to figure out how we're going to turn the date/time columns into a timestamp.
    stamper = timestamp.Stamper(headers, dayfirst=dayfirst, yearfirst=yearfirst)
    
    #We will remove the columns that are used to generate the timestamp.
    cols = range( len(headers) )
    for c in stamper.time_columns:
        cols.remove(c)
    
    #define a few of objects we'll use later on.
    timestamps = list()
    data = data = list()
    finished = False
    i=0
    
    #loop until the end of the file:
    while not finished:
        line = source.Next()
        i+=1
        print i
        #continue unless we're at the end of the file
        if type(line) == float: #should pick up a nan (type 'list' is good)
            finished = True
        else:
            #generate a timestamp
            stamp = stamper.Parse(line)
            timestamps.append(stamp)
            
            #filter out the columns we used to generate the timestamp
            data_row = []
            for c in cols:
                data_row.append( float(line[c] or np.nan) )
            
            #add this row of data to the big list.
            data_row = np.array(data_row)
            data.append(data_row)
   
   
    #make the big list into a big array
    data = np.array(data, dtype=float)
    data_shape = data.shape
    
    #eiminate any values flagged as NaN
    data = data.flatten()
    NAs = mlab.find(data == NA_flag)
    data[NAs] = np.nan
    data.shape = data_shape
    
    #set up the object data
    series = {}
    series['timestamps'] = np.array(timestamps)
    series['data'] = data
    series['headers'] = list( np.array(headers)[cols] )      
    
    return series   
    
class Source():
    def __init__(self, source):
    
        self.source = source
    
        if type(source) == dict:
            self.row = 0
            self.headers = source.keys()
            self.Next = self.__dict_next__
            
            
        elif type(source) == str:
            self.file = open(source, 'r')
            self.headers = self.file.readline().rstrip('\n').split(',')
            self.Next = self.__file_next__
            
            
    def __dict_next__(self):
        try: next = np.array(self.source.values()).transpose()[self.row]
        except IndexError: next = np.nan
        
        self.row += 1
        
        return next
        
        
    def __file_next__(self):
        next = self.file.readline()
        
        if not next: next = np.nan
        else: next = next.rstrip('\n').split(',')
        
        return next
            
            