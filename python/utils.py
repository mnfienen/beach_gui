import random
import numpy as np
import datetime
import matplotlib.mlab as mlab
import rpy2.robjects as rob
from matplotlib import dates
from dateutil import parser, relativedelta


#Fire up the interface to R
r = rob.r
r.library("pls")





#np.loadtxt uses 'converters' to cast data to floats. Here we generate one (simple!) converter for each column of an input file.
def Make_Converter(headers):
 converter = dict()
 for i in range(len(headers)):
  converter[i] = lambda val: Converter(val)
 return converter


#Try to package val as a date/time, then as a time, then as a float.
def Converter(val):
    if val.count('.') == 2:
        return dates.datestr2num(val)
    else:
        try:
            val.index('/')
            return dates.datestr2num(val)
        except ValueError:
            try:
                val.index(':')
                return dates.datestr2num(val) % 1
            except ValueError:
                try: return float(val or 'nan')
                except ValueError: return np.nan





#Move a dictionary into an R data frame
def Dictionary_to_R(data_dictionary):
 data_frame = r['data.frame'](**data_dictionary)
 return data_frame


 
 
 #draw random numbers without replacement
 def Draw(count, max):
    result = []
    iterations = range(count)
    for i in iterations:
        index = int(max * random.random())
        if index in result: iterations.append(1)
        else: result.append(index)

    return result
 
 
 
 
 
 

#Find the value at the specified quantile of the list.
def Quantile(list, q):
	if q>1 or q<0:
		return np.nan
	else:
		list = np.sort(list)
		position = np.floor(q * (len(list)-1) )
		
		#if len(list) > position+1 : position += 1
		
		return list[position]
	
    
    



def Non_Standard_Deviation(list, pivot):
    var = 0
    for item in list:
        var = var + (item-pivot)**2
        
    return np.sqrt( var/len(list) )
    
    
    



def Import_File(file_name):
    '''Import a ULP-type data file'''

    #open the data source file and read the headers
    infile = file_name
    f = open(infile, 'r')
    headers = f.readline().rstrip('\n').split(',')
    
    #get an array of the valid headers
    h = np.array(headers)
    indx = mlab.find(h != '')
    
    #strip the blank headers and the 'Date' header
    headers = list(h[indx])
    headers = flatten(['year', 'julian', headers[1:]])
    
    #define a couple of objects we'll use later on.
    data = list()
    finished = False
    
    #loop until the end of the file:
    while not finished:
        line = f.readline()
        
        #continue unless we're at the end of the file
        if not line:
            finished = True
        else:
            values = line.rstrip('\n').split(',')
            
            #only process data that has some value in the first field
            if not values[0]: pass
            else:
                #get only those columns with a valid header
                v = np.array(values)
                v = v[indx]
                values = list(v)
                
                #convert the date into our expected form
                date_obj = Date_Objectify(values[0])
                julian = Julian(date_obj)
                
                #flatten the data into a numpy array (from a list of lists)
                data_row = flatten([date_obj.year, julian, values[1:]])
                data_row = np.array(data_row)
                
                #add this row of data to the big list.
                data.append(data_row)
    
    #make the big list into a big array
    data = np.array(data)
    
    txt_indx = mlab.find(data[:,-2] == 'Sunny')
    data[txt_indx,-2] = 0
    
    blank_indx = mlab.find(data == '')
    blank_rows = list( np.unique(blank_indx // data.shape[1]) )
    blank_rows.reverse()
    all_rows = range(data.shape[0])
    
    for row in blank_rows:
        all_rows.remove(row)
    
    data = data[all_rows,:]
    data = data.astype(float)
    
    return [headers, data]






def Read_CSV(file, NA_flag='NA'):
    '''Read a csv data file and return a list that \nconsists of the column headers and the data'''
    infile = file

    headers = open(infile, 'r').readline().lower().rstrip('\n').split(',')
    data_in = np.loadtxt(fname=infile, skiprows=1, dtype=float, unpack=False, delimiter=',', converters=Make_Converter(headers))
    data_in[ mlab.find(data_in==NA_flag) ] = np.nan
    
    #remove any rows with NaN's
    data_length = len(data_in)
    data_width = np.shape(data_in)[1]
    nan_rows = mlab.find( np.isnan(data_in) ) // data_width
    nan_cols = mlab.find( np.isnan(data_in) ) % data_width
    
    rows = np.array( range(data_length) )
    mask = np.ones( data_length, dtype=bool )
    mask[nan_rows] = False
    
    data_in = data_in[mask]

    return [headers, data_in]



    
def Write_CSV(array, columns, location):
    '''Creates a .csv file out of the contents of the array.'''
    
    out_file = open(location, 'w')
    
    for item in range( len(columns) ):
        out_file.write(columns[item])
        if item < len(columns)-1: out_file.write(',')
        else: out_file.write('\n')
        
    for row in range( array.shape[0] ):
        for item in range( array.shape[1] ):
            out_file.write( str(array[row,item]) )
            if item < array.shape[1]-1: out_file.write(',')
            else: out_file.write('\n')
    
    out_file.close()







def Partition_Data(data, headers, year):
    '''Partition the supplied data set into training and validation sets'''    

    #model_data is the set of observations that we'll use to train the model.
    rows_year = mlab.find(data[:,headers.index('year')]>=year)
    model_data = np.delete(data, rows_year, axis=0)
    
    #validation_data is the set of observations we'll use to test the model's predictive ability.
    rows_year = mlab.find(data[:,headers.index('year')]==year)
    validation_data = data[rows_year,:]
    
    model_dict = dict(zip(headers, np.transpose(model_data)))
    validation_dict = dict(zip(headers, np.transpose(validation_data)))

    return [model_data, validation_data]



 
def Time_Objectify(time_string):
     [hour, minute, second] = time_string.split(':')
     time_obj = datetime.time(hour=hour, minute=minute, second=second)
     return time_obj
 
 


def Date_Objectify(date_string):
    '''Create date a date object from from a date string'''
    try:
        #Try the easy way first
        date_obj = parser.parse(date_string)
        
    #Tokenize the string and make sure the tokens are integers
    except ValueError:
        try:
            date_string.index('/')
            values = date_string.split('/')
        except ValueError:
            date_string.index('.')
            values = date_string.split('.')
            values = map(int, values)
        
        #Create and return the date object
        try:
            date_obj = datetime.date(month=values[0], day=values[1], year=values[2])
        except ValueError:
            date_obj = datetime.date(month=values[1], day=values[2], year=values[0])
            
    return date_obj
 
 

def Julian(date):
    '''Get the number of days since the start of this year'''
    year_start = datetime.date(month=1, day=1, year=date.year)
    julian = (date.date() - year_start).days + 1
    return julian

 
 
 
 


 
#flatten a list of lists with a call the the recursive function flatten_backend
def flatten(unflat):
    flat = list()
    flatten_backend(l=unflat, flat=flat)
    return flat

 
 
 
 
 
def flatten_backend(l, flat):
    if isinstance(l, list):
        for item in l: flatten_backend(l=item, flat=flat)
    else:
        flat.append(l)

  
  
  
  
  
#Create a new dictionary that combines data from the matching keys of two separate dictionaries
def Match_Dictionaries(dict1, dict2):
    dict_matched = dict()
    
    for key in dict1.keys():
        if key in dict2:
            val_list = [list(dict1[key]), list(dict2[key])]
            val_list = flatten(val_list)
            dict_matched[key] = val_list

    return dict_matched

 
 
 
 

#Create a new data structure from the matching headers of two separate data structures
def Match_Data(struct1, struct2):
    #unpack the parameters
    [headers1, array1] = struct1
    [headers2, array2] = struct2
    
    #create new lists that we will fill
    matched_headers = list()
    matched_values = list()
    
    #find the headers that appear in both structures
    for col in headers1:
        if col in headers2:
            #combine data from the matching columns
            val_list = [list(array1[:,headers1.index(col)]), list(array2[:,headers2.index(col)])]
            val_list = flatten(val_list)
            
            matched_headers.append(col)
            matched_values.append(val_list)
    
    #turn the combined data into an array and return it with the headers
    matched_values = np.transpose( np.array(matched_values) )
    return [matched_headers, matched_values]



def flatten(x):
    """flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
    
    
