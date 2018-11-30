def readin():
    #stat_file = open('2017dataMatrix.csv','r')
    stat_file = open('./2012-2018/2012-2018_data_matrix.txt','r')
    next(stat_file, None)    #remove header
    stats=[]
    for line in stat_file:
        line = line.strip().split(",")[1:43]+line.strip().split(",")[-1:-25:-1]
        # for new input: line = line.strip().split(",")[3:]
        line = np.array(list(map(float,line)))    #only take the score part
        stats.append(line)    

    #result_file = open('2017resultsMatrix.csv','r')
    result_file = open('./2012-2018/2012-2018_results_matrix.csv','r')
    next(result_file, None)     #remove header
    results=[]
    for line in result_file:
        line = line.strip().split(",")[3:]
        line = line[-1:-3:-1]    #only take the score part
        line = np.array(list(map(float,line)))    #only take the score part
        results.append(line)    
    
    #shuffle to split
    whole_data_set = list(zip(stats,results))
    counts = len(whole_data_set)
    random.shuffle(whole_data_set)
    stat,result = zip(*whole_data_set)
    stat,result = np.array(stat), np.array(result)
    
    #normalization
    stat = (stat-stat.min(axis=0))/(stat.max(axis=0)-stat.min(axis=0))-0.5
    result = (result-result.min(axis=0))/(result.max(axis=0)-result.min(axis=0))-0.5
 
    
    X_train, X_test, Y_train, Y_test = stat[:int(0.75*counts)],stat[int(0.75*counts):],result[:int(0.75*counts)],result[int(0.75*counts):]
    return  X_train, X_test, Y_train, Y_test

