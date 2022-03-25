import numpy as np

def normalize_sample(sample):
    # get min, max and dx and dy
    x_min = np.min(sample[:,0])
    x_max = np.max(sample[:,0])
    dx = x_max - x_min
    y_min = np.min(sample[:,1])
    y_max = np.max(sample[:,1]) 
    dy = y_max - y_min
    
    # choose the dominant axis for 0 to 100 scaling
    d_dominant = dy if dy > dx else dx
    
    # scale values of the sample to normalize largest range to range(0,101)
    sample[:,0] = np.rint((sample[:,0]) / (d_dominant) * 100)
    sample[:,1] = np.rint((sample[:,1] ) / (d_dominant) * 100) # - y_min * 0.5 * (1 + dy / d_dominant)
    
    # re-center
    sample[:,0] = sample[:,0] + 0.5*(100 - max(sample[:,0]) - min(sample[:,0]))
    sample[:,1] = sample[:,1] + 0.5*(100 - max(sample[:,1]) - min(sample[:,1]))
    
    return sample

def load_digits_data(data_file, print_dataset_info=False):
    # print out info stored in 'names'
    if print_dataset_info:
        names_file = data_file.split(".")[0] + ".names"
        file_string = ""
        with open(names_file, 'r') as f:
            for line in f.readlines():
                file_string += line
        print(file_string)
            
    # dict to store digit data           
    data_dict = {
        "0":[],
        "1":[],
        "2":[],
        "3":[],
        "4":[],
        "5":[],
        "6":[],
        "7":[],
        "8":[],
        "9":[],
    }
    sample_flag = False
    digit = -1  # Would be bad if it was this one
    i = 0 # Count nr of lines
    count_x_bigger = 0
    sample = []
    with open(data_file, 'r') as f:
        for line in f.readlines():
            i += 1
            if sample_flag: # if currently writing 
                if "PEN_UP" in line: # stop writing after this line
                    sample_flag = False
                else: # append writing coordinates
                    sample.append(np.fromstring(line, dtype=int, sep=" "))
            elif "PEN_DOWN" in line: # (continue) writing happening after this line
                sample_flag = True
            elif "SEGMENT DIGIT" in line: # if new sample
                if digit != -1: # check if not the first sample (or wrong digit)
                    sample = np.vstack(sample)
                    sample = normalize_sample(sample)
                    data_dict[digit].append(sample.tolist()) # append numpied/list? sample .tolist
                    # check if dx > dy 
                    dx = np.max(sample[:,0]) - np.min(sample[:,0])
                    dy = np.max(sample[:,1]) - np.min(sample[:,1])
                    if dx > dy:
                        count_x_bigger += 1 # used for normalization form 0..100 but for now I test without this normalization
                    sample = [] # empty array for new sample
                digit = line.split('\"')[1] # set current sample digit
        # save last digit
        sample = np.vstack(sample)
        data_dict[digit].append(sample) # append numpied? sample
    print(f"From: {data_file}\nObtained nr. of samples per digit:\n\tDigit:\tSamples")
    total = 0
    for key in data_dict.keys():
        current = len(data_dict[key])
        print(f"\t{key}\t{current}")
        total += current
    print(f"\tTotal:\t{total}")
    print(f"In {count_x_bigger} samples, the x-range is larger than the y-range")
    print("\n")
    
    return data_dict