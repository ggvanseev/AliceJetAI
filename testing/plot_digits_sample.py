"""
Simple tool to plot a single digit sample.
"""

# plot digits sample
from testing.plotting_test import create_sample_plot
from testing_functions import load_digits_data

### --- User input --- ###
# Set hyper space and variables
digit = "9"  # digit to plot
index = 49  # index of digit to plot from dict
print_dataset_info = True  # plot info of dataset

# file_name(s) - comment/uncomment when switching between local/Nikhef
train_file = "samples/pendigits/pendigits-orig.tra"
test_file = "samples/pendigits/pendigits-orig.tes"
names_file = "samples/pendigits/pendigits-orig.names"
file_name = train_file + "," + test_file

# get digits data
train_dict = load_digits_data(train_file, print_dataset_info=print_dataset_info)
test_dict = load_digits_data(test_file)

# mix "0" = 90% as normal data with "9" = 10% as anomalous data
train_data = train_dict["0"][:675] + train_dict["9"][75:150]
# print('Mixed "9": 675 = 90% of normal data with "0": 75 = 10% as anomalous data for a train set of 750 samples')
test_data = test_dict["0"][:360] + test_dict["9"][:40]
# print('Mixed "0": 360 = 90% of normal data with "9": 40 = 10% as anomalous data for a test set of 400 samples')

create_sample_plot(train_dict[digit], index, label=digit)
