"""
Main file of the program

Run analysis scripts from this file. 
I cannot seem to run files if they are inside the "analysis" folder, so this is a workaround

"""

from analysis import qp_training as qp


if __name__ == "__main__":
    file_name = "samples/JetToyHIResultSoftDropSkinny.root"
    qp.train(file_name)
