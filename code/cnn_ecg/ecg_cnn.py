import os
import wfdb

class ecg_one_data_holder():
    def __init__(self, tdir):
        self.sig, self.fields = wfdb.rdsamp(tdir)

# class ecg_all_data_holder():
#     def __init__(self, tdir, control_list):
#         self.tdir = tdir
        
