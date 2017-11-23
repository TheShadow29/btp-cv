# This file downloads the mimic3 matched waveform database to
# a particular specified folder. Note that the complete database
# is around 2.4 TB. It is divided into 10 folders, and each is nearly
# 250GB of data.
# Currently downloading around 20GB data from p00
import wfdb
import numpy as np
import pdb

with open('/home/SharedData/Ark_git_files/mimic-code/mimic3_matcheddb/RECORDS-waveforms.txt') as f:
    all_lines = f.readlines()
    record_waveforms = [l.strip() for l in all_lines]
    record_numerics = [l.strip() + 'n' for l in all_lines]
with open('/home/SharedData/Ark_git_files/mimic-code/mimic3_matcheddb/RECORDS-numerics.txt') as f:
    all_nlines = f.readlines()
    all_record_numerics = [l.strip() for l in all_nlines]
# pdb.set_trace()
# Getting top 2000 person data
n = 1000
batch = 4
record_wf_to_dl = record_waveforms[n:n+n]
record_n_to_dl = record_numerics[n:n+n]
db = 'mimic3wdb/matched'
dl_dir = '/home/SharedData/Ark_git_files/mimic-code/mimic3_matcheddb/'
numeric_problems = list()
completed = 0
for i in range(n//batch):
    # rec_batch = record_wf_to_dl[i*batch:(i+1)*batch]
    rec_n_batch = record_n_to_dl[i*batch:(i+1)*batch]
    rec_n_batch_fin = list()
    rec_batch_fin = list()
    idx = np.arange(batch)
    for j in range(batch):
        if rec_n_batch[j] in record_n_to_dl:
            rec_batch_fin.append(record_wf_to_dl[i*batch + j])
            rec_n_batch_fin.append(rec_n_batch[j])
    completed += len(rec_batch_fin)
    try:
        wfdb.dldatabase(db, dl_dir, records=rec_batch_fin)
        wfdb.dldatabase(db, dl_dir, records=rec_n_batch_fin)
    except Exception as e:
        numeric_problems.append([rec_batch_fin, rec_n_batch_fin])
        pass
    # try:
    #     wfdb.dldatabase(db, dl_dir, records=record_n_to_dl)
    # except Exception as e:
    #     numeric_problems.append()
    #     continue
    print('DL completed', completed)
