from utils import io_util

data = io_util.load('gaia.pkl')

idxs=list(data.keys())
length = len(idxs)
i=0
while i < length:
    sub_len = int(length / 10)
    sub_idxs=idxs[i:i+sub_len]
    tmp_dict = {key: data[key] for key in sub_idxs}
    io_util.save(f'pkl/{i}.pkl', tmp_dict)
    i = i+sub_len
    del tmp_dict
