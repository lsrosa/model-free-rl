import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import os

r_files = glob.glob('models/rewards*.npy')

episodes = [int(re.findall(r'\d+', file_name)[0]) for file_name in r_files]

# get the rewards for the one with most episodes
n = max(episodes)

# make folder for figures
fig_path = os.getcwd()+'/plots'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
    print('creating: ', fig_path)

plt.show()
plt.figure()
r = np.load('models/rewards%d.npy'%n)
plt.plot(r)
plt.savefig(fig_path + '/rewards%d.png'%n)
