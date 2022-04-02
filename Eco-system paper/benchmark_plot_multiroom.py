# Eco-system paper - (c) 2021 Olivier Moulin, Amsterdam Vrije Universiteit 

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 3,71,89,122,145,168,200,235,263,296,314,344,358,371,392,417,447,483,507,537,568,600,637

ibac = np.array([0.02,0.07,0.16,0.21,0.21,0.33,0.41,0.41,0.50,0.53,0.57,0.59,0.59,0.64,0.59,0.65,0.70,0.73,0.75,0.73,0.74,0.76,0.74])
ibac2 = np.array([0.01,0.02,0.01,0.05,0.13,0.14,0.21,0.24,0.30,0.32,0.36,0.45,0.50,0.53,0.59,0.58,0.60,0.66,0.73,0.74,0.75,0.77,0.76])
ibac3 = np.array([0.01,0.15,0.23,0.30,0.30,0.36,0.39,0.49,0.58,0.67,0.70,0.72,0.73,0.75,0.74,0.76,0.75,0.76,0.77,0.75,0.77,0.77,0.76])
ibac4 = np.array([0.01,0.15,0.23,0.65,0.69,0.71,0.72,0.73,0.76,0.76,0.76,0.74,0.76,0.73,0.78,0.75,0.75,0.76,0.76,0.75,0.77,0.77,0.77])
ibac5 = np.array([0.02,0.15,0.25,0.35,0.47,0.46,0.66,0.72,0.75,0.74,0.76,0.74,0.72,0.72,0.76,0.76,0.78,0.77,0.76,0.75,0.76,0.77,0.79])




eco = np.array([0.79,0.82,0.85,0.83,0.84,0.84,0.85,0.83,0.84,0.85,0.84,0.82,0.83,0.85,0.83,0.85,0.84,0.84,0.84,0.85,0.84,0.83,0.84])
eco2 = np.array([0.77,0.84,0.84,0.84,0.84,0.84,0.85,0.83,0.85,0.83,0.83,0.83,0.84,0.85,0.84,0.84,0.84,0.84,0.86,0.85,0.83,0.83,0.83])
eco3 = np.array([0.82,0.83,0.85,0.83,0.83,0.84,0.84,0.84,0.83,0.83,0.82,0.82,0.83,0.85,0.84,0.85,0.84,0.83,0.84,0.84,0.83,0.84,0.84])
eco4 = np.array([0.79,0.84,0.84,0.85,0.83,0.84,0.84,0.84,0.85,0.84,0.85,0.85,0.84,0.83,0.86,0.84,0.85,0.84,0.83,0.84,0.83,0.84,0.84])
eco5 = np.array([0.83,0.83,0.82,0.83,0.83,0.84,0.83,0.86,0.84,0.85,0.85,0.84,0.83,0.84,0.85,0.85,0.84,0.84,0.84,0.83,0.84,0.83,0.84])

plt.title("Mean return on new environments")
plt.xticks([0,4,8,12,16,20],[3,145,263,358,447,568],rotation=0)
plt.ylabel('Avg. reward')
plt.xlabel('# of training steps (x100000)')  
plt.plot(eco,color='blue',label='Eco-system')
plt.plot(eco2,color='blue',label='Eco-system')
plt.plot(eco3,color='blue',label='Eco-system')
plt.plot(eco4,color='blue',label='Eco-system')
plt.plot(eco5,color='blue',label='Eco-system')
plt.plot(ibac,color='red',label='IBAC-SNI')
plt.plot(ibac2,color='red',label='IBAC-SNI')
plt.plot(ibac3,color='red',label='IBAC-SNI')
plt.plot(ibac4,color='red',label='IBAC-SNI')
plt.plot(ibac5,color='red',label='IBAC-SNI')
red_patch = mpatches.Patch(color='red', label='single-agent')
blue_patch = mpatches.Patch(color='blue', label='eco-system')
plt.legend(handles=[red_patch,blue_patch])
plt.savefig('Benchmark_multiroom.png')
plt.close()