import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
ibac = np.array([0.05,0.09,0.21,0.20,0.24,0.23,0.23,0.22])
ibac2 = np.array([0.04,0.25,0.27,0.22,0.28,0.24,0.30,0.29])
ibac3 = np.array([0.01,0.04,0.13,0.05,0.06,0.03,0.07,0.05])
ibac4 = np.array([0.01,0.01,0.05,0.02,0.05,0.09,0.09,0.05])
ibac5 = np.array([0.06,0.20,0.22,0.24,0.30,0.29,0.31,0.31])

min_ibac = np.minimum(ibac,ibac2)
min_ibac = np.minimum(min_ibac,ibac3)
min_ibac = np.minimum(min_ibac,ibac4)
min_ibac = np.minimum(min_ibac,ibac5)

max_ibac = np.maximum(ibac,ibac2)
max_ibac = np.maximum(max_ibac,ibac3)
max_ibac = np.maximum(max_ibac,ibac4)
max_ibac = np.maximum(max_ibac,ibac5)

avg_ibac = ibac + ibac2+ibac3+ibac4+ibac5
avg_ibac = avg_ibac /5

ibac_var = ibac - avg_ibac
ibac_var = np.square(ibac_var)
ibac_var2 = ibac2 - avg_ibac
ibac_var2 = np.square(ibac_var2)
ibac_var3 = ibac3 - avg_ibac
ibac_var3 = np.square(ibac_var3)
ibac_var4 = ibac4 - avg_ibac
ibac_var4 = np.square(ibac_var4)
ibac_var5 = ibac5 - avg_ibac
ibac_var5 = np.square(ibac_var5)

ibac_std_dev = ibac_var + ibac_var2 + ibac_var3 + ibac_var4 + ibac_var5
ibac_std_dev = ibac_std_dev/5
ibac_std_dev = np.sqrt(ibac_std_dev)

eco = np.array([0.16,0.34,0.41,0.42,0.44,0.45,0.45,0.47])
eco2 = np.array([0.18,0.39,0.43,0.43,0.45,0.41,0.43,0.45])
eco3 = np.array([0.18,0.36,0.41,0.43,0.44,0.47,0.47,0.47])
eco4 = np.array([0.11,0.36,0.43,0.43,0.45,0.43,0.43,0.46])
eco5 = np.array([0.18,0.39,0.43,0.43,0.45,0.46,0.47,0.46])

min_eco = np.minimum(eco,eco2)
min_eco = np.minimum(min_eco,eco3)
min_eco = np.minimum(min_eco,eco4)
min_eco = np.minimum(min_eco,eco5)

max_eco = np.maximum(eco,eco2)
max_eco = np.maximum(max_eco,eco3)
max_eco = np.maximum(max_eco,eco4)
max_eco = np.maximum(max_eco,eco5)

avg_eco = eco+eco2+eco3+eco4+eco5
avg_eco = avg_eco /5

eco_var = eco - avg_eco
eco_var = np.square(eco_var)
eco_var2 = eco2 - avg_eco
eco_var2 = np.square(eco_var2)
eco_var3 = eco3 - avg_eco
eco_var3 = np.square(eco_var3)
eco_var4 = eco4 - avg_eco
eco_var4 = np.square(eco_var4)
eco_var5 = eco5 - avg_eco
eco_var5 = np.square(eco_var5)

eco_std_dev = eco_var + eco_var2 + eco_var3 + eco_var4 + eco_var5
eco_std_dev = eco_std_dev/5
eco_std_dev = np.sqrt(eco_std_dev)

x=[0,1,2,3,4,5,6,7]
plt.title("Adaptability index on new environments (FourRoom)", size=15,y=1.06)
plt.xticks([0,1,2,3,4,5,6,7],[5,150,402,564,696,829,919,1000],rotation=0)
plt.ylabel('Avg. expected return', size=14)
plt.xlabel('# of training steps (x100000)',size =14)  
plt.plot(avg_eco,color='purple',label='Eco-system')
#plt.plot(max_eco,color='blue',label='Eco-system')
plt.fill_between(x, avg_eco - eco_std_dev, avg_eco + eco_std_dev,facecolor="purple",color='purple',alpha=0.2)  

plt.plot(avg_ibac,color='orange',label='IBAC-SNI')
#plt.plot(max_ibac,color='red',label='IBAC-SNI')
plt.fill_between(x, avg_ibac - ibac_std_dev, avg_ibac + ibac_std_dev,facecolor="orange",color='orange',alpha=0.2)  

red_patch = mpatches.Patch(color='orange', label='single-agent')
blue_patch = mpatches.Patch(color='purple', label='eco-system')
plt.legend(handles=[red_patch,blue_patch])
plt.savefig('Benchmark_fourroom.pdf')
plt.close()
