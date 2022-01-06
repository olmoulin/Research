import numpy as np
np.random.seed(1231231)

from datetime import datetime
from scipy.special import softmax
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from PIL import Image

import time

def generate_results_comparison():
    plt.title("Training time on initial environments (Submarine hard)", size=15,y=1.06)
    x = [1,2,3,4,5,6]
    plt.xticks([1,2,4,6],[50,100,200,300],rotation=0)
    plt.ylabel('seconds', size=14)
    plt.xlabel('# of environment trained' , size=14)
    OAMT_training_time_0 = np.load('One_agent_multi_training_time0.npy')
    MAOT_training_time_0 = np.load('Multi_agent_one_training_time0.npy')
    OAMT_training_time_1 = np.load('One_agent_multi_training_time1.npy')
    MAOT_training_time_1 = np.load('Multi_agent_one_training_time1.npy')
    OAMT_training_time_2 = np.load('One_agent_multi_training_time2.npy')
    MAOT_training_time_2 = np.load('Multi_agent_one_training_time2.npy')
    OAMT_training_time_3 = np.load('One_agent_multi_training_time3.npy')
    MAOT_training_time_3 = np.load('Multi_agent_one_training_time3.npy')
    OAMT_training_time_4 = np.load('One_agent_multi_training_time4.npy')
    MAOT_training_time_4 = np.load('Multi_agent_one_training_time4.npy')
    OAMT_training_time_0 = np.delete(OAMT_training_time_0,0)
    MAOT_training_time_0 = np.delete(MAOT_training_time_0,0)
    OAMT_training_time_1 = np.delete(OAMT_training_time_1,0)
    MAOT_training_time_1 = np.delete(MAOT_training_time_1,0)
    OAMT_training_time_2 = np.delete(OAMT_training_time_2,0)
    MAOT_training_time_2 = np.delete(MAOT_training_time_2,0)
    OAMT_training_time_3 = np.delete(OAMT_training_time_3,0)
    MAOT_training_time_3 = np.delete(MAOT_training_time_3,0)
    OAMT_training_time_4 = np.delete(OAMT_training_time_4,0)
    MAOT_training_time_4 = np.delete(MAOT_training_time_4,0)

    min_OAMT = np.minimum(OAMT_training_time_0,OAMT_training_time_1)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_time_2)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_time_3)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_time_4)
    
    max_OAMT = np.maximum(OAMT_training_time_0,OAMT_training_time_1)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_time_2)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_time_3)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_time_4)
    
    avg_OAMT = OAMT_training_time_0 + OAMT_training_time_1+OAMT_training_time_2+OAMT_training_time_3+OAMT_training_time_4
    avg_OAMT = avg_OAMT / 5
    
    OAMT_var = OAMT_training_time_0 - avg_OAMT
    OAMT_var = np.square(OAMT_var)
    OAMT_var2 = OAMT_training_time_1 - avg_OAMT
    OAMT_var2 = np.square(OAMT_var2)
    OAMT_var3 = OAMT_training_time_2 - avg_OAMT
    OAMT_var3 = np.square(OAMT_var3)
    OAMT_var4 = OAMT_training_time_3 - avg_OAMT
    OAMT_var4 = np.square(OAMT_var4)
    OAMT_var5 = OAMT_training_time_4 - avg_OAMT
    OAMT_var5 = np.square(OAMT_var5)
    
    OAMT_std_dev = OAMT_var + OAMT_var2 + OAMT_var3 + OAMT_var4 + OAMT_var5
    OAMT_std_dev = OAMT_std_dev/5
    OAMT_std_dev = np.sqrt(OAMT_std_dev)
    
    min_MAOT = np.minimum(MAOT_training_time_0,MAOT_training_time_1)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_time_2)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_time_3)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_time_4)
    
    max_MAOT = np.maximum(MAOT_training_time_0,MAOT_training_time_1)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_time_2)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_time_3)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_time_4)        

    avg_MAOT = MAOT_training_time_0 + MAOT_training_time_1+MAOT_training_time_2+MAOT_training_time_3+MAOT_training_time_4
    avg_MAOT = avg_MAOT / 5

    MAOT_var = MAOT_training_time_0 - avg_MAOT
    MAOT_var = np.square(MAOT_var)
    MAOT_var2 = MAOT_training_time_1 - avg_MAOT
    MAOT_var2 = np.square(MAOT_var2)
    MAOT_var3 = MAOT_training_time_2 - avg_MAOT
    MAOT_var3 = np.square(MAOT_var3)
    MAOT_var4 = MAOT_training_time_3 - avg_MAOT
    MAOT_var4 = np.square(MAOT_var4)
    MAOT_var5 = MAOT_training_time_4 - avg_MAOT
    MAOT_var5 = np.square(MAOT_var5)

    MAOT_std_dev = MAOT_var + MAOT_var2 + MAOT_var3 + MAOT_var4 + MAOT_var5
    MAOT_std_dev = MAOT_std_dev/5
    MAOT_std_dev = np.sqrt(MAOT_std_dev)

    plt.plot(x,avg_MAOT, color='purple', label='eco-system duration')
#    plt.plot(max_MAOT, color='blue', label='eco-system duration')
    plt.fill_between(x,avg_MAOT-MAOT_std_dev, avg_MAOT+MAOT_std_dev,facecolor="purple", color='purple',alpha=0.2)  
    plt.plot(x,avg_OAMT, color='orange', label='one-agent duration')
 #   plt.plot(max_OAMT, color='red', label='one-agent duration')
    plt.fill_between(x,avg_OAMT-OAMT_std_dev, avg_OAMT+OAMT_std_dev,facecolor="orange", color='orange',alpha=0.2)  
    red_patch = mpatches.Patch(color='orange', label='single-agent')
    blue_patch = mpatches.Patch(color='purple', label='eco-system')
    plt.legend(handles=[red_patch,blue_patch])
    plt.savefig('Architecture_comparison_duration_hard.pdf')
    plt.close()

    plt.title("Catastrophic forgetting avoidance index (Submarine hard)", size=15,y=1.06)
    x = [1,2,3,4,5,6]
    plt.xticks([1,2,4,6],[50,100,200,300],rotation=0)
    plt.ylabel('% accuracy', size=14)
    plt.xlabel('# of environment trained', size=14)
    OAMT_training_forget_0 = np.load('One_agent_multi_training_forget0.npy')
    MAOT_training_forget_0 = np.load('Multi_agent_one_training_forget0.npy')
    OAMT_training_forget_1 = np.load('One_agent_multi_training_forget1.npy')
    MAOT_training_forget_1 = np.load('Multi_agent_one_training_forget1.npy')
    OAMT_training_forget_2 = np.load('One_agent_multi_training_forget2.npy')
    MAOT_training_forget_2 = np.load('Multi_agent_one_training_forget2.npy')
    OAMT_training_forget_3 = np.load('One_agent_multi_training_forget3.npy')
    MAOT_training_forget_3 = np.load('Multi_agent_one_training_forget3.npy')
    OAMT_training_forget_4 = np.load('One_agent_multi_training_forget4.npy')
    MAOT_training_forget_4 = np.load('Multi_agent_one_training_forget4.npy')
    OAMT_training_forget_0 = np.delete(OAMT_training_forget_0,0)
    MAOT_training_forget_0 = np.delete(MAOT_training_forget_0,0)
    OAMT_training_forget_1 = np.delete(OAMT_training_forget_1,0)
    MAOT_training_forget_1 = np.delete(MAOT_training_forget_1,0)
    OAMT_training_forget_2 = np.delete(OAMT_training_forget_2,0)
    MAOT_training_forget_2 = np.delete(MAOT_training_forget_2,0)
    OAMT_training_forget_3 = np.delete(OAMT_training_forget_3,0)
    MAOT_training_forget_3 = np.delete(MAOT_training_forget_3,0)
    OAMT_training_forget_4 = np.delete(OAMT_training_forget_4,0)
    MAOT_training_forget_4 = np.delete(MAOT_training_forget_4,0)
    
    min_OAMT = np.minimum(OAMT_training_forget_0,OAMT_training_forget_1)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_forget_2)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_forget_3)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_forget_4)

    max_OAMT = np.maximum(OAMT_training_forget_0,OAMT_training_forget_1)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_forget_2)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_forget_3)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_forget_4)
    
    avg_OAMT = OAMT_training_forget_0 + OAMT_training_forget_1+OAMT_training_forget_2+OAMT_training_forget_3+OAMT_training_forget_4
    avg_OAMT = avg_OAMT / 5 
    
        
    OAMT_var = OAMT_training_forget_0 - avg_OAMT
    OAMT_var = np.square(OAMT_var)
    OAMT_var2 = OAMT_training_forget_1 - avg_OAMT
    OAMT_var2 = np.square(OAMT_var2)
    OAMT_var3 = OAMT_training_forget_2 - avg_OAMT
    OAMT_var3 = np.square(OAMT_var3)
    OAMT_var4 = OAMT_training_forget_3 - avg_OAMT
    OAMT_var4 = np.square(OAMT_var4)
    OAMT_var5 = OAMT_training_forget_4 - avg_OAMT
    OAMT_var5 = np.square(OAMT_var5)
    
    OAMT_std_dev = OAMT_var + OAMT_var2 + OAMT_var3 + OAMT_var4 + OAMT_var5
    OAMT_std_dev = OAMT_std_dev/5
    OAMT_std_dev = np.sqrt(OAMT_std_dev)

    min_MAOT = np.minimum(MAOT_training_forget_0,MAOT_training_forget_1)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_forget_2)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_forget_3)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_forget_4)

    max_MAOT = np.maximum(MAOT_training_forget_0,MAOT_training_forget_1)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_forget_2)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_forget_3)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_forget_4)       
    
    avg_MAOT = MAOT_training_forget_0 + MAOT_training_forget_1+MAOT_training_forget_2+MAOT_training_forget_3+MAOT_training_forget_4
    avg_MAOT = avg_MAOT / 5
    
    MAOT_var = MAOT_training_forget_0 - avg_MAOT
    MAOT_var = np.square(MAOT_var)
    MAOT_var2 = MAOT_training_forget_1 - avg_MAOT
    MAOT_var2 = np.square(MAOT_var2)
    MAOT_var3 = MAOT_training_forget_2 - avg_MAOT
    MAOT_var3 = np.square(MAOT_var3)
    MAOT_var4 = MAOT_training_forget_3 - avg_MAOT
    MAOT_var4 = np.square(MAOT_var4)
    MAOT_var5 = MAOT_training_forget_4 - avg_MAOT
    MAOT_var5 = np.square(MAOT_var5)
    
    MAOT_std_dev = MAOT_var + MAOT_var2 + MAOT_var3 + MAOT_var4 + MAOT_var5
    MAOT_std_dev = MAOT_std_dev/5
    MAOT_std_dev = np.sqrt(MAOT_std_dev)
    
    plt.plot(x,avg_MAOT*100, color='purple', label='eco-system accuracy')
    #plt.plot(max_MAOT, color='blue', label='eco-system accuracy')
    plt.fill_between(x,np.clip(avg_MAOT-MAOT_std_dev,0,1)*100, np.clip(avg_MAOT+MAOT_std_dev,0,1)*100,facecolor="purple", color='purple',alpha=0.2)  
    plt.plot(x,avg_OAMT*100, color='orange', label='one-agent accuracy')
    #plt.plot(max_OAMT, color='red', label='one-agent accuracy')
    plt.fill_between(x,np.clip(avg_OAMT-OAMT_std_dev,0,1)*100, np.clip(avg_OAMT+OAMT_std_dev,0,1)*100,facecolor="orange", color='orange',alpha=0.2)  
    red_patch = mpatches.Patch(color='orange', label='single-agent')
    blue_patch = mpatches.Patch(color='purple', label='eco-system')
    plt.legend(handles=[red_patch,blue_patch])
    plt.savefig('Architecture_comparison_forget_hard.pdf')
    plt.close()


    plt.title("Adaptability index on new environments (Submarine hard)", size=15,y=1.06)
    x = [1,2,3,4,5,6]
    plt.xticks([1,2,4,6],[50,100,200,300],rotation=0)
    plt.ylabel('% solved', size = 14)
    plt.xlabel('# of environment trained', size = 14)
    OAMT_training_general_0 = np.load('One_agent_multi_training_general0.npy')
    MAOT_training_general_0 = np.load('Multi_agent_one_training_general0.npy')
    OAMT_training_general_1 = np.load('One_agent_multi_training_general1.npy')
    MAOT_training_general_1 = np.load('Multi_agent_one_training_general1.npy')
    OAMT_training_general_2 = np.load('One_agent_multi_training_general2.npy')
    MAOT_training_general_2 = np.load('Multi_agent_one_training_general2.npy')
    OAMT_training_general_3 = np.load('One_agent_multi_training_general3.npy')
    MAOT_training_general_3 = np.load('Multi_agent_one_training_general3.npy')
    OAMT_training_general_4 = np.load('One_agent_multi_training_general4.npy')
    MAOT_training_general_4 = np.load('Multi_agent_one_training_general4.npy')
    OAMT_training_general_0 = np.delete(OAMT_training_general_0,0)
    MAOT_training_general_0 = np.delete(MAOT_training_general_0,0)
    OAMT_training_general_1 = np.delete(OAMT_training_general_1,0)
    MAOT_training_general_1 = np.delete(MAOT_training_general_1,0)
    OAMT_training_general_2 = np.delete(OAMT_training_general_2,0)
    MAOT_training_general_2 = np.delete(MAOT_training_general_2,0)
    OAMT_training_general_3 = np.delete(OAMT_training_general_3,0)
    MAOT_training_general_3 = np.delete(MAOT_training_general_3,0)
    OAMT_training_general_4 = np.delete(OAMT_training_general_4,0)
    MAOT_training_general_4 = np.delete(MAOT_training_general_4,0)
    
    min_OAMT = np.minimum(OAMT_training_general_0,OAMT_training_general_1)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_general_2)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_general_3)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_general_4)

    max_OAMT = np.maximum(OAMT_training_general_0,OAMT_training_general_1)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_general_2)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_general_3)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_general_4) 
    
    avg_OAMT = OAMT_training_general_0 + OAMT_training_general_1+OAMT_training_general_2+OAMT_training_general_3+OAMT_training_general_4
    avg_OAMT = avg_OAMT / 5 
        
    OAMT_var = OAMT_training_general_0 - avg_OAMT
    OAMT_var = np.square(OAMT_var)
    OAMT_var2 = OAMT_training_general_1 - avg_OAMT
    OAMT_var2 = np.square(OAMT_var2)
    OAMT_var3 = OAMT_training_general_2 - avg_OAMT
    OAMT_var3 = np.square(OAMT_var3)
    OAMT_var4 = OAMT_training_general_3 - avg_OAMT
    OAMT_var4 = np.square(OAMT_var4)
    OAMT_var5 = OAMT_training_general_4 - avg_OAMT
    OAMT_var5 = np.square(OAMT_var5)
    
    OAMT_std_dev = OAMT_var + OAMT_var2 + OAMT_var3 + OAMT_var4 + OAMT_var5
    OAMT_std_dev = OAMT_std_dev/5
    OAMT_std_dev = np.sqrt(OAMT_std_dev)

    min_MAOT = np.minimum(MAOT_training_general_0,MAOT_training_general_1)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_general_2)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_general_3)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_general_4)

    max_MAOT = np.maximum(MAOT_training_general_0,MAOT_training_general_1)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_general_2)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_general_3)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_general_4)   
    
    avg_MAOT = MAOT_training_general_0 + MAOT_training_general_1+MAOT_training_general_2+MAOT_training_general_3+MAOT_training_general_4
    avg_MAOT = avg_MAOT / 5
    
    MAOT_var = MAOT_training_general_0 - avg_MAOT
    MAOT_var = np.square(MAOT_var)
    MAOT_var2 = MAOT_training_general_1 - avg_MAOT
    MAOT_var2 = np.square(MAOT_var2)
    MAOT_var3 = MAOT_training_general_2 - avg_MAOT
    MAOT_var3 = np.square(MAOT_var3)
    MAOT_var4 = MAOT_training_general_3 - avg_MAOT
    MAOT_var4 = np.square(MAOT_var4)
    MAOT_var5 = MAOT_training_general_4 - avg_MAOT
    MAOT_var5 = np.square(MAOT_var5)
    
    MAOT_std_dev = MAOT_var + MAOT_var2 + MAOT_var3 + MAOT_var4 + MAOT_var5
    MAOT_std_dev = MAOT_std_dev/5
    MAOT_std_dev = np.sqrt(MAOT_std_dev)
         
    plt.plot(x,avg_MAOT, color='purple', label='eco-system accuracy')
    #plt.plot(max_MAOT, color='blue', label='eco-system accuracy')
    plt.fill_between(x,np.clip(avg_MAOT-MAOT_std_dev,0,100), np.clip(avg_MAOT+MAOT_std_dev,0,100),facecolor="purple", color='purple',alpha=0.2)  
    plt.plot(x,avg_OAMT, color='orange', label='one-agent accuracy')
    #plt.plot(max_OAMT, color='red', label='one-agent accuracy')
    plt.fill_between(x,np.clip(avg_OAMT-OAMT_std_dev,0,100), np.clip(avg_OAMT+OAMT_std_dev,0,100),facecolor="orange", color='orange',alpha=0.2)  
    red_patch = mpatches.Patch(color='orange', label='single-agent')
    blue_patch = mpatches.Patch(color='purple', label='eco-system')
    plt.legend(handles=[red_patch,blue_patch])
    plt.savefig('Architecture_comparison_general_hard.pdf')
    plt.close()
    
    plt.title("Number of accesses to environments (Submarine hard)", size=15,y=1.06)
    x = [1,2,3,4,5,6]
    plt.xticks([1,2,4,6],[50,100,200,300],rotation=0)
    plt.ylabel('# accesses', size = 14)
    plt.xlabel('# of environment trained', size = 14)
    OAMT_training_access_0 = np.load('One_agent_multi_training_access0.npy')
    MAOT_training_access_0 = np.load('Multi_agent_one_training_access0.npy')
    OAMT_training_access_1 = np.load('One_agent_multi_training_access1.npy')
    MAOT_training_access_1 = np.load('Multi_agent_one_training_access1.npy')
    OAMT_training_access_2 = np.load('One_agent_multi_training_access2.npy')
    MAOT_training_access_2 = np.load('Multi_agent_one_training_access2.npy')
    OAMT_training_access_3 = np.load('One_agent_multi_training_access3.npy')
    MAOT_training_access_3 = np.load('Multi_agent_one_training_access3.npy')
    OAMT_training_access_4 = np.load('One_agent_multi_training_access4.npy')
    MAOT_training_access_4 = np.load('Multi_agent_one_training_access4.npy')
    OAMT_training_access_0 = np.delete(OAMT_training_access_0,0)
    MAOT_training_access_0 = np.delete(MAOT_training_access_0,0)
    OAMT_training_access_1 = np.delete(OAMT_training_access_1,0)
    MAOT_training_access_1 = np.delete(MAOT_training_access_1,0)
    OAMT_training_access_2 = np.delete(OAMT_training_access_2,0)
    MAOT_training_access_2 = np.delete(MAOT_training_access_2,0)
    OAMT_training_access_3 = np.delete(OAMT_training_access_3,0)
    MAOT_training_access_3 = np.delete(MAOT_training_access_3,0)
    OAMT_training_access_4 = np.delete(OAMT_training_access_4,0)
    MAOT_training_access_4 = np.delete(MAOT_training_access_4,0)
    
    min_OAMT = np.minimum(OAMT_training_access_0,OAMT_training_access_1)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_access_2)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_access_3)
    min_OAMT = np.minimum(min_OAMT,OAMT_training_access_4)
    
    max_OAMT = np.maximum(OAMT_training_access_0,OAMT_training_access_1)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_access_2)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_access_3)
    max_OAMT = np.maximum(max_OAMT,OAMT_training_access_4) 
    
    avg_OAMT = OAMT_training_access_0 + OAMT_training_access_1+OAMT_training_access_2+OAMT_training_access_3+OAMT_training_access_4
    avg_OAMT = avg_OAMT / 5 
        
    OAMT_var = OAMT_training_access_0 - avg_OAMT
    OAMT_var = np.square(OAMT_var)
    OAMT_var2 = OAMT_training_access_1 - avg_OAMT
    OAMT_var2 = np.square(OAMT_var2)
    OAMT_var3 = OAMT_training_access_2 - avg_OAMT
    OAMT_var3 = np.square(OAMT_var3)
    OAMT_var4 = OAMT_training_access_3 - avg_OAMT
    OAMT_var4 = np.square(OAMT_var4)
    OAMT_var5 = OAMT_training_access_4 - avg_OAMT
    OAMT_var5 = np.square(OAMT_var5)
    
    OAMT_std_dev = OAMT_var + OAMT_var2 + OAMT_var3 + OAMT_var4 + OAMT_var5
    OAMT_std_dev = OAMT_std_dev/5
    OAMT_std_dev = np.sqrt(OAMT_std_dev)
    
    min_MAOT = np.minimum(MAOT_training_access_0,MAOT_training_access_1)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_access_2)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_access_3)
    min_MAOT = np.minimum(min_MAOT,MAOT_training_access_4)
    
    max_MAOT = np.maximum(MAOT_training_access_0,MAOT_training_access_1)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_access_2)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_access_3)
    max_MAOT = np.maximum(max_MAOT,MAOT_training_access_4)       
    
    avg_MAOT = MAOT_training_access_0 + MAOT_training_access_1+MAOT_training_access_2+MAOT_training_access_3+MAOT_training_access_4
    avg_MAOT = avg_MAOT / 5
    
    MAOT_var = MAOT_training_access_0 - avg_MAOT
    MAOT_var = np.square(MAOT_var)
    MAOT_var2 = MAOT_training_access_1 - avg_MAOT
    MAOT_var2 = np.square(MAOT_var2)
    MAOT_var3 = MAOT_training_access_2 - avg_MAOT
    MAOT_var3 = np.square(MAOT_var3)
    MAOT_var4 = MAOT_training_access_3 - avg_MAOT
    MAOT_var4 = np.square(MAOT_var4)
    MAOT_var5 = MAOT_training_access_4 - avg_MAOT
    MAOT_var5 = np.square(MAOT_var5)
    
    MAOT_std_dev = MAOT_var + MAOT_var2 + MAOT_var3 + MAOT_var4 + MAOT_var5
    MAOT_std_dev = MAOT_std_dev/5
    MAOT_std_dev = np.sqrt(MAOT_std_dev)
     
    plt.plot(x,avg_MAOT, color='purple', label='eco-system access')
    #plt.plot(max_MAOT, color='blue', label='eco-system access')
    plt.fill_between(x,avg_MAOT-MAOT_std_dev, avg_MAOT+MAOT_std_dev,facecolor="purple", color='purple',alpha=0.2)  
    plt.plot(x,avg_OAMT, color='orange', label='one-agent access')
    #plt.plot(max_OAMT, color='red', label='one-agent access')
    plt.fill_between(x,avg_OAMT-OAMT_std_dev, avg_OAMT+OAMT_std_dev,facecolor="orange", color='orange',alpha=0.2)  
    red_patch = mpatches.Patch(color='orange', label='single-agent')
    blue_patch = mpatches.Patch(color='purple', label='eco-system')
    plt.legend(handles=[red_patch,blue_patch])
    plt.savefig('Architecture_comparison_access_hard.pdf')
    plt.close()
    

def main():
    generate_results_comparison()

if __name__ == "__main__":
    main()
