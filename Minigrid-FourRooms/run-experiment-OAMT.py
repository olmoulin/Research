import numpy as np
import copy
from datetime import datetime
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from PIL import Image


from stable_baselines3 import PPO

import time
from gym_minigrid.wrappers import *


class Agent():
	def print_environment(self):
		pe = self.env.unwrapped.grid.encode()
		pe[self.env.agent_pos[0],self.env.agent_pos[1]]=10
		pe = pe[:,:,0]
		for i in range(0,19):
			line =""
			for j in range(0,19):
				if pe[i,j]==1:
					line=line+" "
				if pe[i,j]==2:
					line = line + "#"
				if pe[i,j]==8:
					line=line+"X"
				if pe[i,j]==10:
					line=line+"@"
			print(line)
		

	def __init__ (self,env_nb):
		self.env = gym.make('MiniGrid-FourRooms-v0')
		self.env = ImgObsWrapper(self.env)
		self.env.seed(env_nb)	
		self.env.reset()	
		self.print_environment()
		self.model = PPO("MlpPolicy",self.env,verbose=0)
		
	def change_environment(self,nb_env):
		self.env.seed(nb_env)

	def check(self):
		state = self.env.reset()
		total_rew =0
		done = False
		while done == False:
			action, _states = self.model.predict(state,deterministic=True)
			state, reward, done, _ = self.env.step(action)
			total_rew+=reward
		return total_rew

	def solve(self):
		self.print_environment()
		solved = False
		while solved==False:
			self.model.learn(total_timesteps=100000)
			chk = self.check()
			print("Check for stopping training : ",chk)
			if chk>=0.80:
				solved = True
			sys.stdout.flush()
			
	def save(self,i,it):
		self.model.save('Agent_'+str(i)+'_'+str(it)+'.mdl')
	
	def load(self,i,it):
		self.model = PPO.load('Agent_'+str(i)+'_'+str(it)+'.mdl')
			

class OneAgentMultipleTrainings():
	def __init__ (self):
		self.single_agent = Agent(0) 

	def train(self,env_seed):
		print("Training on environment ",env_seed)
		start_time = time.time()
		self.single_agent.change_environment(env_seed)
		if self.single_agent.check()<0.8:
			self.single_agent.solve()
		elapsed_time = time.time() - start_time
		print("Time elapsed: ",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
		return elapsed_time

	def checkpoint(self,it):
			self.single_agent.save(0,it)
	
	def restore(self,it):
		self.single_agent = Agent(0)
		self.single_agent.load(0,it)		
	
	def test(self,nb_env):
		self.single_agent.change_environment(nb_env)
		return self.single_agent.check()

def generate_results_OAMT():
	OAMT=OneAgentMultipleTrainings()
	training_time=[]
	forget = []
	general=[]
	access=[]
	t_time=0
	for i in range(0,1001):
		t_time+=OAMT.train(np.random.randint(65000))
		if i % 50 ==0:
			OAMT.checkpoint(i)
			count_tested = 0.0
			count_ok = 0.0
			for j in range(0,1001):
				count_tested+=1.0
				if OAMT.test(np.random.randint(65000))>=0.8:
					count_ok+=1.0
			general.append(count_ok/count_tested)
			print("*******************************Generalizability test : ",count_ok/count_tested,"*****************")
	g_npy=np.array(general)
	np.save('One_agent_multiple_training_general.npy',g_npy)
	plt.title('% generalization on new environments')
	plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[0, 50, 100, 150, 200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000],rotation=90)
	plt.plot(general, color='black', label='accuracy')
	plt.legend()
	plt.savefig('One_agent_multiple_training_general.png')
	plt.close()


def main():
	random.seed(123456)
	np.random.seed(123456)
	
	generate_results_OAMT()

if __name__ == "__main__":
	main()
