from __future__ import division
import numpy as np
# import gym
# import gym_gridworld
# /import random
from pylab import *

# env = gym.make('GridWorld-v0')

# STATE_DIM = env.n_states
# ACTION_DIM = env.n_actions

def new_plot(reward_epi, episodes):
	# plot
	plt.figure(figsize=(15, 10))
	plt.xlabel('Training Step', fontsize=20)
	# plt.xlim(-4, episodes+4)
	plt.ylabel('Cumulative Reward', fontsize=20)
	mean_reward = np.mean(reward_epi, axis=0)
	std_reward = np.std(reward_epi, axis=0)
	plt.errorbar(range(episodes), mean_reward, color='b', linewidth=1.5)
	plt.fill_between(range(episodes), (mean_reward - std_reward), (mean_reward + std_reward), color='b', alpha=0.3)
	# plt.savefig('Qlearn_cumulative_reward_curve.png')
	plt.title("Q-learning on Gridworld domain")
	plt.show()

	# fname = 'Gridworld_Qlearn_s_{}'.format(init_s)
	# save_results(fname, DISCOUNT, LEARNING_RATE, episodes, reward_epi)


def get_data(K, filename):

	# Create some test data
	dx = 1
	X = np.arange(0, K, 1)

	filename1 = filename + "1.txt"
	text_file1 = open(filename1, "r")
	rewards1 = text_file1.read().split(' ')

	filename2 = filename + "2.txt"
	text_file2 = open(filename2, "r")
	rewards2 = text_file2.read().split(' ')


	filename3 = filename + "3.txt"
	text_file3 = open(filename3, "r")
	rewards3 = text_file3.read().split(' ')

	filename4 = filename + "4.txt"
	text_file4 = open(filename4, "r")
	rewards4 = text_file4.read().split(' ')

	filename5 = filename + "5.txt"
	text_file5 = open(filename5, "r")
	rewards5 = text_file5.read().split(' ')

	rewards1 = map(float, rewards1)
	rewards2 = map(float, rewards2)
	rewards3 = map(float, rewards3)
	rewards4 = map(float, rewards4)
	rewards5 = map(float, rewards5)


	Y = np.zeros(K)
	for i in range(K):
		# if i % 25 == 0:
			Y[i] = (rewards1[i] + rewards2[i] + rewards3[i] + rewards4[i] +rewards5[i])/5
		# Y[i] = rewards1[i]

	return X,Y


def show_cdf():

	X1,Y1 = get_data(2000, "Results/GeneralTsallisV2/generaltsallisv2_2_85_copy")

	X2,Y2 = get_data(2000, "Results/GeneralTsallisV2/generaltsallisv2_1000_0003_copy")

	# X3,Y3 = get_data(2000, "Results/GeneralTsallisV2/generaltsallisv2_10000_00005_20_copy")

	X3,Y3 = get_data(2000, "Results/GeneralTsallisV2/generaltsallisv2_5000_00085_copy")

	# X3,Y3 = get_data(2000, "Results/pcl/pcl_copy")


	X4,Y4 = get_data(2000, "Results/Tsallis/tsallis_2_5_copy")

	X5,Y5 = get_data(2000, "generaltsallis_1000_0004_20_copy")

	X6,Y6 = get_data(2000, "Results/GeneralTsallis/generaltsallis_5000_00085_copy")



	# X3,Y3 = get_data(2000, "Results/GeneralTsallisV2/generaltsallisv2_1000_0003_DuplicatedInput")


	plot(X1, Y1, 'ro-', marker=None, markevery=1)
	plot(X2, Y2, 'bo-', marker=None, markevery=1)
	plot(X3, Y3, 'go-', marker=None, markevery=1)
	plot(X4, Y4, 'yo-', marker=None, markevery=1)
	plot(X5, Y5, 'ko-', marker=None, markevery=1)
	plot(X6, Y6, 'o-', marker=None, markevery=1, color="#aec7e8")




	title(r'generaltsallis vs generaltsallisv2 vs tsallis, Copy-V0 - Action space=20')
	xlabel('Training Steps')
	ylabel('Average Rewards')

	legend(('General Tsallis v2 q=2, k=0.85', 'General Tsallis v2 q=1000, k=0.0003',
			'General Tsallis v2 q=5000, k=0.00085', 'Tsallis q=2, k=0.5',
			'General Tsallis q=1000, k=0.0004', 'General Tsallis q=5000, k=0.00085'
			),
		   shadow=True, loc=(0.3, 0.05))

	show()

# show_cdf(10000)

show_cdf()
# show_cdf(2000, 'rewards2.txt')
# show_cdf(2000, 'rewards3.txt')
# show_cdf(2000, 'rewards4.txt')
# show_cdf(2000, 'rewards5.txt')


# text_file1 = open("total_reward4.txt", "r")
# rewards1 = text_file1.read().split(' ')
#
# # text_file3 = open("total_reward4.txt", "r")
# # rewards3 = text_file3.read().split(' ')
#
# rewards1 = map(float, rewards1)
# # rewards3 = map(float, rewards3)
#
# new_plot(rewards1, 2000)

