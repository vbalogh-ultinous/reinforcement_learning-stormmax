import gym
from gym import wrappers
import qlearning
import numpy
import matplotlib.pyplot as plt
import my_env


NUM_EPISODES = 20000
N_BINS = [8, 8, 8, 8]
MAX_STEPS =my_env.N_EDGES + 1
FAIL_PENALTY = -100
WIN_BONUS = 100
EPSILON = 0.6
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.9

RECORD = True

MIN_VALUES = [-0.5, -2.0, -0.5, -3.0]
MAX_VALUES = [0.5, 2.0, 0.5, 3.0]



def train(agent, env, history, num_episodes=NUM_EPISODES):
  for i in xrange(NUM_EPISODES):
    if i % 100:
      print "Episode {}".format(i + 1)
    obs = env.reset()
    cur_state = obs
    
    for t in xrange(MAX_STEPS):
      action = agent.get_action(cur_state)
      observation, reward, done, info = env.step(action)
      next_state = observation
      if done:
        if info['win']:
          print('WIN')
          reward = WIN_BONUS
        elif info['gameover']:
          print('LOSE')
          reward = FAIL_PENALTY
        agent.learn(cur_state, action, next_state, reward, done)
        print("Episode finished after {} timesteps".format(t + 1))
        print('#################################\n')
        history.append(t + 1)
        break
      agent.learn(cur_state, action, next_state, reward, done)
      cur_state = next_state
      if t == MAX_STEPS - 1:
        history.append(t + 1)
        print("Episode finished after {} timesteps".format(t + 1))
  return agent, history


env = my_env.MyEnv() #gym.make('CartPole-v0') # TODO myenv
# if RECORD:
#   env = wrappers.Monitor(env, '/home/vbalogh/git/reinforcement_learning-stormmax/my-experiment-1', force=True)

def get_actions(current_state):
  all_actions = list(range(my_env.N_EDGES))
  if 0 in current_state:
    current_pos = current_state.index(0)
    del all_actions[current_pos]

  return all_actions


agent = qlearning.QLearningAgent(get_actions,
                                     epsilon=EPSILON,
                                     alpha=LEARNING_RATE,
                                     gamma=DISCOUNT_FACTOR,
                                     epsilon_decay=EPSILON_DECAY)

history = []

agent, history = train(agent, env, history)

# if RECORD:
#   env.monitor.close()

avg_reward = [numpy.mean(history[i*100:(i+1)*100]) for i in xrange(int(len(history)/100))]
f_reward = plt.figure(1)
plt.plot(numpy.linspace(0, len(history), len(avg_reward)), avg_reward)
plt.ylabel('Rewards')
f_reward.show()
print 'press enter to continue'
# raw_input()
plt.close()


# Display:
print 'press ctrl-c to stop'
while True:
  obs = env.reset()
  cur_state = obs
  done = False

  t = 0
  while not done:
    env.render()
    t = t+1
    action = agent.get_action(cur_state)
    print('curr:', cur_state)
    print('action: ', action)
    observation, reward, done, info = env.step(action)
    next_state = observation
    print('next:', next_state)
    if done:
      if info['win']:
        print('WIN')
        reward = WIN_BONUS
      elif info['gameover']:
        print('LOSE')
        reward = FAIL_PENALTY
      agent.learn(cur_state, action, next_state, reward, done)
      print("Episode finished after {} timesteps".format(t+1))
      print('###################\n')
      history.append(t+1)
      break
    agent.learn(cur_state, action, next_state, reward, done)
    cur_state = next_state
