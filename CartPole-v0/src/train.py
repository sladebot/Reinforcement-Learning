from collections import deque
import torch
from gym.wrappers.monitoring import video_recorder
import torch.optim as optim
import numpy as np
from util import get_device
from policy import Policy #, Memory
from env import Gym


class Trainer:
    def __init__(self, env_name, episodes=1000, gamma=1.0, lr=1e-2):
        self.device = get_device()
        gym = Gym(env_name, env_seed=0)
        self.env_name = env_name
        self.environment = gym.environment
        self.policy = Policy(state_space=4, n_actions=2)
        self.policy = self.policy.to(device=self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.episodes = episodes
        self.gamma = gamma

    def reinforce(self, max_t=1000, print_freq=100):
        self.policy.train()
        scores_queue = deque(maxlen=100)
        scores = []
        for e in range(1, self.episodes):
            saved_log_probs = []
            rewards = []
            state = self.environment.reset()
            for t in range(max_t):
                action, log_prob = self.policy.select_action(state)
                saved_log_probs.append(log_prob)
                state, reward, done, _ = self.environment.step(action)
                rewards.append(reward)
                if done:
                    break
            scores_queue.append(sum(rewards))
            scores.append(sum(rewards))

            discounts = [self.gamma ** i for i in range(len(rewards) + 1)]
            R = sum([a * b for a, b in zip(discounts, rewards)])

            policy_loss = []

            for log_prob in saved_log_probs:
                policy_loss.append(-log_prob * R)

            policy_loss = torch.cat(policy_loss).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            if e % print_freq == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_queue)))
                print(f'Mean: {np.mean(scores_queue)}')
            if np.mean(scores_queue) >= 180.0:
                print("Environment solved in {} episodes. \t Avg score: {:.2f}".format(e - 100, np.mean(scores_queue)))
                break
        return scores

    def export_video(self):
        gym = Gym(self.env_name)
        env = gym.environment
        vid = video_recorder.VideoRecorder(env, path="video/{}.mp4".format(self.env_name))
        state = env.reset()
        done = False
        self.policy.eval()
        for t in range(self.episodes):
            print("Running")
            vid.capture_frame()
            action, _ = self.policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break
        vid.close()
        env.close()