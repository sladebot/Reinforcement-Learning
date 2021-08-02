import gym


class Gym:
    def __init__(self, env_name, env_seed=1):
        self.environment = gym.make(env_name)
        self.environment.seed(env_seed)
        print(f"""Initialized environment: {env_name} \n
        Observations: {self.get_observations()} \n
        Actions: {self.get_actions()}\n
        """)

    def get_observations(self):
        n_observations = self.environment.observation_space
        return n_observations

    def get_actions(self):
        n_actions = self.environment.action_space.n
        return n_actions
