"""
Custom Environment for CityLearn-like KPI testing with stable-baselines3.

This module defines `CustomCityEnv`, a Gymnasium-compatible environment that
simulates building energy control scenarios. It uses a dummy data generator
and the AICrowdControl module for KPI calculation and reward scoring.
The environment is designed for episodic rewards, where KPIs and scores are
calculated at the end of each episode.
"""
import gym
from gym import spaces
import numpy as np
import pandas as pd # Required for generate_dummy_environment_data
from AICrowdControl import ControlTrackReward, PhaseWeights, BASELINE_KPIS # Assuming AICrowdControl.py is in the same directory or accessible in PYTHONPATH

# Copied from train_agent.py - consider refactoring this into a common utility module later
def generate_dummy_environment_data(num_timesteps, num_buildings, action=None):
    """
    Generates a dictionary of dummy data for all KPIs for one episode.
    'action': A conceptual parameter. In this version, if provided, it applies a simple
    modification to the baseline electricity consumption array ('e_episode_array') for the episode.
    """
    # Electricity consumption for each building, each timestep
    e_episode_array = np.random.rand(num_timesteps, num_buildings) * 50

    # Apply a conceptual action effect: if action is, e.g., 0 (low), 1 (medium), 2 (high)
    # This is a very simplistic way to make actions have some effect.
    if action is not None:
        if action == 0: # "low consumption" target
            e_episode_array *= 0.8
        elif action == 2: # "high consumption" target (less efficient)
            e_episode_array *= 1.2
        # action == 1 (medium) would be the default random values

    emission_rate_episode = np.random.rand(num_timesteps) * 0.5 + 0.1
    temp_episode_array = np.random.rand(num_timesteps, num_buildings) * 10 + 15
    setpoint_episode_array = np.full_like(temp_episode_array, 22.5)
    occupancy_episode_array = np.random.randint(0, 2, size=temp_episode_array.shape)
    district_consumption_episode = np.sum(e_episode_array, axis=1)
    outage_timesteps_episode = np.random.choice([True, False], size=num_timesteps, p=[0.1, 0.9])
    demand_episode_array = e_episode_array * np.random.uniform(1.0, 1.2, size=e_episode_array.shape)
    served_episode_array = demand_episode_array * np.random.uniform(0.7, 1.0, size=demand_episode_array.shape)
    served_episode_array = np.minimum(demand_episode_array, served_episode_array)

    # This function in the plan was intended to generate data for the whole episode.
    # For a gym step, we need data for ONE timestep.
    # So, this function will be used by reset() to generate data for the whole episode,
    # and step() will iterate through it.

    return {
        'e': e_episode_array,
        'emission_rate': emission_rate_episode,
        'temp': temp_episode_array,
        'setpoint': setpoint_episode_array,
        'band': 1.0,
        'occupancy': occupancy_episode_array,
        'district_consumption': district_consumption_episode,
        'hours_per_day': 24,
        'outage_timesteps': outage_timesteps_episode,
        'demand': demand_episode_array,
        'served': served_episode_array
    }

class CustomCityEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, phase_weights: PhaseWeights, num_buildings=2, timesteps_per_episode=24*7):
        """
        Initializes the CustomCityEnv.

        Args:
            phase_weights (PhaseWeights): Weights for different KPI components in reward calculation.
            num_buildings (int): Number of buildings to simulate.
            timesteps_per_episode (int): Number of timesteps that constitute one episode.
        """
        super(CustomCityEnv, self).__init__()

        self.num_buildings = num_buildings
        self.timesteps_per_episode = timesteps_per_episode
        self.current_timestep = 0

        self.reward_calculator = ControlTrackReward(baseline=BASELINE_KPIS, phase=phase_weights)

        # Action: 0 (low consumption target), 1 (medium), 2 (high)
        self.action_space = spaces.Discrete(3)

        # Observation space:
        # For simplicity, let's define a fixed-size observation.
        # This needs to be carefully designed based on what the agent should see.
        # Example: [building1_e, building2_e, emission_rate_t, building1_temp, building2_temp, district_consumption_t, outage_t]
        # This is a simplified observation for num_buildings=2.
        # A more general approach would be to flatten selected parts of environment_data.
        # Let's assume a fixed number of buildings for now for simplicity of observation space definition.
        # Observation: [e_b1, e_b2, temp_b1, temp_b2, district_consumption_t, emission_rate_t, outage_t]
        # All are floats. outage_t will be 1.0 if True, 0.0 if False.
        obs_size = self.num_buildings * 2 + 3 # (e for each building, temp for each building) + district_consumption + emission_rate + outage
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.episode_data = None # To store data for the current episode

    def _get_observation(self):
        # Construct observation for the current timestep
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Note: The observation vector structure is fixed based on current components:
        # [e_building1, ..., e_buildingN, temp_building1, ..., temp_buildingN, district_consumption, emission_rate, outage_flag]
        # If more observation components are added, this logic and `obs_size` in __init__ need updates.

        # Ensure data is available
        if self.episode_data is None or self.current_timestep >= self.timesteps_per_episode:
            # This should not happen if reset() is called correctly and current_timestep is managed.
            # Return a zero observation or handle error appropriately.
            print("Warning: _get_observation called with no data or at end of episode.")
            return obs

        # Example: [e_b1, e_b2, temp_b1, temp_b2, district_consumption_t, emission_rate_t, outage_t]
        obs_idx = 0
        for i in range(self.num_buildings):
            obs[obs_idx] = self.episode_data['e'][self.current_timestep, i]
            obs_idx += 1
        for i in range(self.num_buildings):
            obs[obs_idx] = self.episode_data['temp'][self.current_timestep, i]
            obs_idx += 1

        obs[obs_idx] = self.episode_data['district_consumption'][self.current_timestep]
        obs_idx += 1
        obs[obs_idx] = self.episode_data['emission_rate'][self.current_timestep]
        obs_idx += 1
        obs[obs_idx] = 1.0 if self.episode_data['outage_timesteps'][self.current_timestep] else 0.0

        return obs

    def reset(self):
        self.current_timestep = 0
        # The action passed to generate_dummy_environment_data here is None,
        # as it's the start of the episode before any agent action.
        self.episode_data = generate_dummy_environment_data(self.timesteps_per_episode, self.num_buildings, action=None)
        return self._get_observation()  # Original gym reset returns just the observation

    def step(self, action):
        if self.episode_data is None:
            raise Exception("Must call reset() before step()")

        # action parameter is part of the gym API. It's not used in this specific
        # step logic as episode data is pre-generated at reset.
        # The learning algorithm uses it to associate states with chosen actions.

        self.current_timestep += 1
        done = False
        reward = 0.0  # Default reward for intermediate steps
        info = {}

        if self.current_timestep >= self.timesteps_per_episode:
            done = True
            try:
                raw_kpis = self.reward_calculator.get_all_kpi_values(self.episode_data)
                reward = float(self.reward_calculator.score(raw_kpis)) # Ensure reward is float
                info = {'episode_kpis': raw_kpis}
            except Exception as e:
                print(f"Error calculating KPIs/reward at end of episode: {e}")
                reward = -1.0 # Penalize if KPI calculation fails, ensure float
                info = {'error': str(e)}

            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._get_observation() # Fetches observation for s_{t+1}

        return observation, reward, done, info

    def render(self, mode='human'):
        # For now, no complex rendering
        pass

    def close(self):
        pass

if __name__ == '__main__':
    # Example Usage and Test
    print("Testing CustomCityEnv...")
    phase_weights = PhaseWeights(w1=0.5, w2=0.5, w3=0.0, w4=0.0) # Comfort and Emissions
    env = CustomCityEnv(phase_weights=phase_weights, num_buildings=2, timesteps_per_episode=24*1) # 1 day episode
    env.seed(42) # Set seed for reproducibility
    obs = env.reset()
    print("Initial Observation Shape:", obs.shape)
    print("Initial Observation:", obs)
    # print("Initial Info:", info) # Info is not returned by reset in gym 0.21.0

    done = False
    total_reward = 0
    step_count = 0
    while not done:
        action = env.action_space.sample() # Random action
        obs, reward, done, info = env.step(action) # Use 'done'
        total_reward += reward # Accumulate reward (though it's only non-zero at the end)
        step_count += 1
        if done: # Check 'done'
            print(f"Episode finished after {step_count} steps. Done: {done}")
            print(f"Final Reward: {reward}") # This is the episode reward
            if 'episode_kpis' in info:
                print("Episode KPIs:")
                for k, v in info['episode_kpis'].items():
                    print(f"  {k}: {v:.4f}")
        # else:
            # print(f"Step {step_count}, Action: {action}, Obs: {obs[:3]}..., Reward: {reward}, Done: {done}")


    print("Environment test complete.")

    # Test with stable-baselines3 checker
    from stable_baselines3.common.env_checker import check_env
    try:
        check_env(env, warn=True)
        print("stable-baselines3 check_env passed!")
    except Exception as e:
        print(f"stable-baselines3 check_env failed: {e}")
