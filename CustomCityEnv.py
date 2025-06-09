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

        # Calculate the expected observation size
        # For N buildings, the observation size is:
        # - N buildings * (2 metrics + 2 trends) = 4N
        # - 3 district metrics = 3
        # - 4 time features = 4
        # - 9 action history (3 actions * 3 one-hot) = 9
        # Total = 4N + 3 + 4 + 9 = 4N + 16
        self.expected_obs_size = 4 * num_buildings + 16
        
        # Define the observation space with the correct size
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.expected_obs_size,),
            dtype=np.float32
        )

        self.episode_data = None # To store data for the current episode

    def _get_observation(self):
        """Construct observation vector with enhanced state information.
        
        The observation includes:
        - Building-level metrics (consumption, temperature, setpoints, occupancy)
        - District-level metrics (aggregate consumption, emission rates)
        - Time features (hour of day, day of week)
        - Recent history statistics (moving averages, trends)
        - Action history (last few actions)
        """
        # Initialize observation vector with zeros
        obs = []
        
        # Ensure data is available
        if self.episode_data is None or self.current_timestep >= self.timesteps_per_episode:
            print("Warning: _get_observation called with no data or at end of episode.")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Track building-specific features
        for b in range(self.num_buildings):
            # Current timestep values
            current_e = self.episode_data['e'][self.current_timestep, b]
            current_temp = self.episode_data['temp'][self.current_timestep, b]
            
            # Add normalized features
            obs.append(current_e / 100.0)  # Normalized consumption (assuming max 100 kW)
            obs.append((current_temp - 15.0) / 10.0)  # Normalized temperature around 20°C
            
            # Add short-term trends (difference from previous timestep)
            if self.current_timestep > 0:
                prev_e = self.episode_data['e'][self.current_timestep-1, b]
                prev_temp = self.episode_data['temp'][self.current_timestep-1, b]
                obs.append((current_e - prev_e) / 10.0)  # Consumption trend
                obs.append((current_temp - prev_temp) / 5.0)  # Temperature trend
            else:
                obs.extend([0.0, 0.0])  # No trend data for first step
        
        # District-level metrics
        district_consumption = self.episode_data['district_consumption'][self.current_timestep]
        emission_rate = self.episode_data['emission_rate'][self.current_timestep]
        outage = self.episode_data['outage_timesteps'][self.current_timestep]
        
        obs.append(district_consumption / 1000.0)  # Normalized district consumption
        obs.append(emission_rate / 1000.0)  # Normalized emission rate
        obs.append(1.0 if outage else 0.0)  # Binary outage flag
        
        # Time features
        hour_of_day = self.current_timestep % 24
        day_of_week = (self.current_timestep // 24) % 7
        
        # Circular encoding for time features (sine/cosine)
        obs.append(np.sin(2 * np.pi * hour_of_day / 24.0))  # Hour of day (sine)
        obs.append(np.cos(2 * np.pi * hour_of_day / 24.0))  # Hour of day (cosine)
        obs.append(np.sin(2 * np.pi * day_of_week / 7.0))   # Day of week (sine)
        obs.append(np.cos(2 * np.pi * day_of_week / 7.0))   # Day of week (cosine)
        
        # Action history (last 3 actions, one-hot encoded)
        action_history = np.zeros(3 * 3)  # 3 actions, each one-hot encoded
        for i, action in enumerate(self.action_history[-3:], 1):  # Last 3 actions
            action_history[(i-1)*3 + action] = 1.0
        obs.extend(action_history)
        
        return np.array(obs, dtype=np.float32)

    def reset(self):
        self.current_timestep = 0
        # Initialize or clear action history
        self.action_history = []
        
        # Generate new episode data
        self.episode_data = generate_dummy_environment_data(
            self.timesteps_per_episode, 
            self.num_buildings, 
            action=None
        )
        
        # Generate new episode data
        self.episode_data = generate_dummy_environment_data(
            self.timesteps_per_episode, 
            self.num_buildings, 
            action=None
        )
        
        # Get initial observation
        obs = self._get_observation()
        
        # Verify observation shape matches the space
        if obs.shape[0] != self.expected_obs_size:
            raise ValueError(
                f"Observation shape {obs.shape} does not match expected shape "
                f"({self.expected_obs_size},). Please check _get_observation() implementation."
            )
        
        return obs

    def step(self, action):
        if self.episode_data is None:
            raise Exception("Must call reset() before step()")

        # Convert action to integer if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
            
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {self.action_space}")
            
        # Store the action in history before processing
        self.action_history.append(action)

        # Apply the action to influence the environment state
        # Action 0: Low consumption target (more aggressive energy saving)
        # Action 1: Medium consumption target (balanced approach)
        # Action 2: High consumption target (prioritize comfort)
        action_effect = {
            0: 0.8,  # Reduce consumption by 20%
            1: 1.0,  # No change
            2: 1.2   # Increase consumption by 20%
        }[action]
        
        # Apply action effect to the current timestep's consumption
        if 'e' in self.episode_data and self.current_timestep < self.timesteps_per_episode:
            # Apply action effect to electricity consumption for all buildings
            self.episode_data['e'][self.current_timestep] *= action_effect
            
            # Ensure consumption stays within reasonable bounds (10% to 200% of original)
            self.episode_data['e'][self.current_timestep] = np.clip(
                self.episode_data['e'][self.current_timestep],
                0.1,  # Minimum 10% of original consumption
                2.0    # Maximum 200% of original consumption
            )
            
            # Update district consumption
            if 'district_consumption' in self.episode_data:
                self.episode_data['district_consumption'][self.current_timestep] = np.sum(
                    self.episode_data['e'][self.current_timestep]
                )

        # Increment timestep and check if episode is done
        self.current_timestep += 1
        done = self.current_timestep >= self.timesteps_per_episode
        info = {}
        
        # Calculate intermediate reward based on current timestep
        reward = 0.0
        
        if done:
            try:
                # Calculate final KPIs for the episode
                raw_kpis = self.reward_calculator.get_all_kpi_values(self.episode_data)
                
                # Get individual KPI components
                emissions = raw_kpis.get('carbon_emissions', 1000)
                unmet_hours = raw_kpis.get('unmet_hours', 1.0)
                ramping = raw_kpis.get('ramping', 0)
                load_factor = raw_kpis.get('load_factor', 0)
                peaks = raw_kpis.get('peaks', 0)
                resilience = raw_kpis.get('resilience', 0)
                
                # Calculate individual scores (0-1 range, higher is better)
                comfort_score = 1.0 - min(1.0, unmet_hours)
                emissions_score = np.exp(-emissions / 500.0)  # Exponential scaling
                
                # Calculate base reward from all KPIs
                base_reward = float(self.reward_calculator.score(raw_kpis))
                
                # Enhanced reward shaping with better scaling and shaping
                # MODIFIED: reward_components = {
                # MODIFIED: 'comfort': 0.5 * comfort_score,
                # MODIFIED: 'emissions': 0.4 * emissions_score,
                # MODIFIED: 'ramping': 0.05 * (1.0 - min(1.0, ramping / 50.0)),
                # MODIFIED: 'load_factor': 0.05 * load_factor,
                # MODIFIED: 'peaks': 0.05 * (1.0 - min(1.0, peaks / 100.0))
                # MODIFIED: }
                
                # Calculate base reward
                reward = base_reward # MODIFIED: Use score from AICrowdControl
                
                # Add shaped rewards for better learning
                # 1. Bonus for maintaining comfort
                if comfort_score > 0.95:
                    reward += 0.05 # MODIFIED: Bonus reduced
                    
                # 2. Bonus for low emissions
                if emissions_score > 0.9:
                    reward += 0.05 # MODIFIED: Bonus reduced
                    
                # 3. Penalize extreme actions (encourage smoother control)
                if hasattr(self, 'last_action') and action is not None:
                    action_change = abs(action - self.last_action)
                    reward -= 0.1 * action_change  # Small penalty for large action changes
                
                # Scale reward to a reasonable range
                reward = 10.0 * reward
                
                # Clip to stable range
                reward = np.clip(reward, -5.0, 15.0)
                
                # Store detailed info for analysis
                info = {
                    'episode_kpis': raw_kpis,
                    'comfort_score': comfort_score,
                    'emissions_score': emissions_score,
                    'carbon_emissions': emissions,
                    'unmet_hours': unmet_hours,
                    'ramping': ramping,
                    'load_factor': load_factor,
                    'peaks': peaks,
                    'resilience': resilience,
                    'reward_components': {
                        'comfort': comfort_score,
                        'emissions': emissions_score,
                        'ramping': 1.0 - min(1.0, ramping / 50.0),
                        'load_factor': load_factor,
                        'peaks': 1.0 - min(1.0, peaks / 100.0)
                    }
                }
                
            except Exception as e:
                print(f"Error calculating KPIs/reward at end of episode: {e}")
                reward = -5.0  # Significant penalty for failure
                info = {'error': str(e)}
                
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            # Calculate a small intermediate reward based on current state
            # This helps guide the learning process
            try:
                # Get current timestep data
                current_data = {k: v[:self.current_timestep+1] for k, v in self.episode_data.items() 
                              if isinstance(v, (list, np.ndarray)) and len(v) > self.current_timestep}
                
                # Calculate intermediate KPIs for the current episode segment
                raw_kpis = self.reward_calculator.get_all_kpi_values(current_data)
                
                # Intermediate reward with better shaping
                comfort_score = 1.0 - min(1.0, raw_kpis.get('unmet_hours', 1.0))
                emissions = raw_kpis.get('carbon_emissions', 1000)
                emissions_score = np.exp(-emissions / 500.0)
                
                # Calculate base reward components
                reward_components = {
                    'comfort': 0.6 * comfort_score,
                    'emissions': 0.4 * emissions_score
                }
                
                # Calculate base reward
                reward = sum(reward_components.values())
                
                # Add exploration bonus (decaying with time)
                if not hasattr(self, 'last_action'):
                    self.last_action = action
                    self.exploration_bonus = 0.1
                else:
                    # Decaying exploration bonus
                    self.exploration_bonus = max(0.01, self.exploration_bonus * 0.999)
                    if action != self.last_action:
                        reward += self.exploration_bonus
                    self.last_action = action
                
                # Scale and clip intermediate reward
                reward = 2.0 * reward  # Scale to (0-2) range
                reward = np.clip(reward, -0.5, 3.0)  # Allow small negative rewards
                
            except Exception as e:
                # If something goes wrong with intermediate reward, continue with zero reward
                reward = 0.0
                
            observation = self._get_observation()  # Get next observation

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
