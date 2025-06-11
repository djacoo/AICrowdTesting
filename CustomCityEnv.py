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
        self.current_step = 0  # Initialize current_step
        self.episode_length = timesteps_per_episode  # Initialize episode_length

        self.reward_calculator = ControlTrackReward(baseline=BASELINE_KPIS, phase=phase_weights)

        # Enhanced action space with 5 discrete actions for more granular control
        # 0: Very low consumption (aggressive energy saving)
        # 1: Low consumption (energy saving)
        # 2: Medium consumption (balanced)
        # 3: High consumption (comfort focus)
        # 4: Very high consumption (maximum comfort)
        self.action_space = spaces.Discrete(5)

        # Calculate the expected observation size
        # For N buildings, the observation size is:
        # - N buildings * (2 metrics + 2 trends) = 4N
        # - 3 district metrics = 3
        # - 4 time features = 4
        # - 15 action history (3 actions * 5 one-hot) = 15
        # Total = 4N + 3 + 4 + 15 = 4N + 22
        self.expected_obs_size = 4 * num_buildings + 22
        
        # Define the observation space with the correct size
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.expected_obs_size,),
            dtype=np.float32
        )

        self.episode_data = None # To store data for the current episode
        self.original_e_episode_array = None # To store original e data for consistent action effects

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
            # Max 'e' is ABSOLUTE_MAX_E_PER_BUILDING (e.g. 100.0) due to clipping in step().
            # Scaling by ABSOLUTE_MAX_E_PER_BUILDING.
            # Define ABSOLUTE_MAX_E_PER_BUILDING, e.g. 100.0, consistent with step() method's clipping
            ABSOLUTE_MAX_E_PER_BUILDING = 100.0
            norm_current_e = current_e / ABSOLUTE_MAX_E_PER_BUILDING
            obs.append(norm_current_e)
            
            # Temp is [15, 25]. (temp - 15) / 10 gives [0, 1].
            norm_current_temp = (current_temp - 15.0) / 10.0
            obs.append(norm_current_temp)

            # Add short-term trends (difference from previous normalized timestep)
            if self.current_timestep > 0:
                prev_e = self.episode_data['e'][self.current_timestep-1, b]
                prev_temp = self.episode_data['temp'][self.current_timestep-1, b]

                norm_prev_e = prev_e / ABSOLUTE_MAX_E_PER_BUILDING
                obs.append(norm_current_e - norm_prev_e)  # Consumption trend will be roughly [-1, 1]

                norm_prev_temp = (prev_temp - 15.0) / 10.0
                obs.append(norm_current_temp - norm_prev_temp)  # Temperature trend will be roughly [-1, 1]
            else:
                obs.extend([0.0, 0.0])  # No trend data for first step
        
        # District-level metrics
        district_consumption = self.episode_data['district_consumption'][self.current_timestep]
        emission_rate = self.episode_data['emission_rate'][self.current_timestep]
        outage = self.episode_data['outage_timesteps'][self.current_timestep]
        
        # District consumption: sum of individual 'e' values. Max num_buildings * ABSOLUTE_MAX_E_PER_BUILDING
        # Define ABSOLUTE_MAX_E_PER_BUILDING, e.g. 100.0, consistent with step() method's clipping
        ABSOLUTE_MAX_E_PER_BUILDING = 100.0
        max_district_consumption = self.num_buildings * ABSOLUTE_MAX_E_PER_BUILDING
        obs.append(district_consumption / max_district_consumption if max_district_consumption > 0 else 0.0)

        # Emission rate is [0.1, 0.6]. Dividing by 0.6 gives [0.166, 1].
        obs.append(emission_rate / 0.6)
        obs.append(1.0 if outage else 0.0)  # Binary outage flag
        
        # Time features
        hour_of_day = self.current_timestep % 24
        day_of_week = (self.current_timestep // 24) % 7
        
        # Circular encoding for time features (sine/cosine)
        obs.append(np.sin(2 * np.pi * hour_of_day / 24.0))  # Hour of day (sine)
        obs.append(np.cos(2 * np.pi * hour_of_day / 24.0))  # Hour of day (cosine)
        obs.append(np.sin(2 * np.pi * day_of_week / 7.0))   # Day of week (sine)
        obs.append(np.cos(2 * np.pi * day_of_week / 7.0))   # Day of week (cosine)
        
        # Action history (last 3 actions, one-hot encoded for 5 possible actions)
        action_history = np.zeros(3 * 5)  # 3 actions, each one-hot encoded for 5 possible actions
        for i, action in enumerate(self.action_history[-3:]):  # Last 3 actions
            if i < 3:  # Ensure we don't go out of bounds
                action_history[i*5 + min(action, 4)] = 1.0  # Cap action at 4 (0-4 range)
        obs.extend(action_history)
        
        return np.array(obs, dtype=np.float32)

    def reset(self):
        self.current_timestep = 0
        self.current_step = 0  # Reset current_step
        # Initialize or clear action history
        self.action_history = []
        
        # Generate new episode data
        self.episode_data = generate_dummy_environment_data(
            self.timesteps_per_episode, 
            self.num_buildings, 
            action=None
        )
        # Store a copy of the original electricity consumption data
        self.original_e_episode_array = self.episode_data['e'].copy()
        
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

        # Apply the action to influence the environment state with stronger effects
        # Action 0: Low consumption target (aggressive energy saving)
        # Action 1: Medium consumption target (balanced approach)
        # Enhanced action effects with 5 discrete levels
        action_effect = {
            0: 0.4,   # Very low (60% reduction)
            1: 0.7,   # Low (30% reduction)
            2: 1.0,   # Medium (no change)
            3: 1.4,   # High (40% increase)
            4: 1.8    # Very high (80% increase)
        }[action]
        
        # Define absolute min/max for consumption per building
        ABSOLUTE_MIN_E_PER_BUILDING = 1.0
        ABSOLUTE_MAX_E_PER_BUILDING = 100.0

        # Apply action effect to the current timestep's consumption based on original data
        if self.original_e_episode_array is not None and \
           'e' in self.episode_data and \
           self.current_timestep < self.timesteps_per_episode:
            
            # Get base consumption from the original data for the current timestep
            base_consumption_t = self.original_e_episode_array[self.current_timestep].copy() # Ensure it's an array for per-building

            # Apply action effect
            modified_consumption_t = base_consumption_t * action_effect
            
            # Clip the modified consumption to absolute limits
            # Ensure clipping is done per building if modified_consumption_t is an array
            if isinstance(modified_consumption_t, np.ndarray):
                self.episode_data['e'][self.current_timestep] = np.clip(
                    modified_consumption_t,
                    ABSOLUTE_MIN_E_PER_BUILDING,
                    ABSOLUTE_MAX_E_PER_BUILDING
                )
            else: # Should be an array if num_buildings > 1, but as a fallback for num_buildings = 1
                 self.episode_data['e'][self.current_timestep, 0] = np.clip(
                    modified_consumption_t, # This assumes modified_consumption_t is scalar if not ndarray
                    ABSOLUTE_MIN_E_PER_BUILDING,
                    ABSOLUTE_MAX_E_PER_BUILDING
                )

            # Update district consumption based on the newly modified 'e' values
            if 'district_consumption' in self.episode_data:
                self.episode_data['district_consumption'][self.current_timestep] = np.sum(
                    self.episode_data['e'][self.current_timestep]
                )

        # Increment timestep and step counter
        self.current_timestep += 1
        self.current_step += 1  # Increment current_step
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
                # Other KPIs can be kept for info if needed
                ramping = raw_kpis.get('ramping', 0)
                load_factor = raw_kpis.get('load_factor', 0)
                peaks = raw_kpis.get('peaks', 0)
                resilience = raw_kpis.get('resilience', 0)

                # 1. Calculate base scores
                # Comfort score: 1 is best (no unmet hours), 0 is worst (1 or more unmet hours ratio)
                comfort_score = 1.0 - min(1.0, unmet_hours)
                
                # Emissions score: Scaled to be higher for lower emissions.
                # Using a simple inverse relationship, normalized. Max emissions could be e.g. 2000 to make score 0.
                # Let's aim for a score between 0 and 1.
                # If baseline emissions are X, and we achieve Y, score is roughly X/Y.
                # Let's use a simpler normalization: e.g. baseline_emissions / (baseline_emissions + current_emissions)
                # Or, more directly, 1 / (1 + normalized_emissions).
                # Assuming emissions around a few hundreds to a thousand.
                # Let's try: 1 - (emissions / (BASELINE_KPIS.get('carbon_emissions', 1000) * 2))
                # This makes emissions_score = 0.5 if current emissions = baseline, 1 if emissions = 0, 0 if emissions = 2*baseline
                baseline_emissions_kpi = BASELINE_KPIS.get('carbon_emissions', 1000) # Default if not in BASELINE_KPIS
                emissions_score = np.clip(1.0 - (emissions / (baseline_emissions_kpi * 2.0)), 0.0, 1.0)

                # 2. Calculate base_reward (focused on comfort and emissions)
                # Simple weighted average, scaled.
                # Weights can be adjusted, e.g., 0.7 for comfort, 0.3 for emissions.
                base_reward = 100.0 * (0.7 * comfort_score + 0.3 * emissions_score)

                # 3. Define comfort_bonus and emissions_bonus (simpler calculation)
                # Bonus if scores exceed a certain threshold, e.g., 0.8
                comfort_bonus = 0.0
                if comfort_score > 0.8:
                    comfort_bonus = 20.0 * (comfort_score - 0.8) # Linear bonus

                emissions_bonus = 0.0
                if emissions_score > 0.8:
                    emissions_bonus = 20.0 * (emissions_score - 0.8) # Linear bonus
                
                # 4. Action penalty (optional, can be kept simple)
                action_penalty = 0.0
                if hasattr(self, 'last_action') and action is not None and hasattr(self, 'action_history') and len(self.action_history) > 1:
                    # Penalize frequent changes if last_action is defined
                    action_change = abs(action - self.action_history[-2]) / (self.action_space.n -1) # Normalize by action space size
                    action_penalty = 5.0 * action_change # Small penalty for action changes
                
                # 5. Combine components
                # Removed progress_bonus, terminal_bonus, progress_factor, temperature, reward_scale
                reward = base_reward + comfort_bonus + emissions_bonus - action_penalty
                
                # 6. Clip to a revised range if necessary.
                # Given the new calculation, max possible (approx): 100 (base) + 20*0.2 (comfort) + 20*0.2 (emissions) = 108
                # Min possible (approx): 0 (base, bonuses) - 5 (penalty) = -5
                # Let's set a range like -10 to 110.
                reward = np.clip(reward, -10.0, 110.0)
                
                # Store detailed info for analysis
                info = {
                    'episode_kpis': raw_kpis,
                    'comfort_score': comfort_score,
                    'emissions_score': emissions_score,
                    'carbon_emissions': emissions,
                    'unmet_hours': unmet_hours,
                    'base_reward_calculated': base_reward,
                    'comfort_bonus_calculated': comfort_bonus,
                    'emissions_bonus_calculated': emissions_bonus,
                    'action_penalty_calculated': action_penalty,
                    'final_reward_before_clip': base_reward + comfort_bonus + emissions_bonus - action_penalty,
                    # Keep other KPIs for logging if needed
                    'ramping': ramping,
                    'load_factor': load_factor,
                    'peaks': peaks,
                    'resilience': resilience
                }
                
            except Exception as e:
                print(f"Error calculating KPIs/reward at end of episode: {e}")
                reward = -10.0  # Penalty for failure, consistent with new clip range
                info = {'error': str(e)}
                
            # For the terminal state, the observation is typically not used by SB3,
            # but providing a zero vector is a common practice.
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            # Intermediate reward: simple, based on current comfort and emissions scores
            reward = 0.0 # Default to 0 for intermediate steps
            try:
                # Use only the current timestep's data for a less complex intermediate signal
                # This requires properties of episode_data to be sliceable up to current_timestep
                
                # For a very simple intermediate reward, we might not even calculate full KPIs
                # but use proxies from the current observation or very recent data.
                # However, to keep it somewhat aligned with final goals:
                
                # Let's get current 'unmet_hours' and 'carbon_emissions' if possible,
                # or proxies. The observation itself contains normalized values that could be used.
                # For simplicity, let's assume we can get a rough idea of current comfort/emissions.
                
                # Simplified: Use a proxy for comfort and emissions from current step data if available
                # This is a placeholder; a robust implementation might need more careful KPI calculation
                # or rely on values that are updated incrementally.
                
                # For now, let's provide a small, consistent reward based on the *change*
                # in comfort and emissions if that data were readily available per step.
                # Since it's not easily available without re-calculating KPIs on partial data,
                # let's keep intermediate rewards simple: a small positive value for not being done,
                # or a small penalty for undesirable states if detectable.

                # For this simplification, we will make intermediate rewards zero.
                # The learning will be driven by the terminal reward.
                # This is a common approach in episodic tasks.
                reward = 0.0

            except Exception as e:
                # If something goes wrong with intermediate reward, default to 0
                reward = 0.0
                # print(f"Warning: Error calculating intermediate reward: {e}") # Optional: log this
                
            observation = self._get_observation()  # Get next observation

        # Update last_action after processing the current action
        self.last_action = action

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
