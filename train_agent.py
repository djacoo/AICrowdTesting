import numpy as np
import pandas as pd
from AICrowdControl import ControlTrackReward, PhaseWeights, BASELINE_KPIS, PHASE_I, PHASE_II

def generate_dummy_environment_data(num_timesteps, num_buildings):
    """
    Generates a dictionary of dummy data for all KPIs for one episode.
    """
    # Electricity consumption for each building, each timestep
    # Shape: (timesteps, num_buildings)
    e_episode_array = np.random.rand(num_timesteps, num_buildings) * 50 # kW for example

    # Emission rate per timestep
    # Shape: (timesteps,)
    emission_rate_episode = np.random.rand(num_timesteps) * 0.5 + 0.1 # kgCO2/kWh

    # Temperature, setpoint, occupancy for each building (zone), each timestep
    # Shape: (timesteps, num_buildings) assuming 1 zone per building for simplicity
    temp_episode_array = np.random.rand(num_timesteps, num_buildings) * 10 + 15 # deg C
    setpoint_episode_array = np.full_like(temp_episode_array, 22.5) # deg C
    occupancy_episode_array = np.random.randint(0, 2, size=temp_episode_array.shape)

    # District consumption: sum of consumption of all buildings per timestep
    # Shape: (timesteps,)
    district_consumption_episode = np.sum(e_episode_array, axis=1)

    # Outage: boolean array per timestep (applies to whole district for resilience KPIs)
    # Shape: (timesteps,)
    outage_timesteps_episode = np.random.choice([True, False], size=num_timesteps, p=[0.1, 0.9]) # 10% chance of outage

    # Demand and Served energy for each building, each timestep
    # Shape: (timesteps, num_buildings)
    # Demand can be slightly higher than consumption, served can be less than demand
    demand_episode_array = e_episode_array * np.random.uniform(1.0, 1.2, size=e_episode_array.shape)
    # During outage, served might be much lower or zero. For simplicity, this doesn't explicitly model that yet.
    # The KPI function for unserved_energy itself will check the outage array.
    served_episode_array = demand_episode_array * np.random.uniform(0.7, 1.0, size=demand_episode_array.shape)
    # Ensure served does not exceed demand due to multiplication
    served_episode_array = np.minimum(demand_episode_array, served_episode_array)


    current_environment_data = {
        'e': e_episode_array,
        'emission_rate': emission_rate_episode,
        'temp': temp_episode_array,
        'setpoint': setpoint_episode_array,
        'band': 1.0,  # deg C
        'occupancy': occupancy_episode_array,
        'district_consumption': district_consumption_episode,
        'hours_per_day': 24, # Assuming hourly timesteps
        'outage_timesteps': outage_timesteps_episode, # Used by thermal_resilience & unserved_energy
        'demand': demand_episode_array,
        'served': served_episode_array
    }
    return current_environment_data

def main():
    print("Starting training simulation...")
    print("Running test with Comfort and Emissions KPIs Combined (w1=0.5, w2=0.5).")

    # Define comfort and emissions combined weights
    comfort_emissions_weights = PhaseWeights(w1=0.5, w2=0.5, w3=0.0, w4=0.0)

    # Instantiate ControlTrackReward and set to comfort_emissions_weights phase
    reward_calculator = ControlTrackReward(baseline=BASELINE_KPIS, phase=comfort_emissions_weights)

    num_episodes = 10 # Small number for testing
    num_timesteps_per_episode = 24 * 7 # One week of hourly data
    num_buildings = 2 # Example number of buildings

    episode_logs = []

    # --- Training Loop (Placeholder) ---
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        # In a real scenario, data would be collected from an environment interaction loop
        # For now, we generate dummy data for the whole episode at once.
        current_environment_data = generate_dummy_environment_data(num_timesteps_per_episode, num_buildings)

        # Call get_all_kpi_values
        try:
            raw_kpis = reward_calculator.get_all_kpi_values(current_environment_data)
            print(f"Episode {episode + 1} Raw KPIs:")
            for k, v in raw_kpis.items():
                print(f"  {k}: {v:.4f}")

            log_entry = {'episode': episode + 1}
            log_entry.update(raw_kpis)
            episode_logs.append(log_entry)

        except Exception as e:
            print(f"Error calculating raw KPIs for episode {episode + 1}: {e}")
            # Potentially skip scoring if KPIs can't be calculated
            continue

        # Scoring with Comfort and Emissions KPIs Combined
        # The reward_calculator.phase is already set to comfort_emissions_weights
        try:
            # active_grid_kpis=None and active_resilience_kpis=None are default.
            # Since w3 and w4 are 0.0, these specific active lists don't alter the score here.
            comfort_emissions_score = reward_calculator.score(raw_kpis)
            print(f"Episode {episode + 1} Reward (Comfort & Emissions w1=0.5, w2=0.5): {comfort_emissions_score:.4f}")

            # Add score to log_entry if needed for CSV
            log_entry['comfort_emissions_score'] = comfort_emissions_score

        except Exception as e:
            print(f"Error scoring with Comfort & Emissions KPIs for episode {episode + 1}: {e}")

        # (Placeholder) agent.learn(comfort_emissions_score)

    # --- After training loop ---
    if episode_logs:
        df_logs = pd.DataFrame(episode_logs)
        try:
            df_logs.to_csv('kpi_logs_comfort_emissions.csv', index=False)
            print("\nKPI logs saved to kpi_logs_comfort_emissions.csv")
        except Exception as e:
            print(f"\nError saving KPI logs to CSV: {e}")
    else:
        print("\nNo episode logs were generated.")

    print("\nTraining simulation finished.")

if __name__ == "__main__":
    main()
