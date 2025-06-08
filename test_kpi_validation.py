import numpy as np

# Assuming AICrowdControl.py is in the same directory or accessible in PYTHONPATH
try:
    from AICrowdControl import ControlTrackReward
except ImportError:
    print("Warning: AICrowdControl.py not found. Using a placeholder for ControlTrackReward.")
    # Define a placeholder class if the real one isn't available
    class ControlTrackReward:
        def __init__(self, baseline, phase=None):
            self.baseline = baseline
            self.phase = phase
            print("Placeholder ControlTrackReward instantiated.")

        def carbon_emissions(self, e, emission_rate):
            if not isinstance(e, np.ndarray): raise TypeError("Input 'e' must be a numpy array.")
            if not isinstance(emission_rate, (list, np.ndarray)): raise TypeError("Input 'emission_rate' must be a list or numpy array.")
            if len(emission_rate) == 0: raise ValueError("Input 'emission_rate' cannot be empty.")
            return np.sum(e[:, 0] * emission_rate[0])

        def unmet_hours(self, temp, setpoint, band, occupancy):
            if not all(isinstance(arr, np.ndarray) for arr in [temp, setpoint, occupancy]): raise TypeError("Inputs 'temp', 'setpoint', 'occupancy' must be numpy arrays.")
            if not isinstance(band, (float, int)): raise TypeError("Input 'band' must be a number.")
            if temp.shape != setpoint.shape or temp.shape != occupancy.shape: raise ValueError("Shape mismatch.")
            if temp.size == 0: return 0.0
            lower = setpoint - band
            upper = setpoint + band
            unmet = (temp < lower) | (temp > upper)
            return np.sum(unmet & (occupancy == 1)) / temp.size

        def ramping(self, district_consumption):
            if not isinstance(district_consumption, (list, np.ndarray)): raise TypeError("Input 'district_consumption' must be a list or numpy array.")
            dc = np.array(district_consumption)
            return 0.0 if len(dc) < 2 else np.sum(np.abs(np.diff(dc)))

        def load_factor(self, district_consumption, hours_per_day): # hours_per_day unused by this logic
            if not isinstance(district_consumption, (list, np.ndarray)): raise TypeError("Input 'district_consumption' must be a list or numpy array.")
            dc = np.array(district_consumption)
            if dc.size == 0: return 0.0
            min_load, max_load = np.min(dc), np.max(dc)
            if max_load == 0: return 1.0 if min_load == 0 else 0.0 # All zero demand -> LF=1, else (min<0 or min>0 with max=0) LF=0
            return min_load / max_load


        def daily_peak(self, district_consumption, hours_per_day):
            if not isinstance(district_consumption, (list, np.ndarray)): raise TypeError("Input 'district_consumption' must be a list or numpy array.")
            if not isinstance(hours_per_day, int) or hours_per_day <= 0: raise ValueError("hours_per_day must be a positive integer.")
            dc = np.array(district_consumption)
            if dc.size == 0: return 0.0
            if dc.size % hours_per_day != 0: print(f"Warning: data size {dc.size} not multiple of hours_per_day {hours_per_day}.")
            num_days = dc.size // hours_per_day
            if num_days == 0: return np.max(dc) if dc.size > 0 else 0.0
            daily_peaks = [np.max(dc[i*hours_per_day:(i+1)*hours_per_day]) for i in range(num_days)]
            return np.mean(daily_peaks) if daily_peaks else 0.0

        def all_time_peak(self, district_consumption):
            if not isinstance(district_consumption, (list, np.ndarray)): raise TypeError("Input 'district_consumption' must be a list or numpy array.")
            dc = np.array(district_consumption)
            return 0.0 if dc.size == 0 else np.max(dc)

        def thermal_resilience(self, temp, setpoint, band, occupancy, outage):
            if not all(isinstance(arr, np.ndarray) for arr in [temp, setpoint, occupancy, outage]): raise TypeError("Inputs must be numpy arrays.")
            if not isinstance(band, (float, int)): raise TypeError("Band must be number.")
            if not (temp.ndim == 2 and temp.shape == setpoint.shape and temp.shape == occupancy.shape): raise ValueError("Shape mismatch for temp, setpoint, occupancy.")
            if not (outage.ndim == 1 and outage.shape[0] == temp.shape[0]): raise ValueError("Shape mismatch for outage array.")
            num_total_zones = temp.shape[0]
            if num_total_zones == 0: return 1.0
            resilient_zones_under_outage_count = 0
            lower_bound, upper_bound = setpoint - band, setpoint + band
            for z in range(num_total_zones):
                if outage[z]:
                    is_zone_resilient = True
                    for t in range(temp.shape[1]):
                        if occupancy[z, t] == 1 and (temp[z, t] < lower_bound[z, t] or temp[z, t] > upper_bound[z, t]):
                            is_zone_resilient = False; break
                    if is_zone_resilient: resilient_zones_under_outage_count += 1
            return resilient_zones_under_outage_count / num_total_zones

        def unserved_energy(self, demand, served, outage):
            """
            Placeholder for unserved_energy.
            Calculates total unserved energy during outages / total demand during outages.
            demand, served: (consumers, timesteps)
            outage: (consumers,) boolean array
            """
            if not all(isinstance(arr, np.ndarray) for arr in [demand, served, outage]):
                raise TypeError("Inputs must be numpy arrays.")
            if not (demand.ndim == 2 and demand.shape == served.shape):
                raise ValueError("Shape mismatch for demand and served arrays.")
            if not (outage.ndim == 1 and outage.shape[0] == demand.shape[0]):
                raise ValueError("Shape mismatch for outage array.")

            num_consumers = demand.shape[0]
            if num_consumers == 0: return 0.0 # No consumers, no unserved energy

            total_unserved_during_outage = 0.0
            total_demand_during_outage = 0.0

            for c in range(num_consumers):
                if outage[c]: # Only consider consumers affected by outage
                    for t in range(demand.shape[1]): # Iterate through timesteps
                        current_demand = demand[c, t]
                        current_served = served[c, t]

                        unserved_this_step = max(0.0, current_demand - current_served)

                        total_unserved_during_outage += unserved_this_step
                        total_demand_during_outage += current_demand

            if total_demand_during_outage == 0:
                # If no demand during outages (e.g. all outage zones had zero demand),
                # then unserved energy fraction is 0.
                return 0.0

            ratio = total_unserved_during_outage / total_demand_during_outage
            return np.clip(ratio, 0.0, 1.0)


# Dummy baseline
dummy_baseline = {
    "co2_emissions": 0, "fuel_consumption": 0, "travel_time": 0, "idle_time": 0,
    "speed_limit_violations": 0, "distance_travelled": 0, "stops": 0,
    "unmet_hours": 0, "ramping": 0, "load_factor": 0, "daily_peak": 0,
    "all_time_peak": 0, "thermal_resilience": 0, "unserved_energy": 0,
}

class TestKPIValidation:
    def __init__(self):
        self.reward_calculator = ControlTrackReward(baseline=dummy_baseline)

    def run_test(self, test_name, method_name, inputs, expected_output):
        method_to_call = getattr(self.reward_calculator, method_name)
        calculated_output = method_to_call(**inputs)
        assert np.isclose(calculated_output, expected_output), \
            f"{test_name} test failed: Expected {expected_output}, got {calculated_output}"
        print(f"{test_name} passed.")

    def test_carbon_emissions(self):
        inputs = {'e': np.array([[1.0, 2.0], [0.5, 1.5], [2.0, 0.0]]), 'emission_rate': [0.5, 0.6, 0.4]}
        self.run_test("test_carbon_emissions", "carbon_emissions", inputs, 1.75)

    def test_unmet_hours(self):
        inputs = {'temp': np.array([[22.0, 23.0], [25.0, 21.0], [20.0, 26.0]]), 'setpoint': np.array([[22.5, 22.5], [22.5, 22.5], [22.5, 22.5]]), 'band': 1.0, 'occupancy': np.array([[1, 1], [1, 0], [1, 1]])}
        self.run_test("test_unmet_hours", "unmet_hours", inputs, 0.5)

    def test_ramping(self):
        inputs = {'district_consumption': [10.0, 12.0, 9.0, 15.0, 14.0]}
        self.run_test("test_ramping", "ramping", inputs, 12.0)

    def test_load_factor(self):
        inputs = {'district_consumption': [5.0, 10.0, 5.0, 10.0, 10.0, 20.0, 10.0, 20.0], 'hours_per_day': 4}
        self.run_test("test_load_factor", "load_factor", inputs, 0.25)

    def test_daily_peak(self):
        inputs = {'district_consumption': [5.0, 10.0, 8.0, 6.0, 12.0, 9.0, 15.0, 11.0], 'hours_per_day': 4}
        self.run_test("test_daily_peak", "daily_peak", inputs, 12.5)

    def test_all_time_peak(self):
        inputs = {'district_consumption': [5.0, 10.0, 8.0, 6.0, 12.0, 9.0, 15.0, 11.0]}
        self.run_test("test_all_time_peak", "all_time_peak", inputs, 15.0)

    def test_thermal_resilience(self):
        inputs = {'temp': np.array([[25.0, 22.0], [26.0, 23.0], [20.0, 25.5], [22.0, 22.0]]), 'setpoint': np.array([[22.5, 22.5], [22.5, 22.5], [22.5, 22.5], [22.5, 22.5]]), 'band': 1.0, 'occupancy': np.array([[1, 1], [1, 0], [1, 1], [0, 1]]), 'outage': np.array([True, True, False, True])}
        self.run_test("test_thermal_resilience", "thermal_resilience", inputs, 0.25)

    def test_unserved_energy(self):
        inputs = {
            'demand': np.array([[10.0, 5.0], [12.0, 6.0], [8.0, 4.0]]),
            'served': np.array([[8.0, 5.0], [10.0, 3.0], [8.0, 2.0]]),
            'outage': np.array([True, True, False])
        }
        self.run_test("test_unserved_energy", "unserved_energy", inputs, 7.0 / 33.0)

if __name__ == "__main__":
    print("Executing KPI Validation Test Script...")
    kpi_tester = TestKPIValidation()

    kpi_tester.test_carbon_emissions()
    kpi_tester.test_unmet_hours()
    kpi_tester.test_ramping()
    kpi_tester.test_load_factor()
    kpi_tester.test_daily_peak()
    kpi_tester.test_all_time_peak()
    kpi_tester.test_thermal_resilience()
    kpi_tester.test_unserved_energy()

    print("KPI Validation Test Script finished successfully.")
