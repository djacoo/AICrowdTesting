with open("AICrowdControl.py", "r") as f:
    content = f.readlines()

new_content = []
in_baseline_kpis_block = False
modification_applied = False
# Keep track of which keys have been modified to avoid issues if keys appear multiple times (though unlikely for a dict def)
keys_modified = {
    "ramping": False,
    "1-load_factor": False,
    "daily_peak": False,
    "all_time_peak": False
}

for line_idx, line in enumerate(content):
    if "BASELINE_KPIS = {" in line:
        in_baseline_kpis_block = True
        new_content.append(line)
        continue

    if in_baseline_kpis_block:
        original_line_appended = False
        if '"ramping"' in line and not keys_modified["ramping"]:
            new_content.append('    "ramping": 200.0,                 # MODIFIED: Baseline for load ramping (kW) - based on observed values ~3500-4500, new target closer to good performance\n')
            modification_applied = True
            keys_modified["ramping"] = True
        elif '"1-load_factor"' in line and not keys_modified["1-load_factor"]:
            new_content.append('    "1-load_factor": 0.10,             # MODIFIED: Baseline for load factor (1 - actual_load_factor) - based on observed values ~0.4-0.48, new target closer to good performance\n')
            modification_applied = True
            keys_modified["1-load_factor"] = True
        elif '"daily_peak"' in line and not keys_modified["daily_peak"]:
            # Ensure it's not "all_time_peak"
            if '"all_time_peak"' not in line:
                new_content.append('    "daily_peak": 20.0,                # MODIFIED: Baseline for daily peak demand (kW) - based on observed values ~82-90, new target closer to good performance\n')
                modification_applied = True
                keys_modified["daily_peak"] = True
            else: # It was "all_time_peak", append original and let next condition handle it
                new_content.append(line)
                original_line_appended = True
        elif '"all_time_peak"' in line and not keys_modified["all_time_peak"]:
            new_content.append('    "all_time_peak": 25.0,            # MODIFIED: Baseline for all-time peak demand (kW) - based on observed values ~90-98, new target closer to good performance\n')
            modification_applied = True
            keys_modified["all_time_peak"] = True

        if not original_line_appended and not any(key in line for key in keys_modified if keys_modified[key]):
             # If the line wasn't one of the modified ones (or a key that's already been processed this pass)
             # and it's not a line we just added, append the original line.
             # This check is a bit complex due to the direct append of new lines.
             # A safer way is to only append to new_content *after* all conditions.
             pass # This logic needs to be outside the conditional appends.

        if not any(key in line for key in ["ramping", "1-load_factor", "daily_peak", "all_time_peak"] if keys_modified[key] and key in line):
             if not original_line_appended: # if it wasn't all_time_peak masquerading as daily_peak
                new_content.append(line)

        if "}" in line: # End of the BASELINE_KPIS block
            # This check ensures we only process lines within the block.
            # If the "}" is on the same line as a KPI, it should have been handled.
            # If "}" is on its own line, it gets appended by the logic above.
            in_baseline_kpis_block = False
            # Append any remaining lines from original content if processing stopped early
            # This should not be needed if the loop continues to the end.
    else:
        new_content.append(line)

# Rebuild new_content more safely to ensure correct lines are added/replaced
final_content = []
in_baseline_kpis_block_rebuild = False
rebuild_modification_applied = False
keys_modified_rebuild = {
    "ramping": False,
    "1-load_factor": False,
    "daily_peak": False,
    "all_time_peak": False
}

for line in content:
    if "BASELINE_KPIS = {" in line:
        in_baseline_kpis_block_rebuild = True
        final_content.append(line)
        continue

    appended_this_iteration = False
    if in_baseline_kpis_block_rebuild:
        if '"ramping"' in line and not keys_modified_rebuild["ramping"]:
            final_content.append('    "ramping": 200.0,                 # MODIFIED: Baseline for load ramping (kW) - based on observed values ~3500-4500, new target closer to good performance\n')
            rebuild_modification_applied = True
            keys_modified_rebuild["ramping"] = True
            appended_this_iteration = True
        elif '"1-load_factor"' in line and not keys_modified_rebuild["1-load_factor"]:
            final_content.append('    "1-load_factor": 0.10,             # MODIFIED: Baseline for load factor (1 - actual_load_factor) - based on observed values ~0.4-0.48, new target closer to good performance\n')
            rebuild_modification_applied = True
            keys_modified_rebuild["1-load_factor"] = True
            appended_this_iteration = True
        elif '"daily_peak"' in line and not keys_modified_rebuild["daily_peak"] and '"all_time_peak"' not in line:
            final_content.append('    "daily_peak": 20.0,                # MODIFIED: Baseline for daily peak demand (kW) - based on observed values ~82-90, new target closer to good performance\n')
            rebuild_modification_applied = True
            keys_modified_rebuild["daily_peak"] = True
            appended_this_iteration = True
        elif '"all_time_peak"' in line and not keys_modified_rebuild["all_time_peak"]:
            final_content.append('    "all_time_peak": 25.0,            # MODIFIED: Baseline for all-time peak demand (kW) - based on observed values ~90-98, new target closer to good performance\n')
            rebuild_modification_applied = True
            keys_modified_rebuild["all_time_peak"] = True
            appended_this_iteration = True

        if not appended_this_iteration:
            final_content.append(line)

        if "}" in line: # End of the BASELINE_KPIS block
            in_baseline_kpis_block_rebuild = False
    else:
        final_content.append(line)


if rebuild_modification_applied:
    with open("AICrowdControl.py", "w") as f:
        f.writelines(final_content)
    print("AICrowdControl.py BASELINE_KPIS modified successfully.")
else:
    print("Could not find BASELINE_KPIS block or relevant keys in AICrowdControl.py. No changes made.")
