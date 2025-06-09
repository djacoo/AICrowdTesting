try:
    with open("CustomCityEnv.py", "r") as f:
        content = f.readlines()
except FileNotFoundError:
    print("Error: CustomCityEnv.py not found.")
    exit(1)

modification_applied = False
# More robust modification loop:
idx = 0
final_lines = []
while idx < len(content):
    current_line = content[idx]
    if "if comfort_score > 0.95:" in current_line:
        final_lines.append(current_line) # Add the 'if' line
        if idx + 1 < len(content) and "reward += 0.5" in content[idx+1]:
            # Ensure the line ends with a newline
            replacement_line = content[idx+1].replace("reward += 0.5", "reward += 0.05 # MODIFIED: Bonus reduced").rstrip('\n') + '\n'
            final_lines.append(replacement_line)
            idx += 1 # Increment to skip the original 'reward += 0.5' line
            modification_applied = True
        # else: if next line is not the expected one, current 'if' line is already added. Next line will be added in next iteration.

    elif "if emissions_score > 0.9:" in current_line:
        final_lines.append(current_line) # Add the 'if' line
        if idx + 1 < len(content) and "reward += 0.5" in content[idx+1]:
            # Ensure the line ends with a newline
            replacement_line = content[idx+1].replace("reward += 0.5", "reward += 0.05 # MODIFIED: Bonus reduced").rstrip('\n') + '\n'
            final_lines.append(replacement_line)
            idx += 1 # Increment to skip the original 'reward += 0.5' line
            modification_applied = True
        # else: similar to above.

    else:
        final_lines.append(current_line)
    idx += 1

if modification_applied:
    with open("CustomCityEnv.py", "w") as f:
        f.writelines(final_lines)
    print("CustomCityEnv.py bonuses modified successfully.")
else:
    print("Could not find the exact bonus lines to modify in CustomCityEnv.py. No changes made.")
