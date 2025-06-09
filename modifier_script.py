# Task: Modify CustomCityEnv.py to correctly use AICrowdControl.score()

# Read the content of CustomCityEnv.py
try:
    with open("CustomCityEnv.py", "r") as f:
        content = f.readlines()
except FileNotFoundError:
    print("Error: CustomCityEnv.py not found.")
    exit(1)

# Simpler strategy:
# 1. Read all lines.
# 2. Find the line `base_reward = float(self.reward_calculator.score(raw_kpis))`
# 3. After this, find `reward_components = {`. Mark its start.
# 4. Find the matching `}` for this dict. Mark its end.
# 5. Find `reward = sum(reward_components.values())`.
# 6. Replace the sum line with `reward = base_reward`.
# 7. Comment out the `reward_components` dict.

modified_lines = list(content) # Work on a copy
found_base_reward_line_idx = -1
reward_components_block_indices = [] # Will store [start_idx, end_idx]
sum_line_index = -1
reward_section_active = False # To ensure we are in the `if done:` block's reward calculation part

# First, locate the `if done:` block and the `base_reward` calculation within step
step_func_idx = -1
done_block_idx = -1

for i, line_text in enumerate(modified_lines):
    if "def step(self, action):" in line_text:
        step_func_idx = i
        continue
    if step_func_idx != -1 and "if done:" in line_text and i > step_func_idx :
        done_block_idx = i
        continue

    if done_block_idx != -1 and \
       "base_reward = float(self.reward_calculator.score(raw_kpis))" in line_text and \
       i > done_block_idx:
        found_base_reward_line_idx = i
        reward_section_active = True # We are in the relevant part

    if reward_section_active:
        if "reward_components = {" in line_text and not reward_components_block_indices:
            # Ensure this is after base_reward has been found
            if found_base_reward_line_idx != -1 and i > found_base_reward_line_idx:
                reward_components_block_indices.append(i) # Start index
                open_braces = 0
                for char_in_line in line_text: # Count braces on the starting line itself
                    if char_in_line == '{':
                        open_braces += 1
                    elif char_in_line == '}':
                        open_braces -=1

                if open_braces == 0: # Dict is a single line
                    reward_components_block_indices.append(i) # End index is same as start
                else: # Multi-line dictionary
                    for j in range(i + 1, len(modified_lines)):
                        for char_in_line_j in modified_lines[j]:
                            if char_in_line_j == '{':
                                open_braces += 1
                            elif char_in_line_j == '}':
                                open_braces -=1
                        if open_braces == 0:
                            reward_components_block_indices.append(j) # End index
                            break

        if reward_components_block_indices and len(reward_components_block_indices) == 2:
            # Check if we are past the reward_components block and find the sum line
            if i > reward_components_block_indices[1] and "reward = sum(reward_components.values())" in line_text:
                sum_line_index = i
                # Found all markers, no need to search further in this active section
                reward_section_active = False # Reset, block found
                break

    # If we leave the step function or a new function starts, reset relevant indices
    if "def " in line_text and "def step" not in line_text and i > step_func_idx and step_func_idx != -1:
        step_func_idx = -1
        done_block_idx = -1
        found_base_reward_line_idx = -1
        reward_section_active = False
        reward_components_block_indices = []
        sum_line_index = -1


# Perform the modification
if found_base_reward_line_idx != -1 and \
   len(reward_components_block_indices) == 2 and \
   sum_line_index != -1:

    print("Applying modifications to CustomCityEnv.py reward calculation.")

    # Comment out reward_components dictionary
    first_line_indent = modified_lines[reward_components_block_indices[0]][:modified_lines[reward_components_block_indices[0]].find("reward_components")]
    for i in range(reward_components_block_indices[0], reward_components_block_indices[1] + 1):
        current_line_text = modified_lines[i].lstrip()
        # Ensure the comment preserves the original line ending if it had one
        comment_prefix = first_line_indent + "# MODIFIED: "
        original_line_content = modified_lines[i].lstrip().rstrip('\n')
        modified_lines[i] = comment_prefix + original_line_content + '\n'


    # Replace the sum line
    indent = modified_lines[sum_line_index][:modified_lines[sum_line_index].find("reward")]
    modified_lines[sum_line_index] = indent + "reward = base_reward # MODIFIED: Use score from AICrowdControl\n"

    # Add a blank line after for readability if it's not there
    # and the next line is not empty and not a comment
    if sum_line_index + 1 < len(modified_lines):
        next_line_stripped = modified_lines[sum_line_index + 1].strip()
        if next_line_stripped != "" and not next_line_stripped.startswith("#"):
             modified_lines.insert(sum_line_index + 1, indent + "\n")


    with open("CustomCityEnv.py", "w") as f:
        f.writelines(modified_lines)
    print("CustomCityEnv.py modified successfully.")
else:
    print("Could not find the exact lines to modify in CustomCityEnv.py. No changes made.")
    print(f"Debug info:")
    print(f"  Found base_reward line index: {found_base_reward_line_idx}")
    print(f"  Reward components block indices: {reward_components_block_indices}")
    print(f"  Sum line index: {sum_line_index}")
