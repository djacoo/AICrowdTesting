try:
    with open("train_agent.py", "r") as f:
        content_lines = f.readlines()
except FileNotFoundError:
    print("Error: train_agent.py not found.")
    exit(1)

current_content_str = "".join(content_lines)

# --- Stage 1: Global string replacements for non-LR schedule modifications ---
current_content_str = current_content_str.replace("TOTAL_TIMESTEPS_2KPI = 100000", "TOTAL_TIMESTEPS_2KPI = 200000  # MODIFIED: Increased from 100k", 1)
current_content_str = current_content_str.replace("TOTAL_TIMESTEPS_MULTI = 150000", "TOTAL_TIMESTEPS_MULTI = 250000  # MODIFIED: Increased from 150k", 1)
current_content_str = current_content_str.replace("net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])", "net_arch=dict(pi=[256, 256], vf=[256, 256]) # MODIFIED: Smaller network", 2)

# PPO model init arguments
current_content_str = current_content_str.replace("n_epochs=10,            # More epochs for better optimization",
                                                  "n_epochs=8,            # MODIFIED: Fewer epochs", 1)
current_content_str = current_content_str.replace("ent_coef=0.02,          # Slightly higher entropy for better exploration",
                                                  "ent_coef=0.025,          # MODIFIED: Increased entropy", 1)
current_content_str = current_content_str.replace("n_epochs=12,            # More epochs for better optimization",
                                                  "n_epochs=10,            # MODIFIED: Fewer epochs", 1)
current_content_str = current_content_str.replace("ent_coef=0.015,         # Slightly lower entropy for more focused learning",
                                                  "ent_coef=0.02,         # MODIFIED: Increased entropy", 1)

# --- Stage 2: Line-by-line processing for LR Schedules ---
lines = current_content_str.splitlines(True) # Keep line endings
new_lines = []
in_2kpi_lr_schedule_cosine_decay = False
modified_2kpi_lr_warmup = False
modified_2kpi_lr_min_max = False

in_multi_kpi_lr_schedule_cosine_decay = False
modified_multi_kpi_lr_warmup = False
modified_multi_kpi_lr_min_max = False

# Exact original lines for LR schedules (with leading spaces and trailing newline)
lr_2kpi_warmup_original = "            return 5e-4 * warmup_progress  # Higher initial learning rate\n"
lr_2kpi_min_original = "        min_lr = 1e-5\n"
lr_2kpi_max_original = "        max_lr = 5e-4  # Higher max learning rate\n"

lr_multi_kpi_warmup_original = "            return 4e-4 * warmup_progress  # Slightly lower initial learning rate than 2-KPI\n"
lr_multi_kpi_min_original = "        min_lr = 5e-6  # Slightly lower minimum learning rate for fine-tuning\n"
lr_multi_kpi_max_original = "        max_lr = 4e-4\n"


for i, line in enumerate(lines):
    appended = False # Flag to check if line was handled and appended

    # --- 2-KPI LR Schedule ---
    if not modified_2kpi_lr_warmup and line == lr_2kpi_warmup_original:
        new_lines.append("            return 3e-4 * warmup_progress  # MODIFIED: Lower initial LR\n")
        # Check if the next relevant lines are for cosine decay context
        if i + 2 < len(lines) and "progress = (1.0 - progress_remaining - warmup_frac) / (1.0 - warmup_fraction)" in lines[i+2]:
            in_2kpi_lr_schedule_cosine_decay = True
        modified_2kpi_lr_warmup = True
        appended = True
    elif in_2kpi_lr_schedule_cosine_decay and not modified_2kpi_lr_min_max:
        if line == lr_2kpi_min_original:
            new_lines.append("        min_lr = 5e-6 # MODIFIED: Lower min LR\n")
            appended = True
        elif line == lr_2kpi_max_original:
            new_lines.append("        max_lr = 3e-4 # MODIFIED: Lower max LR\n")
            # This block is done
            in_2kpi_lr_schedule_cosine_decay = False
            modified_2kpi_lr_min_max = True
            appended = True

    # --- Multi-KPI LR Schedule ---
    if not appended and not modified_multi_kpi_lr_warmup and line == lr_multi_kpi_warmup_original:
        new_lines.append("            return 2e-4 * warmup_progress  # MODIFIED: Lower initial LR\n")
        if i + 2 < len(lines) and "progress = (1.0 - progress_remaining - warmup_frac) / (1.0 - warmup_fraction)" in lines[i+2]:
            in_multi_kpi_lr_schedule_cosine_decay = True
        modified_multi_kpi_lr_warmup = True
        appended = True
    elif in_multi_kpi_lr_schedule_cosine_decay and not modified_multi_kpi_lr_min_max:
        if line == lr_multi_kpi_min_original:
            new_lines.append("        min_lr = 1e-6 # MODIFIED: Lower min LR\n")
            appended = True
        elif line == lr_multi_kpi_max_original:
            new_lines.append("        max_lr = 2e-4 # MODIFIED: Lower max LR\n")
            in_multi_kpi_lr_schedule_cosine_decay = False
            modified_multi_kpi_lr_min_max = True
            appended = True

    if not appended:
        new_lines.append(line)

final_content_str = "".join(new_lines)
num_modifications = final_content_str.count("# MODIFIED")

if num_modifications >= 14 :
    with open("train_agent.py", "w") as f:
        f.write(final_content_str)
    print(f"train_agent.py hyperparameters modified successfully. ({num_modifications} instances of '# MODIFIED').")
else:
    print(f"train_agent.py modification count low ({num_modifications}). Expected at least 14 '# MODIFIED' tags.")
    print("No file changes written due to insufficient modification count.")
