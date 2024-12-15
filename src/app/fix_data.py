import config 
import json

# Original machine dictionary
machines = {
    "Large Capacity Cutting Machine 1": "ast-yhccl1zjue2t",
    "Medium Capacity Cutting Machine 1": "ast-ha448od5d6bd",
    "Large Capacity Cutting Machine 2": "ast-6votor3o4i9l",
    "Medium Capacity Cutting Machine 2": "ast-5aggxyk5hb36",
    "Medium Capacity Cutting Machine 3": "ast-anxkweo01vv2",
    "Low Capacity Cutting Machine 1": "ast-6nv7viesiao7",
    "Laser Cutter": "ast-xpimckaf3dlf",
    "Laser Welding Machine 1": "ast-hnsa8phk2nay",
    "Laser Welding Machine 2": "ast-206phi0b9v6p",
    "Assembly Machine 1": "ast-pwpbba0ewprp",
    "Assembly Machine 2": "ast-upqd50xg79ir",
    "Assembly Machine 3": "ast-sfio4727eub0",
    "Testing Machine 1": "ast-nrd4vl07sffd",
    "Testing Machine 2": "ast-pu7dfrxjf2ms",
    "Testing Machine 3": "ast-06kbod797nnp",
    "Riveting Machine": "ast-o8xtn5xa8y87",
}

# Revert the dictionary
reverted_machine = {v: k for k, v in machines.items()}

# Load the JSON file from CLEANED_PREDICTED_DATA_PATH
with open(config.ORIGINAL_ADAPTED_DATA_PATH) as f:
    data = json.load(f)  # Assumes JSON file structure

for i in range(len(data["asset_id"])):
    data["name"][str(i)] = reverted_machine[data["asset_id"][str(i)]]

# Save the updated JSON back to the file (optional)
with open(config.ORIGINAL_ADAPTED_DATA_PATH, "w") as f:
    json.dump(data, f, indent=4)

# Print to verify
print("Updated Data:", json.dumps(data, indent=4))
