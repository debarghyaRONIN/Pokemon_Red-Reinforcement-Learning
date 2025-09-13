import re
import json

def convert_to_linear_format(file_path, output_json):
    linear_format_dict = {}

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):  # Skip empty lines and comments
                continue

            # Match "Bank:Offset Label" format
            match = re.match(r"([0-9A-Fa-f]+):([0-9A-Fa-f]+)\s+(.+)", line)
            if match:
                bank = int(match.group(1), 16)  # Convert bank to integer
                offset = int(match.group(2), 16)  # Convert offset to integer
                label = match.group(3).strip()  # Extract label

                # Compute new address
                linear_address = f"0xD{offset:03X}"  # Convert offset to 0xDXXX format
                entry_number = offset % 8  # Assign entry number (simulating -X format)
                final_key = f"{linear_address}-{entry_number}"

                linear_format_dict[final_key] = label
            else:
                print(f"Skipping unrecognized line: {line}")  # Debugging output

    # Save the result as a JSON file
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(linear_format_dict, json_file, indent=4, ensure_ascii=False)

    print(f"Converted data saved to {output_json}")

# Path to your game.sym file
file_path = "game.sym"
output_json = "game_data.json"

# Run conversion and save JSON
convert_to_linear_format(file_path, output_json)
