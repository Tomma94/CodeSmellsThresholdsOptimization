#!/bin/bash

json_file="$1"


function reconstructed() {
  KEY=$1
  OUTPUT="output/$KEY";
  mkdir -p "$OUTPUT";

  NAME="run_${KEY}";
  COMMAND="module load anaconda	; source activate thresholds_new ; python -u main.py -n $KEY ${@:2}"
  sbatch --mail-type=FAIL --mail-user=tommas@post.bgu.ac.il --job-name="$NAME" --output="${OUTPUT}/${KEY}.txt" --wrap "$COMMAND";
  
}

# Check if the JSON file exists
if [[ ! -f "$json_file" ]]; then
    echo "File does not exist: $json_file"
    exit 1
fi

# Iterate over each key-value pair in the JSON file
jq -c 'to_entries[] | {key: .key, value: .value | to_entries | map("\(.key) \(.value)") | join(" ")}' "$json_file" | while read -r line; do
    key=$(echo "$line" | jq -r '.key')
    # Prepare arguments by converting array to a string of elements with flags and values
    args=$(echo "$line" | jq -r '.value')
    read -ra values_array <<< "$args"
    
    reconstructed "$key" "${values_array[@]}"
     
done

