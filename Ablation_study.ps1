# Define the paths to the Python scripts
$script1 = ".\Experiment.py"
$script2 = "path\to\your\second_script.py"
$script3 = "path\to\your\third_script.py"

# Run the first Python script
Write-Host "Running first script: $script1"
python $script1 --ER
Start-Sleep -Seconds 2 # Pauses for 2 seconds. Adjust as needed.
Clear-Host # Clears the terminal

# # Run the second Python script
# Write-Host "Running second script: $script2"
# python $script2
# Start-Sleep -Seconds 2 # Pauses for 2 seconds. Adjust as needed.
# Clear-Host # Clears the terminal

# # Run the third Python script
# Write-Host "Running third script: $script3"
# python $script3
# Start-Sleep -Seconds 2 # Pauses for 2 seconds. Adjust as needed.
# Clear-Host # Clears the terminal at the end

# Write-Host "All scripts have been executed."
