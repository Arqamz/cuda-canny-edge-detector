#!/bin/bash

make clean

make 

# Define the input directory
INPUT_DIR="input"

# Define test parameters (sigma, tlow, thigh)
declare -a params=(
    "4 0.25 0.5"
    "0.8 0.2 0.4"
    "10 0.3 0.6"
)

# Get all pgm files in the input directory
files=$(find $INPUT_DIR -name "*.pgm")

# Log file for results
LOG_FILE="canny_benchmark_results.txt"
echo "Performance Results" > $LOG_FILE
echo "===================" >> $LOG_FILE
echo "Format: algorithm,image,sigma,tlow,thigh,run" >> $LOG_FILE
echo "" >> $LOG_FILE

# Function to run tests
run_test() {
    algorithm=$1
    image=$2
    sigma=$3
    tlow=$4
    thigh=$5
    
    # Add separator for the algorithm run
    echo "%%%%%% Algorithm: $algorithm %%%%%%" >> $LOG_FILE
    echo "" >> $LOG_FILE

    for run in {1..3}; do
        # Add separator for each individual run
        echo "&&&& Running on Image: $(basename $image) &&&&" >> $LOG_FILE
        echo "" >> $LOG_FILE
        
        # Add parameter details
        echo "Running with parameters: sigma=$sigma, tlow=$tlow, thigh=$thigh (Run $run)" >> $LOG_FILE
        echo "" >> $LOG_FILE

        # Run the binary and capture output, redirect to the log file
        $algorithm $image $sigma $tlow $thigh >> $LOG_FILE 2>&1
        
        # Add separation after each run
        echo "-------------------------------------------" >> $LOG_FILE
        echo "" >> $LOG_FILE
    done
}

# Loop through each image and run all combinations of parameters
for file in $files; do
    # Add separation for each image
    echo "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" >> $LOG_FILE
    echo "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" >> $LOG_FILE
    echo "Processing Image: $(basename $file)" >> $LOG_FILE
    echo "" >> $LOG_FILE

    for param in "${params[@]}"; do
        # Split parameters
        read sigma tlow thigh <<< $param
        
        # Add separation for each parameter combination
        echo "%%%%%% Parameters: sigma=$sigma, tlow=$tlow, thigh=$thigh %%%%%%" >> $LOG_FILE
        echo "" >> $LOG_FILE

        # Run CUDA version
        run_test "bin/canny_cuda" "$file" "$sigma" "$tlow" "$thigh"
        
        # Run serial version
        run_test "bin/canny_serial" "$file" "$sigma" "$tlow" "$thigh"
        
        # Add extra line break after each parameter set
        echo "" >> $LOG_FILE
    done
done

echo "All tests completed. Results saved to $LOG_FILE"

echo "Outputs are in output/ directory"