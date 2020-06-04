# Defining a constant to hold the dataset
DATASET="mnist"

# Defining a constant to hold the possible meta-heuristics
MH=("ba" "bh" "de" "ga" "pso")

# Creating a loop of meta-heuristics
for i in "${MH[@]}"; do
    # Performs the optimization procedure
    python drbm_optimization.py ${DATASET} ${MH} -n_hidden 128 -lr 0.1 -seed 0
done