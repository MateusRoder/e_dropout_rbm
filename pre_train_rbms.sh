# Defining a counter for iterating purposes
i=0

# Defining a constant for number of pre-trained RBMs
N_RBMS=15

# Defining a constant to hold the dataset
DATASET="mnist"

# Defining a constant to hold the type of RBM
MODEL="rbm"

# Creating a loop of `N_RBMS`
while [ $i -lt $N_RBMS ]; do
    # Pre-training amount of desired RBMs
    python rbm_reconstruction.py ${DATASET} ${MODEL} -n_hidden 128 -lr 0.1 -seed ${i}

    # Incrementing the counter
    i=$(($i+1))
done