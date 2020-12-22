# Common variables definition
DATA="mnist"
BATCH_SIZE=256
EPOCHS=50
DEVICE="gpu"
N_RUNS=10

# Architecture variables
MODEL="edrbm"
N_VISIBLE=784
N_HIDDEN=1024
STEPS=1
LR=0.1
MOMENTUM=0
DECAY=0.0
T=1
P=0.5

# Iterates through all possible seeds
for SEED in $(seq 1 $N_RUNS); do
    # Pre-training amount of desired RBMs
    python rbm_reconstruction.py ${DATA} ${MODEL} -n_visible ${N_VISIBLE} -n_hidden ${N_HIDDEN} -steps ${STEPS} -lr ${LR} -momentum ${MOMENTUM} -decay ${DECAY} -temperature ${T} -p ${P} -batch_size ${BATCH_SIZE} -epochs ${EPOCHS} -device ${DEVICE} -seed ${SEED}
done
