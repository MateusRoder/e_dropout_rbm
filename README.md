# Energy-based Dropout in Restricted Boltzmann Machines: Why not go random

*This repository holds all the necessary code to run the very-same experiments described in the paper "Energy-based Dropout in Restricted Boltzmann Machines: Why not go random".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
RODER, M. ; ROSA, G. ; de Albuquerque, V. H. C. ; ROSSI, A. L. D. ; PAPA, JOAO P . Energy-based Dropout in Restricted Boltzmann Machines: Why not go random. IEEE Transactions on Emerging Topics in Computational Intelligence, 2020.
```

---

## Structure

 * `utils`
   * `loader.py`: Utility to load datasets and split them into training, validation and testing sets;
   * `objects.py`: Wraps objects instantiation for command line usage;
   
   
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```
* If you encounter any problems with the automatic installation of the "learnergy" package, it is possible that its direct installation from the "https://github.com/gugarosa/learnergy" repository corrects incompatibilities, but do not hesitate in contact us.

### Data configuration

In order to run the experiments, you can use `torchvision` to load pre-implemented datasets.

---

## Usage

### Model Training and Reconstruction

The experiment is conducted by pre-training an RBM architecture and post-evaluating them. To accomplish such a step, one needs to use the following script:

```Python
python rbm_reconstruction.py -h
```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate parameters.*

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell script, as follows:

```Bash
./pipeline.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this paper. Furthermore, one can change any input argument that is defined in the script.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or mateus.roder@unesp.br and gustavo.rosa@unesp.br.

---
