# Energy-based Dropout in Restricted Boltzmann Machines: Why not go random

*This repository holds all the necessary code to run the very-same experiments described in the paper "Energy-based Dropout in Restricted Boltzmann Machines: Why not go random".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```BibTex
@article{roder2020edrop,
  author={M. {Roder} and G. H. {de Rosa} and V. H. C. {de Albuquerque} and A. L. D. {Rossi} and J. P. {Papa}},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={Energy-Based Dropout in Restricted Boltzmann Machines: Why Not Go Random}, 
  year={2020},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TETCI.2020.3043764}}
}
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
*If you encounter any problems with the automatic installation of the [learnergy](https://github.com/gugarosa/learnergy) package, contact us.*

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
