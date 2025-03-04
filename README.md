# CKMS: A Test of the Efficiency of a Given Portfolio in High Dimensions

## Overview

This Python package provides an implementation of the high-dimensional Gibbons, Ross and Shanken test as proposed in Chernov, Kelly, Malamud and Schwab (2025). 

## Installation

Clone the repository manually:

```bash
git clone https://github.com/DjoFE2021/CKMSEfficiencyTest.git
cd CKMSEfficiencyTest
conda env create -n <YOUR_ENV_NAME> -f environment.yml
```

## Usage

Here is a basic example of how to use the package:

```python
import numpy as np
from tests.CKMS import CKMS
tilde_z_grid = [0.1, 1, 10, 100]
R = np.random.normal(0, 1, (100, 1000))
R_M = np.random.normal(0, 1, (1, 1000))
test = CKMS(z = tilde_z_grid)
test.test(r = R,
          r_M = R_M,
          adjust_grid = True,
          find_optimal_z = True)
print(test.summary())
```

## Citation
If you use this package in your research, please cite the following paper:

> [Paper Title] - [Authors]  
> [Journal/Conference Name, Year]  
> [DOI/Link]

## License

This package is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you find bugs or have suggestions for improvements.

## Contact
For any inquiries, please reach out via [GitHub Issues](https://github.com/DjoFE2021/CKMSEfficiencyTest.git/issues) or email [your_email@example.com].

