# CKMS: A Test of the Efficiency of a Given Portfolio in High Dimensions

## Overview

This Python package provides an implementation of the classical Gibbons-Ross-Shanken (GRS) test, as well as its high-dimension counterparts as proposed in Chernov, Kelly, Malamud and Schwab (2025). 

## Installation

You can install the package using pip:

```bash
pip install git+https://github.com/YOUR_GITHUB_USERNAME/GRS-RMT.git
```

Or clone the repository manually:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/GRS-RMT.git
cd GRS-RMT
pip install -r requirements.txt
```

## Usage

Here is a basic example of how to use the package:

```python
#INSERT EXAMPLE HERE
print(result)
```

## API Reference

### `grs_test(test_assets, factors, lambda_reg=0.1)`
Runs the GRS test with regularization.

- `test_assets` (numpy.ndarray): Matrix of test asset returns.
- `factors` (numpy.ndarray): Matrix of factor returns.
- `lambda_reg` (float, optional): Regularization parameter for covariance estimation.
- **Returns:** Dictionary with test statistic, p-value, and additional diagnostics.

### `generate_synthetic_data(n_assets, n_obs)`
Generates synthetic test asset returns for simulation purposes.

- `n_assets` (int): Number of test assets.
- `n_obs` (int): Number of observations.
- **Returns:** Simulated asset return matrix.

### `generate_factor_data(n_factors, n_obs)`
Generates synthetic factor returns.

- `n_factors` (int): Number of factors.
- `n_obs` (int): Number of observations.
- **Returns:** Simulated factor return matrix.

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
For any inquiries, please reach out via [GitHub Issues](https://github.com/YOUR_GITHUB_USERNAME/GRS-RMT/issues) or email [your_email@example.com].

