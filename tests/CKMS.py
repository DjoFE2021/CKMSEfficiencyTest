"""
This file contains class to perform CKMS test on both single and multiple portfolios.
"""

#Standard imports
import numpy as np
from scipy.stats import norm

from typing import List
from tabulate import tabulate


class CKMS:
    """
    This class implements every method needed to perform the CKMS test.
    """

    def __init__(self,
                 z=None) -> None:

        """
        The constructor for HDGRS class.
        
        Parameters
        ----------
        z : List[float]
            The ridge penatly to use for the HDGRS test.
        """

        if z is None:
            z = [1e-5]
        self.z = z
        self.z_save = z
        self.test_results = None

    def test(self,
             r: np.ndarray,
             r_M: np.ndarray,
             adjust_grid: bool = True,
             find_optimal_z: bool = False) -> float:

        """
        This function performs the regression and then performs HDGRS test. It is used if z is a fixed list of scalars.
        
        Parameters
        ----------
        r : np.ndarray
            The returns of the stocks. It should be a P x T matrix, where T is the number of observations and P is the number of stocks.
            
        r_M : np.ndarray
            The returns of the factors portfolio. It should be a L x T matrix, where T is the number of observations and L is the number of factors.
            
        adjust_grid : bool, default = True
            Whether to adjust the grid of z to be correctly scaled to the data.
            
        find_optimal_z : bool, default = False
            Whether to perform Leave-One-Out estimation of the power to pick optimal z in the grid
            
        Returns
        -------
        res : dict
            A dictionary containing the test statistic and other informations
        """

        #* 0. Define some useful variables
        p = r.shape[0]
        t = r.shape[1]
        l = r_M.shape[0]
        c = p / t

        ones_t = np.ones((t, 1))
        r_M_bar = r_M @ ones_t / t
        x = np.vstack([ones_t.T, r_M])
        omega_M = r_M @ r_M.T / t - r_M_bar @ r_M_bar.T
        omega_M_inv = np.linalg.inv(omega_M)
        quad_M_Morm = r_M.T @ omega_M_inv @ r_M

        #* 1. Compute hat M(z), we also compute parameters estimate here.
        alpha_tilde = 1 / t * ones_t * (1 + ones_t.T @ quad_M_Morm @ ones_t / t ** 2) - 1 / t * quad_M_Morm @ ones_t / t
        alpha_bar = r @ alpha_tilde
        beta_hat = 1 / t * r @ r_M.T @ omega_M_inv - 1 / t * r @ ones_t @ r_M_bar.T @ omega_M_inv

        B_hat = np.hstack([alpha_bar, beta_hat])

        #* Adjust the grid
        if adjust_grid:
            alpha_hat = r - beta_hat @ r_M

            #* 2. mean(trace(T^{-1}\hat\alpha'\hat\alpha \in \mathbb{R}^{TxT})
            E_X = np.mean((alpha_hat ** 2))
            self.z = E_X * np.array(self.z)

        # * 1.1 We compute the case where P > T, we use inverse directly because using eigenvalue decomposition would
        # also imply needing to inverse Q and is in fact much slower.
        if p > t:

            #We consider tilde_sigma which is a asymmetric txt matrix
            tilde_sigma = 1 / t * (r.T @ r - r.T @ B_hat @ x)

            tilde_sigma_inv = np.array([np.linalg.inv(self.z[i] * np.eye(t) + tilde_sigma) for i in range(len(self.z))])

            hat_M_z = (1 / (1 + r_M_bar.T @ omega_M_inv @ r_M_bar)) * alpha_tilde.T @ tilde_sigma_inv @ r.T @ alpha_bar

        # * 1.2 We compute the case where P =< T, we use eigenvalue decomposition as it is needed later and therefore
        # we reduce computational cost
        else:

            #We consider the usual hat_sigma which is a symmetric pxp matrix
            hat_sigma = 1 / t * (r @ r.T - B_hat @ x @ r.T)

            #hat_sigma is symmetric, so we can use np.linalg.eigh
            eigvals, eigvecs = np.linalg.eigh(hat_sigma)

            #This is exactly the same as np.linalg.inv(hat_sigma + np.eye(p)*self.z)
            hat_sigma_inv = eigvecs @ np.array(
                [np.diag(1 / (self.z[i] + eigvals)) for i in range(len(self.z))]) @ eigvecs.T

            hat_M_z = (1 / (1 + r_M_bar.T @ omega_M_inv @ r_M_bar)) * alpha_bar.T @ hat_sigma_inv @ alpha_bar

        #* 2. Compute hat m(z)
        #* 2.1 We handle the case where P > T
        if p > t:
            hat_m_z = 1 / p * np.array([np.trace(tilde_sigma_inv[i]) for i in range(len(self.z))]) + (p - t) / (
                        p * np.array(self.z))

        #*2.2 We handle the case where P <=T
        else:
            hat_m_z = 1 / p * np.array([np.sum(1 / (eigvals + self.z[i])) for i in range(len(self.z))])

        #* 3. Compute hat m'(z)
        if p > t:
            hat_m_prime_z = 1 / p * np.array(
                [np.trace(tilde_sigma_inv[i] @ tilde_sigma_inv[i]) for i in range(len(self.z))]) + (p - t) / (
                                        p * np.array(self.z) ** 2)
        else:
            hat_m_prime_z = 1 / p * np.array([np.sum(1 / (eigvals + self.z[i]) ** 2) for i in range(len(self.z))])

        #* 4. Compute hat xi(z)
        hat_xi_z = 1 / (1 - c + c * np.array(self.z) * hat_m_z) - 1

        #* 5. Compute hat xi'(z)
        hat_xi_prime_z = ((hat_xi_z + 1) ** 2) * c * (np.array(self.z) * hat_m_prime_z - hat_m_z)

        #* 6. Compute hat_delta_z
        hat_delta_z = 2 * (hat_xi_z + np.array(self.z) * hat_xi_prime_z) * (hat_xi_z + 1) ** 2

        #* 6. Compute the test statistic
        test_stat = (t ** (1 / 2)) * (hat_M_z.reshape(1, -1) - hat_xi_z) / np.sqrt(hat_delta_z)

        #* 8 Find optimal z
        best_z_ind = None
        if find_optimal_z:
            h_zc = self.get_estimated_h(r, r_M)
            power_max = h_zc / np.sqrt(hat_delta_z)
            best_z_ind = np.argmax(power_max)

        self.test_results = {"test_stat": test_stat,
                             "p_value": 1 - norm.cdf(test_stat),
                             "P": p,
                             "T": t,
                             "L": l,
                             "c": c,
                             "init_grid": self.z_save,
                             "z_grid": self.z,
                             "best_z": self.z[best_z_ind],
                             "best_z_ind": best_z_ind,
                             }

    def get_estimated_h(self,
                        r: np.ndarray,
                        r_M: np.ndarray) -> np.ndarray:

        """
        get_estimated_H is a function that computes the estimated H(z;c) for each point in the grid.
        
        Parameters
        ----------
        r : np.ndarray
            The returns of the stocks. It should be a P x T matrix, where T is the number of observations and P is the number of stocks.
        
        r_M : np.ndarray
            The returns of the factors portfolio. It should be a L x T matrix, where T is the number of observations and L is the number of factors.
            
        Returns
        -------
        H : np.ndarray
            The estimated H(z;c) for each point in the grid.
        """

        #* 0. Define some usefule variables
        p = r.shape[0]
        t = r.shape[1]

        #! Notation differs from other implementation
        #*1. Compute \hat\alpha \in \mathbb{R}^{PxT}
        ones_t = np.ones((t, 1))
        r_M_bar = r_M @ ones_t / t
        omega_M = r_M @ r_M.T / t - r_M_bar @ r_M_bar.T
        omega_M_inv = np.linalg.inv(omega_M)

        beta_hat = 1 / t * r @ r_M.T @ omega_M_inv - 1 / t * r @ ones_t @ r_M_bar.T @ omega_M_inv
        alpha_hat = r - beta_hat @ r_M

        #* 2. Compute A = T^{-1}\hat\alpha'\hat\alpha \in \mathbb{R}^{TxT}
        A = 1 / t * alpha_hat.T @ alpha_hat
        grid = np.array(self.z)

        #* 3. Compute G(z) = (zI + A)^{-1}A for each point in the grid
        G = np.array([np.linalg.inv(z * np.eye(t) + A) @ A for z in grid])

        #* 4. Assign w and W
        w = np.array([np.diag(G[i]) for i in range(len(G))])
        W = G @ ones_t

        #* 5. Compute H(z;c)
        scaler = 1 / (1 - 1 / t * np.array([ones_t.T @ W[i] for i in range(len(grid))])).flatten()
        sums = np.array([np.sum((W[i].flatten() - w[i]) / (1 - w[i])) for i in range(len(grid))])
        H = sums * scaler

        return H

    def summary(self):

        """Prints a well-formatted summary of the test results."""

        if self.test_results['best_z_ind'] is None:
            table = [
                ["P", f"{self.test_results['P']}"],
                ["T", f"{self.test_results['T']}"],
                ["L", f"{self.test_results['L']}"],
                ["c", f"{self.test_results['c']}"],
                ["Best z", ""],
                ["Test Statistic", str(np.round(self.test_results['test_stat'], 2))],
                ["P-Value", str(np.round(self.test_results['p_value'], 2))]
            ]

            result = tabulate(table, headers=["Statistic", "Values on grid"], tablefmt="pretty")
            result += "\nInitial Grid:" + str(self.test_results["init_grid"])
            result += "\nAdjusted z Grid:" + str(self.test_results["z_grid"])

        else:

            table = [
                ["P", f"{self.test_results['P']}", ""],
                ["T", f"{self.test_results['T']}", ""],
                ["L", f"{self.test_results['L']}", ""],
                ["c", f"{self.test_results['c']}", ""],
                ["Best z", f"{self.test_results['best_z']:.4f}", f"{self.test_results['best_z']:.4f}"],
                ["Test Statistic", str(np.round(self.test_results['test_stat'], 2)),
                 str(np.round(self.test_results['test_stat'][0, self.test_results['best_z_ind']], 2))],
                ["P-Value", str(np.round(self.test_results['p_value'], 2)),
                 str(np.round(self.test_results['p_value'][0, self.test_results['best_z_ind']], 2))]
            ]

            result = tabulate(table, headers=["Statistic", "Values on grid", "Value for optimal z"], tablefmt="pretty")
            result += "\nInitial Grid:" + str(self.test_results["init_grid"])
            result += "\nAdjusted z Grid:" + str(self.test_results["z_grid"])

        return result


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    from tqdm import tqdm

    from simulate.simulate_returns import simulate_returns

    P = 1000
    T = 1000
    L = 1
    sigma_p_sqrt = np.eye(P)
    mu_M = np.array([[0.07]])
    sigma_M = np.array([[0.14]])

    tests = []
    for i in tqdm(range(1000)):
        R, R_M, Beta, Alpha, _ = simulate_returns(p=P,
                                                  t=T,
                                                  l=L,
                                                  mu_M=mu_M,
                                                  sigma_M_sqrt=sigma_M,
                                                  sigma_p_sqrt=sigma_p_sqrt)

        tester = CKMS(z=[100])
        tester.test(R - Alpha, R_M)
        tests.append(tester.test_results['test_stat'])

    breakpoint()
