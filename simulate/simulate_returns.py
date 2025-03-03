import numpy as np

def simulate_returns(p              : int,
                     t              : int,
                     l              : int,
                     sigma_p_sqrt   : np.ndarray,
                     mu_f           : np.ndarray,
                     sigma_f_sqrt   : np.ndarray,
                     mu_alpha       : float = 0,
                     beta_bound     : float = 3.0,
                     omega_alpha    : np.ndarray = None,
                     target_lambda  : float = None,
                     sigma_p_inv    : np.ndarray = None
                     ):
    
    """
    simulate_returns This functions simulates the returns of the stocks as a linear transformation of the factors, up to some noise.
    
    Parameters:
    -----------
    p : int
        The number of stocks.
        
    t : int
        The number of observations.
        
    l : int
        The number of factors.
        
    sigma_p_sqrt : np.ndarray
    
    mu_f : np.ndarray
        Mean factor returns, of shape (l,1).
        
    sigma_f_sqrt : np.ndarray
        The square root of the covariance matrix of the factor returns, of shape (l,l).
        
    mu_alpha : np.ndarray, default = 0
        The mean of the alphas, of shape (p,1).
        
    omega_alpha_sqrt : np.ndarray, default = None
        The sqrt of the covariance matrix of the alphas, of shape (p,p).
        
    target_lambda : float, default = None
        The target lambda that we want the alpha's to achieve. Where lambda is defined as in the Gibbons, Ross, Shanken paper.
        
    sigma_p_inv : np.ndarray, default = None
        The inverse of the covariance matrix of the stock returns. Computing it once only can save a lot of time when creating many simulations.

    Returns
    -------
    _type_
        _description_
    """
    
    #* Handle some useful cases + some errors check
    assert not((target_lambda is not None) and (sigma_p_inv is None)), "If you want to match the lambda, you need to provide the inverse of the covariance matrix of the stock returns."
    
    if omega_alpha is None:
        omega_alpha = np.eye(p)


    #*1 : Simulate the factors
    r_f = np.random.normal(mu_f,1,size = (l,t))
    r_f = sigma_f_sqrt @ r_f
    
    #* 2. Simulate the betas
    beta = np.random.uniform(-beta_bound, beta_bound, size = (p,l))
    
    #* 3. Simulate the alphas
    alpha = omega_alpha@np.random.normal(mu_alpha, 1, size = (p,1))
    
    #* 4 Match the lambda if required
    scaling = None
    if target_lambda is not None:
        
        if isinstance(mu_f, float):
            theta_p2 = (mu_f/sigma_f_sqrt)**2
            
        else:
            theta_p2 = mu_f.T@np.linalg.inv(sigma_f_sqrt@sigma_f_sqrt)@mu_f
            
            
        lam_hat = (t/(1+theta_p2)*alpha.T@sigma_p_inv@alpha)
        scaling = np.sqrt(target_lambda/lam_hat)
        alpha = scaling*alpha
        
    #* 5. Simulate the stock returns
    eps = np.random.normal(0,1,size = (p,t))
    r = alpha + beta@r_f + sigma_p_sqrt@eps

    return r, r_f, beta, alpha, scaling
    
if __name__ == '__main__':
    
    simulate_returns(p = 10,
                     t = 1000,
                     l = 3,
                     sigma_p_sqrt = np.eye(10),
                     mu_f = np.zeros(3).reshape(-1,1),
                     sigma_f_sqrt = np.eye(3))