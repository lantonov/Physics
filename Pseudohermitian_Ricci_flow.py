import jax.numpy as jnp
import jax
from jax import config as jax_config
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# Enable double precision
jax_config.update("jax_enable_x64", True)

class PseudohermitianRicciFlow:
    """
    Implementation of the pseudohermitian Ricci flow on 3-Sasakian S^7
    """
    
    def __init__(self, base_model):
        """
        Initialize with a validated ThreeSasakianS7 model
        
        Parameters
        ----------
        base_model : ThreeSasakianS7
            The validated base model containing F1, F2, F3 matrices
        """
        self.base_model = base_model
        self.n = 2  # Dimension parameter for S^7 (quaternionic dimension)
        
    def compute_metric_tensor(self, x):
        """
        Compute the metric tensor at point x on S^7
        
        For the standard 3-Sasakian structure, this is the restriction
        of the Euclidean metric to the tangent space of S^7
        
        Parameters
        ----------
        x : array_like, shape (8,)
            Point on S^7 (assumed normalized)
            
        Returns
        -------
        g : ndarray, shape (8, 8)
            Metric tensor (tangent space projection)
        """
        # Normalize to ensure we're on S^7
        x = x / jnp.linalg.norm(x)
        
        # The metric on S^7 is the restriction of Euclidean metric
        # to the tangent space T_x S^7 = {v : ⟨v, x⟩ = 0}
        I = jnp.eye(8)
        radial_projection = jnp.outer(x, x)  # Projects onto radial direction
        
        # Tangent space metric = I - x⊗x (projects away from radial)
        g = I - radial_projection
        
        return g
    
    def compute_tanaka_webster_ricci(self, x):
        """
        Compute the Tanaka-Webster Ricci tensor from the horizontal curvatures
        
        Parameters
        ----------
        x : array_like, shape (8,)
            Point on S^7
            
        Returns
        -------
        ric_tw : ndarray, shape (8, 8)
            Tanaka-Webster Ricci tensor
        """
        x = x / jnp.linalg.norm(x)
        
        # Get the horizontal curvatures from each quaternionic channel
        k1, k2, k3, (k1_planes), (k2_planes), (k3_planes) = \
            self.base_model.extract_horizontal_curvatures(
                self.base_model.F1, self.base_model.F2, self.base_model.F3, x
            )
        
        # Build horizontal distribution basis
        a1, a2, a3 = (self.base_model.F1 @ x, 
                      self.base_model.F2 @ x, 
                      self.base_model.F3 @ x)
        
        # Gram-Schmidt to get orthonormal Reeb vectors
        def gram_schmidt(v, basis):
            w = v - sum((float(jnp.dot(b, v))) * b for b in basis)
            norm = float(jnp.linalg.norm(w))
            return jnp.where(norm > 1e-12, w/norm, jnp.zeros_like(w))
        
        xi1 = a1 / jnp.linalg.norm(a1)
        xi2 = gram_schmidt(a2, [xi1])
        xi3 = gram_schmidt(a3, [xi1, xi2])
        
        # Vertical space [x, ξ₁, ξ₂, ξ₃]
        vertical_basis = jnp.stack([x, xi1, xi2, xi3], axis=1)
        
        # Get horizontal basis (4-dimensional)
        Q, _ = jnp.linalg.qr(vertical_basis, mode='complete')
        H = Q[:, 4:]  # Horizontal basis
        
        # For each quaternionic channel, construct the Ricci contribution
        ric_tw = jnp.zeros((8, 8))
        
        for F, k_sum in [(self.base_model.F1, k1), 
                         (self.base_model.F2, k2), 
                         (self.base_model.F3, k3)]:
            # Restrict F to horizontal distribution
            F_h = H.T @ F @ H  # 4×4 skew-symmetric
            
            # The Ricci contribution from this channel
            # Projects the horizontal curvature back to the full tangent space
            F_full = H @ F_h @ H.T
            
            # Add the symmetric part (Ricci is symmetric)
            ric_contribution = 0.5 * (F_full @ F_full.T + F_full.T @ F_full)
            
            ric_tw += ric_contribution
        
        return ric_tw
    
    def compute_contact_feedback_term(self, x, kappa=1.0):
        """
        Compute the contact-preserving feedback term 2κα₁ ⊗ α₁
        
        Parameters
        ----------
        x : array_like, shape (8,)
            Point on S^7
        kappa : float
            Feedback strength parameter
            
        Returns
        -------
        feedback : ndarray, shape (8, 8)
            Contact feedback tensor
        """
        x = x / jnp.linalg.norm(x)
        
        # Primary contact form vector
        alpha1 = self.base_model.F1 @ x
        alpha1_normalized = alpha1 / jnp.linalg.norm(alpha1)
        
        # Feedback term: 2κ α₁ ⊗ α₁
        feedback = 2 * kappa * jnp.outer(alpha1_normalized, alpha1_normalized)
        
        return feedback
    
    def flow_rhs(self, t, g_flat, x):
        """
        Right-hand side of the flow equation: ∂g/∂t = F(g, t)
        
        Parameters
        ----------
        t : float
            Time parameter
        g_flat : array_like, shape (64,)
            Flattened metric tensor
        x : array_like, shape (8,)
            Point on S^7 (fixed during integration)
            
        Returns
        -------
        dgdt_flat : ndarray, shape (64,)
            Time derivative of flattened metric
        """
        # Reshape to matrix form
        g = g_flat.reshape(8, 8)
        
        # Compute the three terms in the flow equation
        ric_tw = self.compute_tanaka_webster_ricci(x)
        feedback = self.compute_contact_feedback_term(x)
        
        # Flow equation: ∂g/∂t = -2Ric^∇ - 12g + feedback
        dgdt = -2 * ric_tw - 12 * g + feedback
        
        return dgdt.flatten()
    
    def evolve_metric(self, x, T_final=1.0, num_steps=100, kappa=1.0):
        """
        Evolve the metric under pseudohermitian Ricci flow
        
        Parameters
        ----------
        x : array_like, shape (8,)
            Point on S^7 where to evolve the metric
        T_final : float
            Final time
        num_steps : int
            Number of time steps
        kappa : float
            Contact feedback parameter
            
        Returns
        -------
        times : ndarray
            Time points
        metrics : ndarray, shape (num_steps+1, 8, 8)
            Metric evolution
        traces : ndarray
            Ricci trace evolution
        """
        from scipy.integrate import solve_ivp
        
        x = x / jnp.linalg.norm(x)
        
        # Initial metric (standard S^7 metric)
        g0 = self.compute_metric_tensor(x)
        
        # Time span
        times = jnp.linspace(0, T_final, num_steps + 1)
        
        # Set up the ODE with fixed x and kappa
        def rhs_fixed(t, g_flat):
            return self.flow_rhs(t, g_flat, x)
        
        # Solve the ODE
        solution = solve_ivp(
            rhs_fixed, 
            [0, T_final], 
            g0.flatten(),
            t_eval=times,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Reshape solutions back to matrix form
        metrics = solution.y.T.reshape(len(times), 8, 8)
        
        # Compute Ricci trace evolution
        traces = []
        for i, t in enumerate(times):
            g_t = metrics[i]
            # For trace computation, we need the current Ricci tensor
            # This is approximated as the horizontal curvature sum
            k1, k2, k3, _, _, _ = self.base_model.extract_horizontal_curvatures(
                self.base_model.F1, self.base_model.F2, self.base_model.F3, x
            )
            traces.append(k1 + k2 + k3)
        
        return times, metrics, jnp.array(traces)
    
    def analyze_flow_evolution(self, x, T_final=1.0, num_steps=100):
        """
        Comprehensive analysis of the flow evolution
        
        Parameters
        ----------
        x : array_like, shape (8,)
            Point on S^7
        T_final : float
            Evolution time
        num_steps : int
            Number of steps
            
        Returns
        -------
        analysis : dict
            Complete flow analysis
        """
        times, metrics, traces = self.evolve_metric(x, T_final, num_steps)
        
        # Compute various geometric quantities over time
        analysis = {
            'times': times,
            'metrics': metrics,
            'ricci_traces': traces,
            'metric_norms': [],
            'determinants': [],
            'condition_numbers': []
        }
        
        for g in metrics:
            analysis['metric_norms'].append(float(jnp.linalg.norm(g, 'fro')))
            try:
                det_g = jnp.linalg.det(g)
                analysis['determinants'].append(float(det_g))
                
                # Condition number (for numerical stability)
                cond_g = jnp.linalg.cond(g)
                analysis['condition_numbers'].append(float(cond_g))
            except:
                analysis['determinants'].append(float('nan'))
                analysis['condition_numbers'].append(float('inf'))
        
        return analysis
    
    def plot_flow_evolution(self, analysis, save_path=None):
        """
        Plot the evolution of various quantities during the flow
        
        Parameters
        ----------
        analysis : dict
            Output from analyze_flow_evolution
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        times = analysis['times']
        
        # Plot 1: Ricci trace evolution
        axes[0, 0].plot(times, analysis['ricci_traces'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Ricci Trace')
        axes[0, 0].set_title('Ricci Trace Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Metric norm evolution
        axes[0, 1].plot(times, analysis['metric_norms'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('||g||_F')
        axes[0, 1].set_title('Metric Frobenius Norm')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Determinant evolution
        axes[1, 0].semilogy(times, np.abs(analysis['determinants']), 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('|det(g)|')
        axes[1, 0].set_title('Metric Determinant (log scale)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Condition number evolution
        axes[1, 1].semilogy(times, analysis['condition_numbers'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('cond(g)')
        axes[1, 1].set_title('Condition Number (log scale)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ==================== USAGE EXAMPLE ====================

if __name__ == '__main__':
    # Import the validated model
    try:
        from Quaternionic_contact_flow_GS1 import ThreeSasakianS7
    except ImportError:
        print("Could not import ThreeSasakianS7. Please ensure the file is available.")
        exit(1)
    
    # Create base model and flow solver
    base_model = ThreeSasakianS7()
    flow_solver = PseudohermitianRicciFlow(base_model)
    
    # Test points
    test_points = {
        'Generic': jnp.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0]) / jnp.linalg.norm(jnp.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0])),
        'Symmetric': jnp.ones(8) / jnp.linalg.norm(jnp.ones(8))
    }
    
    print("Pseudohermitian Ricci Flow Simulation")
    print("=" * 50)
    
    for name, x in test_points.items():
        print(f"\nAnalyzing flow at {name} point: {x}")
        
        # Run flow evolution
        analysis = flow_solver.analyze_flow_evolution(
            x, T_final=0.1, num_steps=50  # Short time for initial testing
        )
        
        # Print summary
        print(f"Initial Ricci trace: {analysis['ricci_traces'][0]:.6f}")
        print(f"Final Ricci trace: {analysis['ricci_traces'][-1]:.6f}")
        print(f"Initial metric norm: {analysis['metric_norms'][0]:.6f}")
        print(f"Final metric norm: {analysis['metric_norms'][-1]:.6f}")
        
        # Plot evolution
        flow_solver.plot_flow_evolution(analysis, f'flow_evolution_{name.lower()}.png')
    
    print("\nFlow simulation completed!")