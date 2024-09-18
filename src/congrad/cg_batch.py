from warnings import warn

from .monitoring import Monitor, DefaultMonitor

class cg_batch_generic:
    """Generic preconditioned batched CG.  Relies on a backend."""

    def __init__(self, backend):
        self.backend = backend

    def __call__(self, A, b, P=None, x0=None, rtol=1e-3, atol=0, maxiter=1000, warn_unconverged=True, monitor=None, flexible=False):
        """Solves a batch of SPD linear equations of the form Ax=b, with initial guess x0.

        This function makes no assumptions about its inputs except that vectors work as expected with addition, subtraction, pointwise multiplication and division, and backend functions.
        
        :param A: A function that represents batched matvecs by A
        :type A: function
        :param b: A batched vector
        :type b: vector
        :param P: A function that represents matvecs by a preconditioner (default: None)
        :type P: function, optional
        :param x0: A batched vector (default: None)
        :type x0: vector, optional
        :param rtol: Stop when the residual norm is < rtol * |b| (default: 1e-3)
        :type rtol: float, optional
        :param atol: Stop when the residual norm is < atol (default: 0)
        :type atol: float, optional
        :param maxiter: Stop after this many iterations, or None to converge no matter how long it takes (default: 1000)
        :type maxiter: int, optional
        :param warn_unconverged: Raise a warning if iteration is stopped before convergence (default: True)
        :type warn_unconverged: bool, optional
        :param monitor: None or False to not use a monitor, True to use the default monitor (which prints a convergence message every twenty iterations), or a monitor instance (default: None).
        :type monitor: Monitor, optional
        :param flexible: Use "flexible CG" (Polak-Ribère rather than Fletcher-Reeves) -- if you don't know what this is then you don't need it (default: False)
        :type flexible: bool, optional
        
        :return: A tuple x, info where x is the (batched) solution and info is a dict containing:
           - niter: The number of iterations (calls to A_bmm) required.
           - converged: Whether or not the solution has converged.
           - residual: The norm of each residual as computed by the CG algorithm.
        :rtype: tuple
        """

        if P is None:
            P = lambda x: x
        if x0 is None:
            x0 = P(b)
        infinite_iters = (maxiter is None)

        if not isinstance(monitor, Monitor):
            if monitor:
                monitor = DefaultMonitor()
        
        assert b.shape == x0.shape
        assert rtol > 0 or atol > 0
        
        stopping_bound = self.backend.max_vector_scalar(rtol * self.backend.norm(b), atol)

        if monitor:
            monitor.setup(stopping_bound, maxiter)
        
        x = x0
        r = b - A(x)
        residual_norm = self.backend.norm(r)
        converged_per_vector = (residual_norm < stopping_bound)
        z = P(r)
        p = z
        iter_count = 1 # iter_count should count the number of calls to A

        if monitor:
            monitor.step(iter_count, self.backend.presentable_norm(self.backend.norm(r)), x, False)
        
        converged = False

        while infinite_iters or (iter_count < maxiter):
            Ap = A(p)
            rz = self.backend.dot(r, z)
            alpha = rz / self.backend.dot(p, Ap)
            alpha = self.backend.zero_where(alpha, converged_per_vector)
            new_x = x + alpha * p
            new_r = r - alpha * Ap
            residual_norm = self.backend.norm(new_r)
            converged_per_vector = (residual_norm < stopping_bound)
            if self.backend.all_true(converged_per_vector):
                converged = True
                break
            new_z = P(new_r)
            if flexible:
                beta_numerator = self.backend.dot(new_r, new_z - z)
            else:
                beta_numerator = self.backend.dot(new_r, new_z)
            beta = beta_numerator / rz
            new_p = new_z + beta * p

            x, r, z, p = new_x, new_r, new_z, new_p
            
            iter_count += 1
            
            if monitor:
                monitor.step(iter_count, self.backend.presentable_norm(residual_norm), x, False)

        if warn_unconverged and not converged:
            warn("Reached maximum iterations without converging; returning current best approximate solution.", RuntimeWarning)

        if converged and monitor:
            monitor.step(iter_count, self.backend.presentable_norm(residual_norm), x, True)

        info = {
            "niter": iter_count,
            "converged": converged,
            "residual": residual_norm
        }

        return new_x, info
