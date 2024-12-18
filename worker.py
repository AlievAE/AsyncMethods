import numpy as np

class Worker:
    def __init__(self, loss_fn, gradient_fn, computation_time=0, compression_flag = 'none', k = 100, num_of_dim = 1):
        """
        Initialize Worker with loss function, gradient function and computation time.
        
        Args:
            loss_fn: Function that calculates loss
            gradient_fn: Function that calculates gradient
            computation_time: Either a non-negative number or a distribution to sample from
        """
        self.loss_fn = loss_fn
        self.gradient_fn = gradient_fn
        self.computation_time = computation_time
        self.current_x = None
        self.current_gradient = None
        #compression stuff
        self.dimensions = num_of_dim
        self.K = k
        self.compression_flag = compression_flag

    def compute_gradient(self, data, x):
        """
        Compute gradient at point x and store both x and the gradient.
        
        Args:
            x: Point at which to compute gradient
            data: Training data
            
        Returns:
            tuple: (gradient, computation_time) where gradient is computed at point x
                  and computation_time is sampled from the distribution
        """
        self.current_x = x
        self.current_gradient = self.gradient_fn(data, x)
        
        # Sample computation time
        if hasattr(self.computation_time, 'rvs'):  # Check if it's a distribution
            time = self.computation_time.rvs()
        else:  # Assume it's a number
            time = float(self.computation_time)
            
        return self.current_gradient, max(0, time)  # Ensure non-negative time

    def compute_loss(self, data, x):
        """
        Compute loss at point x.
        Args:
            x: Point at which to compute loss
            data: Training data
        Returns:
            loss: Computed loss at point x
        """
        return self.loss_fn(data, x)
    
    def randk_compress_vector(self, vector):
        idx = np.random.randint(0, self.dimensions, size=self.K)
        const = self.dimensions / self.K
        compressed_vector = np.zeros_like(vector)
        for i in idx:
            compressed_vector[i] += vector[i]
        return const * compressed_vector
    
    def topk_compress_vector(self, vector):
        abs_values = np.abs(vector)
        topk_indices = np.argsort(abs_values)[-self.K:]
        mask = np.zeros_like(vector, dtype=bool)
        mask[topk_indices] = True
        compressed_vector = np.where(mask, vector, 0)
        return compressed_vector

    @property
    def x(self):
        """Get the current x value."""
        return self.current_x

    @property
    def gradient(self):
        """Get the current gradient value."""
        if self.compression_flag == 'none':
            return self.current_gradient
        if self.compression_flag == 'randk':
            compressed = self.randk_compress_vector(self.gradient)
        if self.compression_flag == 'topk':
            compressed = self.topk_compress_vector(self.gradient)
        return compressed