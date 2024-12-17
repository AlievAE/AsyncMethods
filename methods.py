from worker import Worker

import numpy as np
import heapq
from dataclasses import dataclass

@dataclass
class WorkerState:
    worker_id: int
    finish_time: float
    
    def __lt__(self, other):
        return self.finish_time < other.finish_time

class BaseGD:
    def __init__(self, initial_x, data, time_distributions, loss_fn, gradient_fns, learning_rate=0.1):
        """Base class for gradient descent implementations"""
        assert len(gradient_fns) == len(time_distributions), f"Number of gradient functions ({len(gradient_fns)}) \
              must match number of time distributions ({len(time_distributions)})"
        
        self.workers = [
            Worker(loss_fn, gradient_fns[i], time_distributions[i])
            for i in range(len(gradient_fns))
        ]
        self.current_x = initial_x
        self.learning_rate = learning_rate
        self.loss_history = []
        self.computation_times = []
        self.current_time = 0
        self.x_history = [initial_x]
        self.data = data
        self.loss_fn = loss_fn

    def _update_loss_history(self):
        """Update loss history with current total loss"""
        total_loss = self.loss_fn(self.data, self.current_x)
        self.loss_history.append(total_loss)

class MinibatchSGD(BaseGD):
    def run_steps(self, num_steps):
        for _ in range(num_steps):
            gradients_and_times = [
                worker.compute_gradient(self.data, self.current_x)
                for worker in self.workers
            ]
            
            gradients, times = zip(*gradients_and_times)
            avg_gradient = np.mean(gradients, axis=0)
            
            self.current_x = self.current_x - self.learning_rate * avg_gradient
            self.current_time += np.max(times)
            self.computation_times.append(self.current_time)
            self._update_loss_history()
            self.x_history.append(self.current_x)
        
        return self.current_x, self.loss_history, self.computation_times, self.x_history

class AsynchronousGD(BaseGD):
    def run_steps(self, num_steps):        
        heap = []
        for i, worker in enumerate(self.workers):
            gradient, time = worker.compute_gradient(self.data, self.current_x)
            heapq.heappush(heap, WorkerState(i, time + self.current_time))
        
        for _ in range(num_steps):
            worker_state = heapq.heappop(heap)
            worker = self.workers[worker_state.worker_id]
            self.current_time = worker_state.finish_time
            
            self.current_x = self.current_x - self.learning_rate * worker.gradient
            
            gradient, compute_time = worker.compute_gradient(self.data, self.current_x)
            heapq.heappush(heap, WorkerState(worker_state.worker_id, self.current_time + compute_time))
            
            self.computation_times.append(self.current_time)
            self._update_loss_history()
            self.x_history.append(self.current_x)
        
        return self.current_x, self.loss_history, self.computation_times, self.x_history

class RennalaSGD(BaseGD):
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def run_steps(self, num_steps):
        for _ in range(num_steps):
            heap = []
            for i, worker in enumerate(self.workers):
                gradient, time = worker.compute_gradient(self.data, self.current_x)
                heapq.heappush(heap, WorkerState(i, time + self.current_time))
            current_gradient = 0
            
            for s in range(self.batch_size):
                state = heapq.heappop(heap)
                worker = self.workers[state.worker_id]
                
                self.current_time = state.finish_time
                
                current_gradient += worker.gradient
                
                _, compute_time = worker.compute_gradient(self.data, self.current_x)
                state.finish_time = self.current_time + compute_time
                heapq.heappush(heap, state)
            
            self.current_x = self.current_x - self.learning_rate * current_gradient / self.batch_size
            self.computation_times.append(self.current_time)
            self._update_loss_history()
            self.x_history.append(self.current_x)
        
        return self.current_x, self.loss_history, self.computation_times, self.x_history