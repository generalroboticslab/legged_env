import torch
from collections.abc import Iterable
from typing import Union
from scipy import signal

class _BaseRingBuffer:
    def __init__(self, buffer_len: int, shape: Union[Iterable,int], dtype=torch.float32, device='cpu'):
        """
        Base class for circular tensor buffers.

        Args:
            buffer_len: Maximum number of tensors the buffer can hold.
            shape: Shape of the tensors to be stored (e.g., (3, 4), (6,)).
            dtype: The data type of the tensors in the buffer (default: torch.float32).
        """
        self.buffer_len = buffer_len
        if not isinstance(shape, Iterable):
            shape = (shape,)
        # self.storage acts as a circular buffer, the element at self._step is the newest element
        self.storage = torch.zeros((buffer_len,*shape), dtype=dtype,device=device)
        self.step = -1 # current step
        self.max_step = self.buffer_len-1

        self.index_mapping = torch.stack([
            torch.arange(step,step-self.buffer_len,-1,device=self.storage.device)
            for step in range(self.buffer_len)
        ])%self.buffer_len
        # print(self.index_mapping)

    def add(self, tensor: torch.Tensor):
        """
        Appends a tensor to the buffer, wrapping around if full.

        Args:
            tensor: The tensor to add to the buffer. Must have the correct shape.
        """
        self.step+=1
        self.storage[self.step%self.buffer_len] = tensor
        
    def __getitem__(self, index: Union[int, slice, Iterable]) -> torch.Tensor:
        """
        Retrieves a tensor (reference) from the buffer by index, newest to oldest.
        NOTE: the tensor is a reference, and could change in the buffer, use torch.clone() to get a copy.
        For example:
          index=0 returns the latest tensor added to the buffer.
          index=-1 returns the oldest tensor added to the buffer.
          index=slice(3) returns the last three tensors added to the buffer, newest to oldest.
        Args:
            index: The index of the tensor to retrieve.

        Returns:
            The tensor at the specified index.
        """
        current_step = self.step
        # if current_step==-1:
        #     raise IndexError("Buffer is empty")  # Handle empty buffer case
        if isinstance(index, slice):
            # start = index.start or 0
            # stop = index.stop or self.buffer_len
            # step = index.step or 1
            start, stop, step = index.indices(self.buffer_len)  # Concise slice handling
            indices = torch.arange(current_step-start, current_step-stop, -step) % self.buffer_len
            return self.storage[indices]
        elif isinstance(index, Iterable):
            indices = (current_step - torch.as_tensor(index)) % self.buffer_len
            return self.storage[indices]
        else:
            return self.storage[(current_step-index) % self.buffer_len]
        
    def get_latest(self):
        """
        Returns the last tensor in the buffer
        NOTE: the tensor is a reference, and could change in the buffer, use torch.clone() to get a copy.
        """
        return self.storage[self.step%self.buffer_len]
    
    def get_latest_n(self, n:int, offset:int=0) -> torch.Tensor:
        """
        Returns the last n tensors in the buffer, newest to oldest
        NOTE: the tensor is a reference, and could change in the buffer, use torch.clone() to get a copy.
        """
        assert n <= self.buffer_len and n>0
        # return self.storage[torch.arange(self.step,self.step-n,-1,device=self.storage.device)%self.buffer_len]
        return self.storage[self.index_mapping[(self.step-offset)%self.buffer_len,:n]]
    
    def get_all(self) -> torch.Tensor:
        """
        Returns all tensors in the buffer
        """
        
        return self.storage[self.index_mapping[self.step%self.buffer_len]]
    

class RingTensorBuffer(_BaseRingBuffer):    
    def __len__(self) -> int:
        """
        Returns the current number of elements in the buffer.

        Returns:
            The number of elements in the buffer.
        """
        return min(self.step+1, self.buffer_len)

    def full(self) -> bool:
        """
        Checks if the buffer is full.

        Returns:
            True if the buffer is full, False otherwise.
        """
        return self.step>=self.max_step
    
    def reset(self):
        """Resets the buffer current step to -1"""
        self.step = -1
        self.storage.zero_()


class RingTensorFilterBuffer:
    def __init__(self,shape,fs:float=200,filter_order:int=4,cut_off_frequency:float=20,dtype=torch.float,device='cpu'):

        
        self.filter_order = filter_order
        self.buf_raw = RingTensorBuffer(buffer_len=filter_order+1, shape=shape,dtype=dtype,device=device) # x, input
        self.buf_filt = RingTensorBuffer(buffer_len=filter_order+1, shape=shape,dtype=dtype,device=device) # y, filtered
        
        self.b, self.a = signal.butter(N=filter_order, Wn=cut_off_frequency, btype='low', analog=False,fs=fs)
        self.a = torch.tensor(self.a,dtype=dtype,device=device)
        self.b = torch.tensor(self.b,dtype=dtype,device=device)
        self.b = self.b.reshape(-1, *([1] * len(self.buf_raw.storage.shape[1:])))
        self.a = self.a.reshape(-1, *([1] * len(self.buf_raw.storage.shape[1:])))[1:filter_order+1]
        # self.b = self.b[:,np.newaxis]
        # self.a = self.a[1:filter_order+1,np.newaxis]
        
    def add(self,sample:torch.Tensor):
        self.buf_raw.add(sample) 
        filtered_sample = (self.b*self.buf_raw[:]).sum(axis=0) - \
            (self.a*self.buf_filt[0:self.filter_order]).sum(axis=0)
        self.buf_filt.add(filtered_sample)
        return filtered_sample

    def fill(self,sample:torch.Tensor):
        self.buf_raw.storage[:] = sample
        self.buf_filt.storage[:] = sample

    def get_latest_filtered(self):
        return self.buf_filt[0].clone()
    
    def get_latest_raw(self):
        return self.buf_raw[0].clone()
    
    def reset(self):
        self.buf_raw.reset()
        self.buf_filt.reset()


class BatchedRingTensorBuffer(_BaseRingBuffer):
    def __init__(self, buffer_len: int, batch_size: int, shape: Union[Iterable,int], dtype=torch.float32, device='cpu'):
        """
        Initializes a circular Tensor Batch Buffer. storage is of shape (buffer_len, batch_size, *shape)

        Args:
            shape: Shape of the tensors to be stored (e.g., (3, 4), (6,)).
            batch_size: Number of tensors in a batch.
            buffer_len: Maximum number of tensors the buffer can hold.
            dtype: The data type of the tensors in the buffer (default: torch.float32).
        """
        self.batch_step = torch.zeros(batch_size, dtype=torch.int64, device=device).fill_(-1)
        if not isinstance(shape, Iterable):
            shape = (shape,)
        super().__init__(buffer_len=buffer_len, shape=(batch_size, *shape), dtype=dtype, device=device)

    def batch_full(self) -> torch.Tensor:
        """Checks if the batch buffer is full batch-wise"""
        return self.batch_step >= self.max_step
        
    def batch_not_full(self) -> torch.Tensor:
        """Checks if the batch buffer is not full batch-wise"""
        return self.batch_step < self.max_step      
    
    def add(self, tensor: torch.Tensor):
        """
        Appends a tensor to the buffer, wrapping around if full.

        Args:
            tensor: The tensor to add to the buffer. Must have the correct shape.
        """
        self.step+=1
        # Update the batch step, increment by 1 and wrap around if needed
        self.batch_step.add_(1)
        self.storage[self.step%self.buffer_len] = tensor

    def add_and_fill_batch(self, tensor: torch.Tensor):
        """Adds a tensor to the buffer if the batch is not full"""
        self.step+=1
        self.batch_step.add_(1)
        batch_idx = self.batch_step < self.max_step
        self.batch_step[batch_idx] = self.max_step
        # add
        self.storage[self.step%self.buffer_len] = tensor
        # fill if batch is not full
        self.storage[:,batch_idx] = tensor[batch_idx]

    def reset_batch(self, batch_idx: Union[int,torch.Tensor]):
        """Resets the batch step for a given batch index"""
        self.batch_step[batch_idx] = -1
        self.storage[:,batch_idx].fill_(0)
    
    def reset(self):
        self.step = -1
        self.batch_step.fill_(-1)
        self.storage.zero_()

class RingBufferCounter:
    def __init__(self, buffer_len: int, batch_size: int,device='cpu'):
        self.batch_size = batch_size
        self.batch_step = torch.zeros(batch_size, dtype=torch.int64, device=device).fill_(-1)
        self.step = -1 # current step
        self.buffer_len = buffer_len
        self.max_step = buffer_len-1

    def increment_step(self):
        """Increments the step counter"""
        self.step+=1
        self.batch_step.add_(1)
        self.current_step = self.step%self.buffer_len

    def warm_start_batch(self):
        """Fills the batch step with max_step if it is not full, return the batch index filled"""
        self.batch_idx_to_fill = self.batch_step < self.max_step
        self.batch_step[self.batch_idx_to_fill] = self.max_step

    def reset_batch(self, batch_idx: Union[int,torch.Tensor]):
        """Resets the batch step for a given batch index"""
        self.batch_step[batch_idx] = -1
    
    def reset(self):
        self.step = -1
        self.batch_step.fill_(-1)

class BatchedRingBufferWithSharedCounter(_BaseRingBuffer):
    
    def __init__(self, counter: RingBufferCounter, shape: Union[Iterable,int], dtype=torch.float32, device='cpu'):
        """
        Initializes a TensorBatchBuffer that may share a buffer counter with other TensorBatchBuffers.

        Args:
            counter: The buffer counter that keeps track of the current step.
            shape: Shape of the tensors to be stored (e.g., (3, 4), (6,)).
            dtype: The data type of the tensors in the buffer (default: torch.float32).
        """
        self.counter = counter
        self.buffer_len = counter.buffer_len
        self.batch_size = counter.batch_size
        if not isinstance(shape, Iterable):
            shape = (shape,)
        # self.storage acts as a circular buffer, the element at self._step is the newest element
        self.storage = torch.zeros((self.buffer_len,self.batch_size,*shape), dtype=dtype,device=device)
    
    @property
    def step(self):
        return self.counter.step

    def add(self, tensor: torch.Tensor):
        """
        Appends a tensor to the buffer, wrapping around if full.
        remember to increment the counter before adding

        Args:
            tensor: The tensor to add to the buffer. Must have the correct shape.
        """
        self.storage[self.counter.current_step] = tensor

    def add_and_fill_batch(self, tensor: torch.Tensor):
        """
        Appends a tensor to the buffer, wrapping around if full.
        remember to increment the counter before adding by:
            counter.increment_step() # Update the batch step, increment by 1
            counter.warm_start_batch() # Fill the batch step with max_step if it is not full
        Args:
            tensor: The tensor to add to the buffer. Must have the correct shape.
        """
        # update the storage
        self.storage[self.counter.step%self.buffer_len] = tensor
        self.storage[:,self.counter.batch_idx_to_fill] = tensor[self.counter.batch_idx_to_fill]