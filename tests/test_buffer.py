import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../.."))  # fix envs imports

from envs.common.buffer import RingTensorBuffer, BatchedRingTensorBuffer, RingBufferCounter, BatchedRingBufferWithSharedCounter
import torch


def test_tensor_buffer():
    b = RingTensorBuffer(buffer_len=4,shape=2)

    assert b.full() == False
    assert len(b) == 0

    b.add(torch.tensor([1, 2]))
    assert b.full() == False
    assert len(b) == 1
    assert torch.equal(b.get_latest(),torch.tensor([1, 2]))

    b.add(torch.tensor([3, 4]))
    assert b.full() == False
    assert len(b) == 2
    assert torch.equal(b.get_latest(),torch.tensor([3, 4]))
    assert torch.equal(b.get_latest_n(2),torch.tensor([[3, 4.],[1, 2.]]))

    b.add(torch.tensor([5, 6]))
    b.add(torch.tensor([7, 8]))
    b.add(torch.tensor([9, 10]))
    expected_storage = torch.tensor(
            [[ 9., 10.],
            [ 3.,  4.],
            [ 5.,  6.],
            [ 7.,  8.]])
    assert torch.equal(b.storage, expected_storage)
    assert b.full() == True
    assert len(b) == 4
    assert torch.equal(b.get_latest(),torch.tensor([9, 10]))
    assert torch.equal(b.get_latest_n(1),torch.tensor([[9, 10.]]))
    assert torch.equal(b.get_latest_n(3),torch.tensor([[9, 10.],[7, 8.],[5, 6.]]))
    assert b.step == 4
    assert torch.equal(b[0],torch.tensor([9, 10]))
    assert torch.equal(b[1],torch.tensor([7, 8]))

    b.reset() 
    assert b.step == -1

    c = BatchedRingTensorBuffer(buffer_len=3, batch_size=2, shape=2)
    c.add(torch.tensor([[1, 2],[3, 4]],dtype=torch.float32))

    assert c.batch_step.dtype == torch.int64
    assert c.batch_step[0] == 0
    assert torch.equal(c.storage[0], torch.tensor([[1, 2],[3, 4]]))

    c.add(torch.tensor([[5, 6],[7, 8]],dtype=torch.float32))

    assert torch.equal(c[0],torch.tensor([[5, 6],[7, 8]]))
    assert torch.equal(c[1],torch.tensor([[1, 2],[3, 4]]))

    c.add(torch.tensor([[9, 10],[11, 12]],dtype=torch.float32))


    c.add_and_fill_batch(torch.tensor([[13, 14],[15, 16]],dtype=torch.float32))

    assert torch.equal(torch.tensor([
            [[13., 14.],
            [15., 16.]],
            [[ 5.,  6.],
            [ 7.,  8.]],
            [[ 9., 10.],
            [11., 12.]]]),c.storage)

    c.reset_batch(0)
    c.add_and_fill_batch(torch.tensor([[17, 18],[19, 20]],dtype=torch.float32))

    assert torch.equal(torch.tensor([
            [[17., 18.],
            [15., 16.]],
            [[17., 18.],
            [19., 20.]],
            [[17., 18.],
            [11., 12.]]]),c.storage)

    assert torch.equal(c[1],torch.tensor([[17, 18],[15, 16]]))


def test_shared_buffer():
    # test shared buffer
    counter = RingBufferCounter(buffer_len=3, batch_size=2)
    b1 = BatchedRingBufferWithSharedCounter(counter, shape=1)
    b2 = BatchedRingBufferWithSharedCounter(counter, shape=(2,))

    b1_storage_0 = b1.storage.clone()
    b2_storage_0 = b2.storage.clone()
    assert b2.counter==b1.counter

    counter.increment_step()
    counter.warm_start_batch()
    assert counter.current_step == 0

    v1 = torch.ones(2,1)
    v2 = torch.ones(2,2)
    b1.add_and_fill_batch(v1)
    assert torch.equal(b1.storage,torch.ones(3,2,1))
    b2.add_and_fill_batch(v2)
    assert torch.equal(b2.storage,torch.ones(3,2,2))

    assert torch.equal(b2[0],v2)
    assert torch.equal(b2.storage[0],v2)

    counter.increment_step()
    counter.warm_start_batch()

    b1.add_and_fill_batch(v1*2)
    b2.add_and_fill_batch(v2*2)

    assert torch.equal(b2[0],v2*2)
    assert torch.equal(b2.storage,torch.stack((v2,v2*2,v2)))
    assert torch.equal(b2[:],torch.stack((v2*2,v2,v2)))

    assert torch.equal(b1[0],v1*2)
    assert torch.equal(b1.storage,torch.stack((v1,v1*2,v1)))
    assert torch.equal(b1[:],torch.stack((v1*2,v1,v1)))

    assert counter.current_step == 1

    counter.reset_batch(1)

    assert torch.equal(counter.batch_step,torch.tensor([ 3, -1]))

    counter.increment_step()
    counter.warm_start_batch()

    assert counter.current_step == 2


    b1.add_and_fill_batch(v1*3)
    b2.add_and_fill_batch(v2*3)

    m1 = torch.stack((v1,v1*2,v1))
    m1[:,1] = (v1*3)[1]
    m1[2] = v1*3
    assert torch.equal(b1.storage,m1)
    assert torch.equal(b1[:],m1[[2,1,0],:,:])

    m2 = torch.stack((v2,v2*2,v2))
    m2[:,1] = (v2*3)[1]
    m2[2] = v2*3
    assert torch.equal(b2.storage,m2)
    assert torch.equal(b2[:],m2[[2,1,0],:,:])


if __name__ == "__main__":
    test_tensor_buffer()
    test_shared_buffer()