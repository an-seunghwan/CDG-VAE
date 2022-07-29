#%%
import torch
import numpy as np
import scipy.linalg as slin
#%%
class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        input: W * W
        output: trace(exp(input))
        """
        # detach so we can cast to NumPy
        E = slin.expm(input.detach().numpy()) # calculate matrix exponential
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: gradient of input(= W * W) = 2 * W
        E: trace(exp(input)) saved from forward
        """
        E, = ctx.saved_tensors # dtype: tuple
        grad_input = E.t() * grad_output
        return grad_input

trace_expm = TraceExpm.apply
#%%
def main():
    W = torch.randn(20, 20, dtype=torch.float, requires_grad=True)
    input = W * W
    assert torch.autograd.gradcheck(trace_expm, input)

    W = torch.tensor([[1., 2.], [0.3, 0.4]], requires_grad=True)
    input = W * W

    """backward from torch.autograd.Function"""
    tre = trace_expm(input)
    f = tre - 2.
    f.backward()
    print('grad\n', W.grad.numpy())

    """naive backward calculation"""
    grad_naive = slin.expm(input.detach().numpy()).T * 2 * W.detach().numpy()
    print('grad_naive\n', grad_naive)

    """check validity of h(W) gradient function"""
    assert (np.abs(grad_naive - W.grad.numpy()) < 1e-8).all() 
#%%
if __name__ == '__main__':
    main()
#%%