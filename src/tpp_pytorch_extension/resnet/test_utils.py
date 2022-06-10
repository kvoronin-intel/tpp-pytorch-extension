import torch
import numpy as np

def compare_weight_grads( opt_weight_grad, ref_weight_grad, label_string):
    opt_tensor_grad = opt_weight_grad.to(torch.float)
    ref_tensor_grad = ref_weight_grad.to(torch.float)

    if opt_tensor_grad.dim() == 6:
        size = [opt_tensor_grad.size(0)*opt_tensor_grad.size(5), opt_tensor_grad.size(1)*opt_tensor_grad.size(4), opt_tensor_grad.size(2), opt_tensor_grad.size(3)]
        opt_tensor_gradp = opt_tensor_grad.permute([0,5,1,4,2,3]).contiguous().view(size)
        #print("opt_tensor_gradp shape = ", opt_tensor_gradp.shape)
        opt_tensor_grad_unblocked = opt_tensor_gradp
    elif opt_tensor_grad.dim() == 7:
        size = [opt_tensor_grad.size(0)*opt_tensor_grad.size(5), opt_tensor_grad.size(1)*opt_tensor_grad.size(4)*opt_tensor_grad.size(6), opt_tensor_grad.size(2), opt_tensor_grad.size(3)]
        opt_tensor_gradp = opt_tensor_grad.permute([0,5,1,4,6,2,3]).contiguous().view(size)
        #print("opt_tensor_gradp shape = ", opt_tensor_gradp.shape)
        opt_tensor_grad_unblocked = opt_tensor_gradp
    else:
        opt_tensor_grad_unblocked = opt_tensor_grad
    #print("opt_tensor_grad_unblocked shape = ", opt_tensor_grad_unblocked.shape)

    print(label_string + " Wgrad Allclose: ", ref_tensor_grad.allclose(opt_tensor_grad_unblocked, rtol=1e-5, atol=1e-6))

    #print("opt_tensor_grad_unblocked shape = ", opt_tensor_grad_unblocked.shape)
    wgrad_rel_norm_diff = (opt_tensor_grad_unblocked - ref_tensor_grad).norm(2) / ref_tensor_grad.norm(2)
    if wgrad_rel_norm_diff > 1.0e-5:
        print("warning, wgrad_rel_norm diff is too large, ", wgrad_rel_norm_diff)
    #for i in range(10):
    #    print("i opt_tensor_grad_unblocked ref_tensor_grad = ", i, opt_tensor_grad_unblocked.view(-1)[i].item(), ref_tensor_grad.view(-1)[i].item())

    print("(opt_tensor_grad.permuted - ref_tensor_grad).abs().norm(inf)               = ", (opt_tensor_grad_unblocked - ref_tensor_grad).norm(p=float('inf')))
    print("(opt_tensor_grad.permuted - ref_tensor_grad).norm(2) / torch.w.grad        = ", (opt_tensor_grad_unblocked - ref_tensor_grad).norm(2) / ref_tensor_grad.norm(2))

    for i in range(10):
        print("i opt_tensor_grad ref_tensor_grad = ", i, opt_tensor_grad_unblocked.view(-1)[i].item(), ref_tensor_grad.view(-1)[i].item())

    return

# For zero_rim_for_opt = True, HW must be third-fourth dimension (as in NCHW or NCHWC):
def compare_padded_tensors(opt_grad, ref_grad, label_string, nonpadded_width = 0, opt_padding = None, zero_rim_for_opt = False):

    ref_tensor = ref_grad.to(torch.float)

    if opt_padding != None and zero_rim_for_opt == True:
        print("debug: Zeroing the rim of the padded (opt) tensor on the Python side (assuming tensor has layout **HW*...*")
        nchwx_shape = opt_grad.shape
        opt_tensor_zeroed_rim = torch.zeros_like(opt_grad)
        print("debug: nchw shape = ", nchwx_shape)
        H = nchwx_shape[2] - opt_padding[0] - opt_padding[1]
        W = nchwx_shape[3] - opt_padding[2] - opt_padding[3]
        if W != nonpadded_width:
            print("Inconsistent parameters: nonpadded_width, shape-inferred W = ", nonpadded_width, W)
            exit()
        print("range = ", 'full', ' ', 'full', opt_padding[0], H + opt_padding[0], opt_padding[2], W + opt_padding[2])
        opt_tensor_zeroed_rim[:,:,opt_padding[0]:H + opt_padding[0],opt_padding[2]:W + opt_padding[2]] = opt_grad[:,:,opt_padding[0]:H + opt_padding[0],opt_padding[2]:W + opt_padding[2]]
        opt_tensor = opt_tensor_zeroed_rim.to(torch.float)
    else:
        opt_tensor = opt_grad.to(torch.float)
    #opt_x_grad = x1.grad.to(torch.float)

    W = nonpadded_width

    if opt_padding != None:
        ref_tensor = torch.nn.functional.pad(ref_tensor, opt_padding,  mode='constant', value=0.0)
        shift = opt_padding[0] * (W + opt_padding[2] + opt_padding[3]) + opt_padding[2]
        #if zero_rim_for_opt:
        if shift > 0:
            shift = shift - 5
        print("shift = ", shift)

    # X gradient
    print(label_string + " Allclose: ", opt_tensor.allclose(ref_tensor, rtol=1e-5, atol=1e-5))
    #print("(opt_tensor - ref_tensor).abs().sum()                                                    = ", (opt_tensor - ref_tensor).abs().sum())
    #print("(opt_tensor - ref_tensor).abs().norm(2)                                                  = ", (opt_tensor - ref_tensor).norm(2))
    print("(opt_tensor - ref_tensor).abs().norm(2) / ref_tensor.norm                                = ", (opt_tensor - ref_tensor).norm(2) / (ref_tensor.norm(2)))
    print("(opt_tensor - ref_tensor).abs().norm(inf)                                                = ", (opt_tensor - ref_tensor).norm(p=float('inf')))
    #print("opt_tensor.norm(2), ref_tensor_norm(2)                                                   = ", opt_tensor.norm(2), ref_tensor.norm(2))
    xgrad_rel_norm_diff = (opt_tensor - ref_tensor).norm(2) / (opt_tensor.norm(2))
    if xgrad_rel_norm_diff > 3.0e-6:
        print("warning, xgrad_rel_norm_diff is too large, ", xgrad_rel_norm_diff)
    #print(opt_tensor)
    #print(ref_tensor)

    for i in range(10):
        ind = i + shift if opt_padding != None else i
        print("ind opt_tensor ref_tensor = ", ind, opt_tensor.view(-1)[ind].item(), ref_tensor.view(-1)[ind].item())

    return