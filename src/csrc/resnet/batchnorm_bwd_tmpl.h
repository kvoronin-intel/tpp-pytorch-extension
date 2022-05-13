RECORD_FUNCTION("batchnorm_bwd", std::vector<c10::IValue>());

// inputs ~ grad_outs
//            ctx.save_for_backward(input, input_add, weight, mean, var, invstd, relu_mask, output)
//        inputs += ctx.saved_tensors

//auto t_O = at::empty(output_size, torch::TensorOptions().dtype(t_I.dtype()));

auto t_GO = inputs[0]; /* grad_output */
auto t_I  = inputs[1]; /* input */
auto t_IA = inputs[2]; /* input_add */
auto t_W  = inputs[3]; /* weight */

auto t_grad_input     = at::empty(t_I.sizes(),  torch::TensorOptions().dtype(t_I.dtype()));
auto t_grad_input_add = at::empty(t_IA.sizes(), torch::TensorOptions().dtype(t_I.dtype()));

auto t_grad_weight    = at::empty(t_W.sizes(),  torch::TensorOptions().dtype(t_W.dtype()));
auto t_grad_bias      = at::empty(t_W.sizes(),  torch::TensorOptions().dtype(t_W.dtype()));

//        (grad_input, grad_input_add, grad_weight, grad_bias) = batchnorm_cpp.batchnorm_bwd( inputs )

return std::vector<at::Tensor>({t_grad_input, t_grad_input_add, t_grad_weight, t_grad_bias});
