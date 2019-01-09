import torch
import numpy
import basiliskpybind
import copy
import torch.nn as nn

def persistence_score(d):
    diag = np.array([[pt.birth,pt.death] for pt in d])
    diag = diag[~np.isinf(diag[:,1])]
    return np.average(diag[:,0] - diag[:,1])

class HomologyRegularizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hiddens, data, model, percentile=None, learning_rate=1e-4):
        this_hiddens = [hiddens[0][0], hiddens[1][0], hiddens[2][0]]
        f = model.compute_static_filtration(data[0,0], this_hiddens)
        m = dion.homology_persistence(f)
        dgms = dion.init_diagrams(m, f)

        this_params = copy.deepcopy(model.parameters())
        for param in model.parameters():
            param.data -= learning_rate * param.grad.data

        _,new_hiddens = model(data)
        model.parameters = this_params
        ctx.save_for_backward()

        return torch.tensor(persistence_score(dgms))

    @staticmethod
    def backward(ctx, error):

        grad_record, input_shape, input_count, sample_count, lambda_depth, dimension_count = ctx.saved_tensors

        numpy_error = error.cpu().detach().numpy()

        return error.new_tensor()


# class PersistenceLandscape(torch.nn.Module):
#     def __init__(self, sample_points, dimensions, lambda_depth):
#         super(PersistenceLandscape, self).__init__()
#         self.sample_points = nn.Parameter(torch.tensor(sample_points), requires_grad=False)
#         self.dimensions = nn.Parameter(torch.tensor(dimensions), requires_grad=False)
#         self.lambda_depth = nn.Parameter(torch.tensor(lambda_depth), requires_grad=False)
#
#     def forward(self, input):
#         return PersistenceLandscapeFunction.apply(input, self.sample_points, self.dimensions, self.lambda_depth)
#
#
# class PersistenceLandscapeBatchFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, sample_points, dimensions, lambda_depth):
#         batch_count = input.size()[0]
#         input_shape = input.size()[1:]
#         input_list = input.view(-1).tolist()
#         sample_points_list = sample_points.tolist()
#         dimensions_list = dimensions.tolist()
#
#         outputs = basiliskpybind.generate_persistence_landscapes_batch(input_list, input_shape, sample_points_list, dimensions_list, lambda_depth, batch_count)
#
#         result = input.new_tensor(outputs[0]).view(int(batch_count), len(dimensions_list), int(lambda_depth), len(sample_points_list))
#         grad = outputs[1]
#
#         ctx.save_for_backward(input.new_tensor(batch_count, dtype=torch.long), input.new_tensor(grad), input.new_tensor(input_shape, dtype=torch.long),
#             input.new_tensor(len(input_list)/batch_count, dtype=torch.long), input.new_tensor(len(sample_points_list), dtype=torch.long),
#             lambda_depth, input.new_tensor(len(dimensions_list), dtype=torch.long))
#
#         return result
#
#     @staticmethod
#     def backward(ctx, error):
#         batch_count, grad_record, input_shape, input_count, sample_count, lambda_depth, dimension_count = ctx.saved_tensors
#
#         grad_record = grad_record.view(int(batch_count.item()), int(input_count.item()), int(dimension_count.item()),
#             lambda_depth.item(), int(sample_count.item()))
#
#         error = error.unsqueeze(1)
#
#         grad_input = error * grad_record
#
#         for i in [4, 3, 2]:
#             grad_input = torch.sum(grad_input, i)
#
#         grad_input = grad_input.view([int(batch_count.item())] + input_shape.tolist())
#
#         batch_count_grad = grad_sample_points = grad_dimension = grad_lambda_depth = None
#         return grad_input, batch_count_grad, grad_sample_points, grad_dimension, grad_lambda_depth
#
#
# class PersistenceLandscapeBatch(torch.nn.Module):
#     def __init__(self, sample_points, dimensions, lambda_depth):
#         super(PersistenceLandscapeBatch, self).__init__()
#         self.sample_points = nn.Parameter(torch.tensor(sample_points), requires_grad=False)
#         self.dimensions = nn.Parameter(torch.tensor(dimensions), requires_grad=False)
#         self.lambda_depth = nn.Parameter(torch.tensor(lambda_depth), requires_grad=False)
#
#     def forward(self, input):
#         return PersistenceLandscapeBatchFunction.apply(input, self.sample_points, self.dimensions, self.lambda_depth)
