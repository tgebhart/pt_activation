# import numpy as np
# from scipy.linalg import toeplitz
# import dionysus as dion
# import math
#
# def conv_filter_as_matrix2(f, n, stride):
#     m = f.shape[0]
#     insert_locs = list(range(m,m*m,m))
#     f_reshaped = np.insert(f.reshape(-1), insert_locs, 0)
#     # f_reshaped = np.flip(f_reshaped)
#     mat = np.zeros(shape=((n-m+stride)*(n-m+stride), n*n), dtype=np.float32)
#     shift = 0
#     for i in range(mat.shape[0]-stride):
#         if (i % n) >= (n-m): # or ((i / n) % n) >= n - m:
#             shift += stride
#
#         mat[i,i+shift:i+shift+f_reshaped.shape[0]] = f_reshaped
#     return mat
#
#
#
# def conv_filter_as_matrix(f,n,stride):
#     '''See http://cs231n.github.io/convolutional-networks/
#     https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
#     https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/Convolution_as_multiplication.ipynb
#     for more info.
#     '''
#     m = f.shape[0]
#     unrolled_K = np.zeros((((n - m)//stride + 1)**2, n**2), dtype=np.float32)
#     skipped = 0
#     for i in range(n ** 2):
#          if (i % n) < (n - m)//stride + 1 and ((i / n) % n) < (n - m)//stride + 1:
#              for j in range(m):
#                  for l in range(m):
#                     unrolled_K[i - skipped, i + j * n + l] = f[j, l]
#          else:
#              skipped += 1
#     return unrolled_K
#
#
# def conv_layer_as_matrix(filters, x, stride):
#     n = x.shape[0]
#     num_filters = filters.shape[0]
#     m = filters.shape[2]
#     ret_height = num_filters*((n-m)//stride + 1)**2
#     ret_width = n**2
#     retmat = np.empty((ret_height, ret_width))
#     for i in range(num_filters):
#         fmat = conv_filter_as_matrix(filters[i], n, stride)
#         retmat[i*fmat.shape[0]:(i+1)*fmat.shape[0]] = fmat
#     return retmat
#
#
#
# def conv_filtration(f, x, conv_weight_data, id_start_1, id_start_2, percentile=None, h0_births=None, stride=1):
#     mat = conv_layer_as_matrix(conv_weight_data, x, stride)
#     x = x.cpu().detach().numpy().reshape(-1)
#     outer = np.absolute(mat*x)
#     if h0_births is None:
#         h0_births = np.zeros(x.shape)
#     else:
#         h0_births = h0_births.reshape(-1)
#
#     if percentile is None:
#         percentile_1 = 0
#     else:
#         percentile_1 = np.percentile(outer, percentile)
#     gtzx = np.argwhere(x > 0)
#
#     h1_births = np.zeros(mat.shape[0])
#     # loop over each entry in the reshaped (column) x vector
#     for xi in gtzx:
#         # compute the product of each filter value with current x in iteration.
#         all_xis = np.absolute(mat[:,xi]*x[xi])
#         max_xi = all_xis.max()
#         # set our x filtration as the highest product
#         if h0_births[xi] < max_xi:
#             h0_births[xi] = max_xi
#             # f.append(dion.Simplex([xi], max_xi))
#         gtpall_xis = np.argwhere(all_xis > percentile_1)[:,0]
#         # iterate over all products
#         for mj in gtpall_xis:
#             # if there is another filter-xi combination that has a higher
#             # product, save this as the birth time of that vertex.
#             if h1_births[mj] < all_xis[mj]:
#                 h1_births[mj] = all_xis[mj]
#             f.append(dion.Simplex([xi+id_start_1, mj+id_start_2], all_xis[mj]))
#
#     for i in np.argwhere(h0_births > 0):
#         f.append(dion.Simplex([i+id_start_1], h0_births[i]))
#
#     return f, h1_births
#
# def linear_filtration(f, h1, fc, h1_births, id_start_1, id_start_2, percentile=None, last=False):
#     h1 = h1.cpu().detach().numpy()
#     mat = fc.weight.data.cpu().detach().numpy()
#     h2_births = np.zeros(mat.shape[0])
#
#     outer = np.absolute(mat*h1)
#     if percentile is None:
#         percentile_2 = 0
#     else:
#         percentile_2 = np.percentile(outer, percentile)
#     gtzh1 = np.argwhere(h1 > 0)
#
#     for xi in gtzh1:
#         all_xis = np.absolute(mat[:,xi]*h1[xi])
#         max_xi = all_xis.max()
#         if h1_births[xi] < max_xi:
#             h1_births[xi] = max_xi
#         gtpall_xis = np.argwhere(all_xis > percentile_2)[:,0]
#
#         for mj in gtpall_xis:
#             if h2_births[mj] < all_xis[mj]:
#                 h2_births[mj] = all_xis[mj]
#             f.append(dion.Simplex([xi+id_start_1, mj+id_start_2], all_xis[mj]))
#
#
#     # now add maximum birth time for each h1 hidden vertex to the filtration.
#     for i in np.argwhere(h1_births > 0):
#         f.append(dion.Simplex([i+id_start_1], h1_births[i]))
#
#     # last linear layer in network, add final
#     if last:
#         for i in np.argwhere(h2_births > 0):
#             f.append(dion.Simplex([i+id_start_2], h2_births[i]))
#         return f
#     return f, h2_births
#
#
# def max_pooling_filtration(f, h1, pool, h1_births, id_start_1, id_start_2, percentile=None):
#     h1 = h1.cpu().detach().numpy()
#     h1_dim_size = len(h1.shape)
#
#     # shorthand for later indexing
#     channel_idx = h1_dim_size - 3
#     height_idx = h1_dim_size - 2
#     width_idx = h1_dim_size - 1
#     num_channels = h1.shape[channel_idx]
#     in_height = h1.shape[height_idx]
#     in_width = h1.shape[width_idx]
#
#     if percentile is None:
#         percentile = 0
#     else:
#         percentile = np.percentile(h1, percentile)
#
#     h1_births = np.reshape(h1_births, (num_channels, in_height, in_width))
#
#     # compute padding shape options
#     padding_height = pool.padding if isinstance(pool.padding, int) else pool.padding[0]
#     padding_width = pool.padding if isinstance(pool.padding, int) else pool.padding[1]
#
#     # compute stride shape options
#     stride_height = pool.stride if isinstance(pool.stride, int) else pool.stride[0]
#     stride_width = pool.stride if isinstance(pool.stride, int) else pool.stride[1]
#
#     # compute kernel shape options
#     kernel_height = pool.kernel_size if isinstance(pool.kernel_size, int) else pool.kernel_size[0]
#     kernel_width = pool.kernel_size if isinstance(pool.kernel_size, int) else pool.kernel_size[1]
#
#     # Compute output dimensions, assuming dilation = 1 and floor function is used.
#     # more info: https://pytorch.org/docs/stable/nn.html
#     out_height = math.floor((in_height + 2 * padding_height - 1 * (kernel_height - 1) - 1)/stride_height + 1)
#     out_width = math.floor((in_width + 2 * padding_width - 1 * (kernel_width - 1) - 1)/stride_width + 1)
#
#     h2_births = np.zeros((num_channels, out_height, out_width))
#
#     for c in range(num_channels):
#         mat = h1[c, :, :]
#         # assume padding zero for now
#         r_s = 0
#         r_e = kernel_width
#         c_s = 0
#         c_e = kernel_height
#
#         oh = 0
#         ow = 0
#
#         r = np.arange(kernel_height)
#         col = np.arange(kernel_width)
#         idx_mat = np.transpose([np.tile(r, len(col)), np.repeat(col, len(r))])
#         step_mat = np.transpose(np.array([np.zeros(idx_mat.shape[0], dtype='int64'), np.ones(idx_mat.shape[0], dtype='int64')]))
#         reset_mat = np.transpose(np.array([np.ones(idx_mat.shape[0], dtype='int64'), np.full(idx_mat.shape[0], in_width-kernel_width-1, dtype='int64')]))
#
#         while idx_mat[-1,0] < mat.shape[0]:
#
#             while idx_mat[-1,-1] < mat.shape[1]:
#
#                 s_mat = np.absolute(np.take(mat, idx_mat))
#                 max_val = s_mat.max()
#                 # save maximum as potential birth of next layer's node
#                 h2_births[c,oh,ow] = max_val
#
#                 # attach edges to next layer's node
#                 for idx in idx_mat:
#                     t = mat[idx[0],idx[1]]
#                     if t > h1_births[c,idx[0],idx[1]]:
#                         h1_births[c,idx[0],idx[1]] = t
#                     if t > percentile:
#                         id_offset = (c*in_height*in_width) + (idx[0]*in_width + idx[1])
#                         id_offset2 = (c*out_height*out_width) + (oh*out_width + ow)
#                         f.append(dion.Simplex([id_offset+id_start_1, id_offset2+id_start_2], t))
#
#
#                 ow += 1
#                 idx_mat = idx_mat + step_mat
#
#             ow = 0
#             oh += 1
#             idx_mat = idx_mat + reset_mat
#
#     # now add maximum birth time for each h1 hidden vertex to the filtration.
#     h1_births = h1_births.flatten()
#     for i in np.argwhere(h1_births > 0):
#         f.append(dion.Simplex([i+id_start_1], h1_births[i]))
#
#     return f, h2_births
#
#
#
#
#
#
#
#
#
# def matrix_to_vector(i):
#     input_h, input_w = i.shape
#     output_vector = np.zeros(input_h*input_w)
#     # flip the input matrix up-down because last row should go first
#     i = np.flipud(i)
#     for j,row in enumerate(i):
#         st = j*input_w
#         nd = st + input_w
#         output_vector[st:nd] = row
#
#     return output_vector
#
# def vector_to_matrix(ipt, output_shape):
#     output_h, output_w = output_shape
#     output = np.zeros(output_shape, dtype='float32')
#     for i in range(output_h):
#         st = i*output_w
#         nd = st + output_w
#         output[i, :] = ipt[st:nd]
#     # flip the output matrix up-down to get correct result
#     output=np.flipud(output)
#     return output
#
#
# def toeplitz_multiply(x, f):
#     i_row_num, i_col_num = x.shape
#     f_row_num, f_col_num = f.shape
#
#     out_row_num = i_row_num + f_row_num - 1
#     out_col_num = i_col_num + f_col_num - 1
#
#     padded_f = np.pad(f, ((out_row_num - f_row_num, 0),
#                      (0, out_col_num - f_col_num)),
#                      'constant', constant_values=0)
#
#     toeplitz_list = []
#     for i in range(padded_f.shape[0]-1, -1, -1):
#         c = padded_f[i,:]
#         r = np.r_[c[0], np.zeros(i_col_num-1)]
#
#         toeplitz_m = toeplitz(c,r)
#         toeplitz_list.append(toeplitz_m)
#
#     c = range(1,padded_f.shape[0]+1)
#     r = np.r_[c[0], np.zeros(i_row_num-1, dtype=int)]
#     doubly_indices = toeplitz(c,r)
#
#     toeplitz_shape = toeplitz_list[0].shape
#     h = toeplitz_shape[0]*doubly_indices.shape[0]
#     w = toeplitz_shape[1]*doubly_indices.shape[1]
#     doubly_blocked_shape = [h,w]
#     doubly_blocked = np.zeros(doubly_blocked_shape, dtype='float32')
#
#     b_h, b_w = toeplitz_shape # hight and withs of each block
#     for i in range(doubly_indices.shape[0]):
#         for j in range(doubly_indices.shape[1]):
#             start_i = i * b_h
#             start_j = j * b_w
#             end_i = start_i + b_h
#             end_j = start_j + b_w
#             doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]
#
#     vectorized_x = matrix_to_vector(x)
#     result_vector = np.matmul(doubly_blocked, vectorized_x)
#     out_shape = [out_row_num, out_col_num]
#
#     return vectorized_x, doubly_blocked, vector_to_matrix(result_vector, out_shape)

import numpy as np
from scipy.linalg import toeplitz
import dionysus as dion
import math
# from profilestats import profile

def spec_hash(t):
    # l,c,i = t[0],t[1],t[2]
    # layer = "{:02d}".format(l)
    # channel = "{:03d}".format(c)
    # i = str(i)
    # return int(i+channel+layer)
    return t

def abs_sort(s1, s2):
    if s1.dimension() != s2.dimension():
        return (s1.dimension() > s2.dimension()) - (s1.dimension() < s2.dimension())
    else:
        return (abs(s1.data) > abs(s2.data)) - (abs(s1.data) < abs(s2.data))


def conv_filter_as_matrix2(f, n, stride):
    m = f.shape[0]
    insert_locs = list(range(m,m*m,m))
    f_reshaped = np.insert(f.reshape(-1), insert_locs, 0)
    # f_reshaped = np.flip(f_reshaped)
    mat = np.zeros(shape=((n-m+stride)*(n-m+stride), n*n), dtype=np.float32)
    shift = 0
    for i in range(mat.shape[0]-stride):
        if (i % n) >= (n-m): # or ((i / n) % n) >= n - m:
            shift += stride

        mat[i,i+shift:i+shift+f_reshaped.shape[0]] = f_reshaped
    return mat



def conv_filter_as_matrix(f,n,stride):
    '''See http://cs231n.github.io/convolutional-networks/
    https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
    https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/Convolution_as_multiplication.ipynb
    for more info.
    '''
    m = f.shape[0]
    unrolled_K = np.zeros((((n - m)//stride + 1)**2, n**2), dtype=np.float32)
    skipped = 0
    for i in range(n ** 2):
         if (i % n) < (n - m)//stride + 1 and ((i / n) % n) < (n - m)//stride + 1:
             for j in range(m):
                 for l in range(m):
                    unrolled_K[i - skipped, i + j * n + l] = f[j, l]
         else:
             skipped += 1
    return unrolled_K


def conv_layer_as_matrix(filters, x, stride):
    n = x.shape[0]
    num_filters = filters.shape[0]
    m = filters.shape[2]
    ret_height = num_filters*((n-m)//stride + 1)**2
    ret_width = n**2
    retmat = np.empty((ret_height, ret_width))
    for i in range(num_filters):
        fmat = conv_filter_as_matrix(filters[i], n, stride)
        retmat[i*fmat.shape[0]:(i+1)*fmat.shape[0]] = fmat
    return retmat

def conv_filtration_static(f, x, conv_weight_data, id_start_1, id_start_2, percentile=None, h0_births=None, stride=1):
    mat = conv_layer_as_matrix(conv_weight_data, x, stride)
    x = x.cpu().detach().numpy().reshape(-1)
    if h0_births is None:
        h0_births = np.zeros(x.shape)
    else:
        h0_births = h0_births.reshape(-1)

    if percentile is None or percentile == 0.0:
        percentile_1 = 0
    else:
        percentile_1 = np.percentile(mat, percentile)
    gtzx = np.argwhere(x > 0)

    h1_births = np.zeros(mat.shape[0])
    # loop over each entry in the reshaped (column) x vector
    for xi in gtzx:
        # compute the product of each filter value with current x in iteration.
        all_xis = np.absolute(mat[:,xi])
        max_xi = all_xis.max()
        # set our x filtration as the highest product
        if h0_births[xi] < max_xi:
            h0_births[xi] = max_xi
            # f.append(dion.Simplex([xi], max_xi))
        gtpall_xis = np.argwhere(all_xis > percentile_1)[:,0]
        # iterate over all products
        for mj in gtpall_xis:
            # if there is another filter-xi combination that has a higher
            # product, save this as the birth time of that vertex.
            if h1_births[mj] < all_xis[mj]:
                h1_births[mj] = all_xis[mj]
            f.append(dion.Simplex([xi+id_start_1, mj+id_start_2], all_xis[mj]))

    for i in np.argwhere(h0_births > 0):
        f.append(dion.Simplex([i+id_start_1], h0_births[i]))

    return f, h1_births


def linear_filtration_static(f, h1, fc, h1_births, id_start_1, id_start_2, percentile=None, last=False):
    h1 = h1.cpu().detach().numpy()
    mat = fc.weight.data.cpu().detach().numpy()
    h2_births = np.zeros(mat.shape[0])

    if percentile is None:
        percentile_2 = 0
    else:
        percentile_2 = np.percentile(mat, percentile)
    gtzh1 = np.argwhere(h1 > 0)

    for xi in gtzh1:
        all_xis = np.absolute(mat[:,xi])
        max_xi = all_xis.max()
        if h1_births[xi] < max_xi:
            h1_births[xi] = max_xi
        gtpall_xis = np.argwhere(all_xis > percentile_2)[:,0]

        for mj in gtpall_xis:
            if h2_births[mj] < all_xis[mj]:
                h2_births[mj] = all_xis[mj]
            f.append(dion.Simplex([xi+id_start_1, mj+id_start_2], all_xis[mj]))


    # now add maximum birth time for each h1 hidden vertex to the filtration.
    for i in np.argwhere(h1_births > 0):
        f.append(dion.Simplex([i+id_start_1], h1_births[i]))

    # last linear layer in network, add final
    if last:
        for i in np.argwhere(h2_births > 0):
            f.append(dion.Simplex([i+id_start_2], h2_births[i]))
        return f
    return f, h2_births

'''
- issue seems to be with keeping track of which nodes are born when.
    - We have access to this information a priori via the hidden activations
    - How do we use this?
'''

def conv_filtration_fast(x, mat, id_start_1, id_start_2, percentile=None, h0_births=None, stride=1):

    x = x.cpu().detach().numpy().reshape(-1)
    x_diag = np.diag(x)
    edges = np.absolute(np.matmul(mat, x_diag))

    if percentile is None:
        this_percentile = 0
    else:
        this_percentile = np.percentile(edges, percentile)

    h0_births = np.max(edges, axis=0)
    h1_births = np.max(edges, axis=1)

    ret = [([j+id_start_1, i+id_start_2], edges[i,j]) for i,j in np.argwhere(edges > this_percentile)]

    return ret, h0_births, h1_births, this_percentile



def conv_filtration_fast2(x, mat, layer, channel, nlc, percentile=0, absolute_value=True):

    x = x.reshape(-1)
    x_diag = np.diag(x)
    edges = np.matmul(mat, x_diag)
    if absolute_value:
        edges_comp = np.absolute(edges)
    else:
        edges_comp = edges

    h0_births = np.max(edges_comp, axis=0)
    h1_births = np.max(edges_comp, axis=1)

    ret = [([spec_hash((layer, channel, j)), spec_hash((layer+1, i//nlc, i%nlc))], [edges_comp[i,j].item(),edges[i,j].item()]) for i,j in np.argwhere(edges_comp > percentile)]

    return ret, h0_births, h1_births



def conv_filtration(f, x, conv_weight_data, id_start_1, id_start_2, percentile=None, h0_births=None, stride=1):
    mat = conv_layer_as_matrix(conv_weight_data, x, stride)
    x = x.cpu().detach().numpy().reshape(-1)
    outer = np.absolute(mat*x)
    if h0_births is None:
        h0_births = np.zeros(x.shape)
    else:
        h0_births = h0_births.reshape(-1)

    if percentile is None:
        percentile_1 = 0
    else:
        percentile_1 = np.percentile(outer, percentile)
    gtzx = np.argwhere(x > 0)

    h1_births = np.zeros(mat.shape[0])
    # loop over each entry in the reshaped (column) x vector
    for xi in gtzx:
        # compute the product of each filter value with current x in iteration.
        all_xis = np.absolute(mat[:,xi]*x[xi])
        max_xi = all_xis.max()
        # set our x filtration as the highest product
        if h0_births[xi] < max_xi:
            h0_births[xi] = max_xi
            # f.append(dion.Simplex([xi], max_xi))
        gtpall_xis = np.argwhere(all_xis > percentile_1)[:,0]
        # iterate over all products
        for mj in gtpall_xis:
            # if there is another filter-xi combination that has a higher
            # product, save this as the birth time of that vertex.
            if h1_births[mj] < all_xis[mj]:
                h1_births[mj] = all_xis[mj]
            f.append(dion.Simplex([xi+id_start_1, mj+id_start_2], all_xis[mj]))

    for i in np.argwhere(h0_births > 0):
        f.append(dion.Simplex([i+id_start_1], h0_births[i]))

    return f, h1_births


def linear_filtration_fast(h1, fc, id_start_1, id_start_2, percentile=None, last=False):
    h1 = h1.cpu().detach().numpy()
    mat = fc.weight.data.cpu().detach().numpy()

    edges = np.absolute(mat*h1)

    if percentile is None:
        this_percentile = 0
    else:
        this_percentile = np.percentile(edges, percentile)

    h1_births = np.max(edges, axis=0)
    h2_births = np.max(edges, axis=1)

    ret = [([j+id_start_1, i+id_start_2], edges[i,j]) for i,j in np.argwhere(edges > this_percentile)]
    return ret, h1_births, h2_births, this_percentile


def linear_filtration_fast2(h1, fc, layer, channel, percentile=0, absolute_value=True):
    # h1 = h1.cpu().detach().numpy()
    mat = fc.weight.data.cpu().detach().numpy()
    edges = mat*h1
    if absolute_value:
        edges_comp = np.absolute(edges)
    else:
        edges_comp = edges

    h1_births = np.max(edges_comp, axis=0)
    h2_births = np.max(edges_comp, axis=1)

    ret = [([spec_hash((layer, channel, j)), spec_hash((layer+1, channel, i))], [edges_comp[i,j],edges[i,j]]) for i,j in np.argwhere(edges_comp > percentile)]
    return ret, h1_births, h2_births



def linear_filtration(f, h1, fc, h1_births, id_start_1, id_start_2, percentile=None, last=False):
    h1 = h1.cpu().detach().numpy()
    mat = fc.weight.data.cpu().detach().numpy()
    h2_births = np.zeros(mat.shape[0])

    outer = np.absolute(mat*h1)
    if percentile is None:
        percentile_2 = 0
    else:
        percentile_2 = np.percentile(outer, percentile)
    gtzh1 = np.argwhere(h1 > 0)

    for xi in gtzh1:
        all_xis = np.absolute(mat[:,xi]*h1[xi])
        max_xi = all_xis.max()
        if h1_births[xi] < max_xi:
            h1_births[xi] = max_xi
        gtpall_xis = np.argwhere(all_xis > percentile_2)[:,0]

        for mj in gtpall_xis:
            if h2_births[mj] < all_xis[mj]:
                h2_births[mj] = all_xis[mj]
            f.append(dion.Simplex([xi+id_start_1, mj+id_start_2], all_xis[mj]))


    # now add maximum birth time for each h1 hidden vertex to the filtration.
    for i in np.argwhere(h1_births > 0):
        f.append(dion.Simplex([i+id_start_1], h1_births[i]))

    # last linear layer in network, add final
    if last:
        for i in np.argwhere(h2_births > 0):
            f.append(dion.Simplex([i+id_start_2], h2_births[i]))
        return f
    return f, h2_births


def max_pooling_filtration(h1, pool, layer, channel, percentile=0, absolute_value=True, linear_next=False):
    h1_dim_size = len(h1.shape)

    # shorthand for later indexing
    # channel_idx = h1_dim_size - 3
    # height_idx = h1_dim_size - 2
    # width_idx = h1_dim_size - 1
    height_idx = 0
    width_idx = 1
    # num_channels = h1.shape[channel_idx]
    in_height = h1.shape[height_idx]
    in_width = h1.shape[width_idx]

    # percentile = np.percentile(h1, percentile)

    h1_births = np.zeros((in_height, in_width))

    # compute padding shape options
    padding_height = pool.padding if isinstance(pool.padding, int) else pool.padding[0]
    padding_width = pool.padding if isinstance(pool.padding, int) else pool.padding[1]

    # compute stride shape options
    stride_height = pool.stride if isinstance(pool.stride, int) else pool.stride[0]
    stride_width = pool.stride if isinstance(pool.stride, int) else pool.stride[1]

    # compute kernel shape options
    kernel_height = pool.kernel_size if isinstance(pool.kernel_size, int) else pool.kernel_size[0]
    kernel_width = pool.kernel_size if isinstance(pool.kernel_size, int) else pool.kernel_size[1]

    dilation_height = pool.dilation if isinstance(pool.dilation, int) else pool.dilation[0]
    dilation_width = pool.dilation if isinstance(pool.dilation, int) else pool.dilation[1]

    # Compute output dimensions, assuming floor function is used.
    # more info: https://pytorch.org/docs/stable/nn.html
    out_height = math.floor(((in_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1)/stride_height) + 1)
    out_width = math.floor(((in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1)/stride_width) + 1)

    h2_births = np.zeros((out_height, out_width))

    mat = h1
    # assume padding zero for now
    r_s = 0
    r_e = kernel_width
    c_s = 0
    c_e = kernel_height

    oh = 0
    ow = 0

    ret = []
    r = np.arange(kernel_height)
    col = np.arange(kernel_width)
    idx_mat = np.transpose([np.tile(r, len(col)), np.repeat(col, len(r))])
    step_mat = np.transpose(np.array([np.zeros(idx_mat.shape[0], dtype='int64'), np.ones(idx_mat.shape[0], dtype='int64')]))
    reset_mat = np.transpose(np.array([np.ones(idx_mat.shape[0], dtype='int64'), np.full(idx_mat.shape[0], in_width-kernel_width-1, dtype='int64')]))

    il = 0
    while idx_mat[-1,0] < mat.shape[0]:

        while idx_mat[-1,-1] < mat.shape[1]:


            s_mat = np.take(mat, idx_mat)
            max_val = s_mat.max()
            # save maximum as potential birth of next layer's node
            h2_births[oh,ow] = max_val

            # attach edges to next layer's node
            for idx in idx_mat:
                t = mat[idx[0],idx[1]]
                # h1_births[idx[0],idx[1]] = t
                if t > percentile:
                    id_offset = (idx[0]*in_width + idx[1]) # + (in_height*in_width)
                    id_offset2 =  (oh*out_width + ow) # + (out_height*out_width)
                    # id_offset2 = il
                    if linear_next:
                        ret.append(([spec_hash((layer,channel,id_offset)), spec_hash((layer+1,0,channel*(out_width*out_height)+id_offset2))], [t,t]))
                    else:
                        ret.append(([spec_hash((layer,channel,id_offset)), spec_hash((layer+1,channel,il))], [t,t]))


            ow += 1
            idx_mat = idx_mat + stride_height*step_mat
            il += 1
        ow = 0
        oh += 1
        idx_mat = idx_mat + reset_mat

    # now add maximum birth time for each h1 hidden vertex to the filtration.
    # for i in np.argwhere(h1_births > 0):
    #     f.append(dion.Simplex([i+id_start_1], h1_births[i]))

    return ret, h1.flatten(), h2_births.flatten()









def matrix_to_vector(i):
    input_h, input_w = i.shape
    output_vector = np.zeros(input_h*input_w)
    # flip the input matrix up-down because last row should go first
    i = np.flipud(i)
    for j,row in enumerate(i):
        st = j*input_w
        nd = st + input_w
        output_vector[st:nd] = row

    return output_vector

def vector_to_matrix(ipt, output_shape):
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype='float32')
    for i in range(output_h):
        st = i*output_w
        nd = st + output_w
        output[i, :] = ipt[st:nd]
    # flip the output matrix up-down to get correct result
    output=np.flipud(output)
    return output


def toeplitz_multiply(x, f):
    i_row_num, i_col_num = x.shape
    f_row_num, f_col_num = f.shape

    out_row_num = i_row_num + f_row_num - 1
    out_col_num = i_col_num + f_col_num - 1

    padded_f = np.pad(f, ((out_row_num - f_row_num, 0),
                     (0, out_col_num - f_col_num)),
                     'constant', constant_values=0)

    toeplitz_list = []
    for i in range(padded_f.shape[0]-1, -1, -1):
        c = padded_f[i,:]
        r = np.r_[c[0], np.zeros(i_col_num-1)]

        toeplitz_m = toeplitz(c,r)
        toeplitz_list.append(toeplitz_m)

    c = range(1,padded_f.shape[0]+1)
    r = np.r_[c[0], np.zeros(i_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c,r)

    toeplitz_shape = toeplitz_list[0].shape
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h,w]
    doubly_blocked = np.zeros(doubly_blocked_shape, dtype='float32')

    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

    vectorized_x = matrix_to_vector(x)
    result_vector = np.matmul(doubly_blocked, vectorized_x)
    out_shape = [out_row_num, out_col_num]

    return vectorized_x, doubly_blocked, vector_to_matrix(result_vector, out_shape)
