import numpy as np
from Utils.data_utils import plot_conv_images

def conv_forward(x, w, b, conv_param):
    """
    Computes the forward pass for a convolutional layer.

    Inputs:
    - x: Input data, of shape (N, H, W, C) 
        N : Input number
        H : Height
        W : Width
        C : Channel
    - w: Weights, of shape (F, WH, WW, C)
        F : Filter number
        WH : Filter height
        WW : Filter width
        C : Channel
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields, of shape (1, SH, SW, 1)
          SH : Stride height
          SW : Stride width
      - 'padding': "valid" or "same". "valid" means no padding.
        "same" means zero-padding the input so that the output has the shape as (N, ceil(H / SH), ceil(W / SW), F)
        If the padding on both sides (top vs bottom, left vs right) are off by one, the bottom and right get the additional padding.
         
    Outputs:
    - out: Output data
    - cache: (x, w, b, conv_param)
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    
    import math
    
    # adjust padding to images
    # images: array of shape (N, H, W, C)
    # lt: # of paddings at left and top
    # rd: # of paddings at ridht and down
    def adjust_padding(images, left, right, top, bottom):
        return np.pad(x, [(0,0), (top, bottom), (left, right), (0,0)], 'constant', constant_values=0)
    
    N, H, W, C = x.shape
    F, WH, WW, C = w.shape
    _, SH, SW, _ = conv_param['stride']
    out = None
    
    # valid padding
    if conv_param['padding'] == 'valid':
        l_padding, r_padding, t_padding, b_padding = 0, 0, 0, 0
        OH, OW = math.ceil((H-WH+1) / SH), math.ceil((W-WW+1) / SW)
        
        
    # same padding
    elif conv_param['padding'] == 'same':
        OH = math.ceil(H / SH)
        OW = math.ceil(W / SW)
        
        PH = (OH-1)*SH - H + WH
        PW = (OW-1)*SW - W + WW
        t_padding = math.floor(PH/2)
        b_padding = PH - t_padding
        l_padding = math.floor(PW/2)
        r_padding = PW - l_padding
        
    padded_x = adjust_padding(x, l_padding, r_padding, t_padding, b_padding)
    out = np.zeros((N, OH, OW, F), dtype=np.float32)

    for N_idx, image in enumerate(padded_x):
        out_maps = np.zeros((OH, OW, F), dtype=np.float32)
        for F_idx, kernel in enumerate(w):
            activation_map = np.zeros((OH, OW), dtype=np.float32)
            
            for h_idx in range(OH):
                for w_idx in range(OW):
                    h_start = SH * h_idx
                    h_end = h_start + WH
                    w_start = SW * w_idx
                    w_end = w_start + WW

                    receptive_field = image[h_start:h_end, w_start:w_end, :]
                    activation_map[h_idx, w_idx] = np.sum(np.multiply(receptive_field, kernel))
                    
            activation_map += b[F_idx] # add bias
            out_maps[:, :, F_idx] = activation_map
            
        out[N_idx, :, :, :] = out_maps
        
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    cache = (x, w, b, conv_param)
    return out, cache
    

def conv_backward(dout, cache):
    """
    Computes the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward

    Outputs:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    
    import math
    def adjust_padding(images, left, right, top, bottom):
        return np.pad(x, [(0,0), (top, bottom), (left, right), (0,0)], 'constant', constant_values=0)
    
    x, w, b, conv_param = cache
    N, H, W, C = x.shape
    F, WH, WW, _ = w.shape
    _, SH, SW, _ = conv_param['stride']
    padding = conv_param['padding']
    _, OH, OW, _ = dout.shape
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # calculate number of paddings
    if padding == 'valid':
        l_pad, r_pad, t_pad, b_pad = 0, 0, 0, 0
    elif padding == 'same':
        oh = math.ceil(H / SH) # out height
        ow = math.ceil(W / SW) # out width
        PH = (oh-1)*SH - H + WH # pad height
        PW = (ow-1)*SW - W + WW # pad width
        t_pad = math.floor(PH/2)
        b_pad = PH - t_pad
        l_pad = math.floor(PW/2)
        r_pad = PW - l_pad
        
        assert(OH == oh)
        assert(OW == ow)

    padded_x = adjust_padding(x, l_pad, r_pad, t_pad, b_pad)
    padded_dx = adjust_padding(dx, l_pad, r_pad, t_pad, b_pad)
    
    # dx: filter wise sum, each dx at each batch
    # dw: batch wise mean, each dw at each filter
    # db: batch wise mean
    
    for n_idx in range(N):
        for f_idx in range(F):
            for oh_idx in range(OH):
                for ow_idx in range(OW):
                    h_start = SH * oh_idx
                    h_end = h_start + WH
                    w_start = SW * ow_idx
                    w_end = w_start + WW
                    padded_dx[n_idx, h_start:h_end, w_start:w_end, :] += w[f_idx, :, :, :] * dout[n_idx, oh_idx, ow_idx, f_idx]
                    dw[f_idx, :, :, :] += padded_x[n_idx, h_start:h_end, w_start:w_end, :] * dout[n_idx, oh_idx, ow_idx, f_idx]
                    db[f_idx] += 1
                    
        # dx[n_idx, :, :, :] = padded_dx 에서 패딩 제외하고 가져오기
        dx[n_idx, :, :, :] = padded_dx[n_idx, t_pad:H+t_pad, l_pad:W+l_pad, :]
        
    dw /= N # batch wise mean
    db /= N # batch wise mean
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return dx, dw, db

def max_pool_forward(x, pool_param):
    """
    Computes the forward pass for a pooling layer.
    
    For your convenience, you only have to implement padding=valid.
    
    Inputs:
    - x: Input data, of shape (N, H, W, C)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The number of pixels between adjacent pooling regions, of shape (1, SH, SW, 1)

    Outputs:
    - out: Output data
    - cache: (x, pool_param)
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    import math
    
    N, H, W, C = x.shape
    p_h = pool_param['pool_height']
    p_w = pool_param['pool_width']
    _, SH, SW, _ = pool_param['stride']
    OH, OW = math.floor((H-p_h) / SH) + 1, math.floor((W-p_w) / SW) + 1
    
    out = np.zeros((N, OH, OW, C), dtype=np.float32)
    for N_idx, input_layer in enumerate(x):
        activation_channels = np.zeros((OH, OW, C), dtype=np.float32)
        for C_idx in range(C):
            activation_map = np.zeros((OH, OW), dtype=np.float32) # activation map at channel C
            for h_idx in range(OH):
                for w_idx in range(OW):
                    h_start = h_idx * SH
                    h_end = h_start + p_h
                    w_start = w_idx * SW
                    w_end = w_start + p_w
                    max_value = np.max(input_layer[h_start:h_end, w_start:w_end, C_idx])
                    activation_map[h_idx, w_idx] = max_value
                    
            activation_channels[:, :, C_idx] = activation_map
            

        out[N_idx, :, :, :] = activation_channels
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    Computes the backward pass for a max pooling layer.

    For your convenience, you only have to implement padding=valid.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in max_pool_forward.

    Outputs:
    - dx: Gradient with respect to x
    """
    ##############################################################################
    #                          IMPLEMENT YOUR CODE                               #
    ##############################################################################
    
    import math
    
    N, OH, OW, C = dout.shape
    x, pool_param = cache
    _N, H, W, _C = x.shape
    _, SH, SW, _ = pool_param['stride']
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    
    assert(N == _N)
    assert(C == _C)
    
    assert(OH == math.ceil((H-PH+1) / SH))
    assert(OW == math.ceil((W-PW+1) / SW))

    # only implement valid padding
    # dx: channel wise, batch wise
    dx = np.zeros_like(x)
    
    for n_idx in range(N):
        for c_idx in range(C):
            for oh_idx in range(OH):
                for ow_idx in range(OW):
                    h_start = oh_idx * SH
                    h_end = h_start + PH
                    w_start = ow_idx * SW
                    w_end = w_start + PW
                    dx[n_idx, h_start:h_end, w_start:w_end, c_idx] = \
                        np.equal(x[n_idx, h_start:h_end, w_start:w_end, c_idx], dout[n_idx, oh_idx, ow_idx, c_idx]).astype(int)
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return dx

def _rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def Test_conv_forward(num):
    """ Test conv_forward function """
    if num == 1:
        x_shape = (2, 4, 8, 3)
        w_shape = (2, 2, 4, 3)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.05, num=2)
        conv_param = {'stride': np.array([1,2,3,1]), 'padding': 'valid'}
        out, _ = conv_forward(x, w, b, conv_param)
        correct_out = np.array([[[[  5.12264676e-02,  -7.46786231e-02],
                                  [ -1.46819650e-03,   4.58694441e-02]],
                                 [[ -2.29811741e-01,   5.68244402e-01],
                                  [ -2.82506405e-01,   6.88792470e-01]]],
                                [[[ -5.10849950e-01,   1.21116743e+00],
                                  [ -5.63544614e-01,   1.33171550e+00]],
                                 [[ -7.91888159e-01,   1.85409045e+00],
                                  [ -8.44582823e-01,   1.97463852e+00]]]])
    else:
        x_shape = (2, 5, 5, 3)
        w_shape = (2, 2, 4, 3)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.05, num=2)
        conv_param = {'stride': np.array([1,3,2,1]), 'padding': 'same'}
        out, _ = conv_forward(x, w, b, conv_param)
        correct_out = np.array([[[[ -5.28344995e-04,  -9.72797373e-02],
                                  [  2.48150793e-02,  -4.31486506e-02],
                                  [ -4.44809367e-02,   3.35499072e-02]],
                                 [[ -2.01784949e-01,   5.34249607e-01],
                                  [ -3.12925889e-01,   7.29491646e-01],
                                  [ -2.82750250e-01,   3.50471227e-01]]],
                                [[[ -3.35956019e-01,   9.55269170e-01],
                                  [ -5.38086534e-01,   1.24458518e+00],
                                  [ -4.41596459e-01,   5.61752106e-01]],                             
                                 [[ -5.37212623e-01,   1.58679851e+00],
                                  [ -8.75827502e-01,   2.01722547e+00],
                                  [ -6.79865772e-01,   8.78673426e-01]]]])

    return _rel_error(out, correct_out)


def Test_conv_forward_IP(x):
    """ Test conv_forward function with image processing """
    w = np.zeros((2, 3, 3, 3))
    w[0, 1, 1, :] = [0.3, 0.6, 0.1]
    w[1, :, :, 2] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    b = np.array([0, 128])
    
    out, _ = conv_forward(x, w, b, {'stride': np.array([1,1,1,1]), 'padding': 'same'})
    plot_conv_images(x, out)
    return
    
def Test_max_pool_forward():   
    """ Test max_pool_forward function """
    x_shape = (2, 5, 5, 3)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    pool_param = {'pool_width': 2, 'pool_height': 3, 'stride': [1,2,4,1]}
    out, _ = max_pool_forward(x, pool_param)
    correct_out = np.array([[[[ 0.03288591,  0.03691275,  0.0409396 ]],
                             [[ 0.15369128,  0.15771812,  0.16174497]]],
                            [[[ 0.33489933,  0.33892617,  0.34295302]],
                             [[ 0.4557047,   0.45973154,  0.46375839]]]])
    
    return _rel_error(out, correct_out)

def _eval_numerical_gradient_array(f, x, df, h=1e-5):
    """ Evaluate a numeric gradient for a function """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        p = np.array(x)
        p[ix] = x[ix] + h
        pos = f(p)
        p[ix] = x[ix] - h
        neg = f(p)
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

def Test_conv_backward(num):
    """ Test conv_backward function """
    if num == 1:
        x = np.random.randn(2, 4, 8, 3)
        w = np.random.randn(2, 2, 4, 3)
        b = np.random.randn(2,)
        conv_param = {'stride': np.array([1,2,3,1]), 'padding': 'valid'}
        dout = np.random.randn(2, 2, 2, 2)
    else:
        x = np.random.randn(2, 5, 5, 3)
        w = np.random.randn(2, 2, 4, 3)
        b = np.random.randn(2,)
        conv_param = {'stride': np.array([1,3,2,1]), 'padding': 'same'}
        dout = np.random.randn(2, 2, 3, 2)
    
    out, cache = conv_forward(x, w, b, conv_param)
    dx, dw, db = conv_backward(dout, cache)
    
    dx_num = _eval_numerical_gradient_array(lambda x: conv_forward(x, w, b, conv_param)[0], x, dout)
    dw_num = _eval_numerical_gradient_array(lambda w: conv_forward(x, w, b, conv_param)[0], w, dout)
    db_num = _eval_numerical_gradient_array(lambda b: conv_forward(x, w, b, conv_param)[0], b, dout)
    
    return (_rel_error(dx, dx_num), _rel_error(dw, dw_num), _rel_error(db, db_num))

def Test_max_pool_backward():
    """ Test max_pool_backward function """
    x = np.random.randn(2, 5, 5, 3)
    pool_param = {'pool_width': 2, 'pool_height': 3, 'stride': [1,2,4,1]}
    dout = np.random.randn(2, 2, 1, 3)
    
    out, cache = max_pool_forward(x, pool_param)
    dx = max_pool_backward(dout, cache)
    
    dx_num = _eval_numerical_gradient_array(lambda x: max_pool_forward(x, pool_param)[0], x, dout)
    
    return _rel_error(dx, dx_num)