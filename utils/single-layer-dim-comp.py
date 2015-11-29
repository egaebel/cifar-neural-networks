from math import floor

def single_layer_compute(width, height, kW, kH, dW, dH, padW=0, padH=0):
    # width = 28
    # height = 28
    # kW = 3
    # kH = 3
    # dW = 1
    # dH = 1
    # padW = 0
    # padH = 0
    owidth  = floor((width  + 2 * padW - kW) / dW + 1)
    oheight = floor((height + 2 * padH - kH) / dH + 1)
    return owidth, oheight

def inception_layer_dims(width, height):
    column1 = (width, height)
    width, height = single_layer_compute(width, height, 3, 3, 1, 1, 0, 0)
    column2 = (width, height)
    width, height = single_layer_compute(width, height, 5, 5, 1, 1, 0, 0)
    column3 = (width, height)
    width, height = single_layer_compute(width, height, 3, 3, 1, 1, 0, 0)
    column4 = (width, height)
    return [column1, column2, column3, column4]

def max_pool_55_11(width, height):
    return single_layer_compute(width, height, 5, 5, 1, 1, 0, 0)

def max_pool_33_11(width, height):
    return single_layer_compute(width, height, 3, 3, 1, 1, 0, 0)

def max_pool_33_22(width, height):
    return single_layer_compute(width, height, 3, 3, 2, 2, 0, 0)

def max_pool_33_33(width, height):
    return single_layer_compute(width, height, 3, 3, 3, 3, 0, 0)

def max_pool_22_11(width, height):
    return single_layer_compute(width, height, 2, 2, 1, 1, 0, 0)

def max_pool_22_22(width, height):
    return single_layer_compute(width, height, 2, 2, 2, 2, 0, 0)

def global_max_pool(width, height, pool_width, pool_height):
    return single_layer_compute(width, height, 
        pool_width, pool_height, 
        pool_width, pool_height, 
        0, 0)

width = 32
height = 32

#def single_layer_compute(width, height, kW, kH, dW, dH, padW=0, padH=0):

width, height = max_pool_33_22(width, height)
width, height = max_pool_33_33(width, height)
width, height = max_pool_33_33(width, height)


print("width: %f height: %f" % (width, height))
