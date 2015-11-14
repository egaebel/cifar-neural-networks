from math import floor

def single_layer_compute(width, height, kW, kH, dW, dH, padW, padH):
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

def convModule(width, height, bigMaxPool=False):
    width, height = single_layer_compute(width, height, 3, 3, 1, 1, 1, 1)
    if bigMaxPool:
        width, height = single_layer_compute(width, height, 2, 2, 2, 2, 0, 0)
    else:
        width, height = single_layer_compute(width, height, 2, 2, 1, 1, 0, 0)
    return width, height

width, height = 32, 32
width, height = convModule(width, height)
width, height = convModule(width, height, bigMaxPool=True)
print("Width: %d, Height: %d\n" % (width, height))

width, height = convModule(width, height)
width, height = convModule(width, height, bigMaxPool=True)
print("Width: %d, Height: %d\n" % (width, height))
