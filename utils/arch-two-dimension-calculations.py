from math import floor

# Convolution layer
#owidth  = floor((width  + 2*padW - kW) / dW + 1)
#oheight = floor((height + 2*padH - kH) / dH + 1)

# Pooling Layer
#owidth  = op((width  + 2*padW - kW) / dW + 1)
#oheight = op((height + 2*padH - kH) / dH + 1)

# 7x7 s(2) convolution
nOutputPlane = 64
width = 32
height = 32
padW = 0
padH = 0
kW = 3
kH = 3
dW = 2
dH = 2
owidth  = floor((width  + 2 * padW - kW) / dW + 1)
oheight = floor((height + 2 * padH - kH) / dH + 1)
##################################################
print("owidth: " + str(owidth))
print("oheight: " + str(oheight))
print("\n")

# 3x3 s(2) max pooling
width = owidth
height = oheight
padW = 0
padH = 0
kW = 3
kH = 3
dW = 2
dH = 2
owidth  = floor((width  + 2 * padW - kW) / dW + 1)
oheight = floor((height + 2 * padH - kH) / dH + 1)
##################################################
print("owidth: " + str(owidth))
print("oheight: " + str(oheight))
print("\n")

# 1x1 s(1) convolution
nOutputPlane = 192
width = owidth
height = oheight
padW = 0
padH = 0
kW = 1
kH = 1
dW = 1
dH = 1
owidth  = floor((width  + 2 * padW - kW) / dW + 1)
oheight = floor((height + 2 * padH - kH) / dH + 1)
##################################################
print("owidth: " + str(owidth))
print("oheight: " + str(oheight))
print("\n")

# 3x3 s(1) convolution
nOutputPlane = 192
width = owidth
height = oheight
padW = 0
padH = 0
kW = 3
kH = 3
dW = 1
dH = 1
owidth  = floor((width  + 2 * padW - kW) / dW + 1)
oheight = floor((height + 2 * padH - kH) / dH + 1)
##################################################
print("owidth: " + str(owidth))
print("oheight: " + str(oheight))
print("\n")

# 3x3 s(2) max pooling
width = owidth
height = oheight
padW = 0
padH = 0
kW = 3
kH = 3
dW = 2
dH = 2
owidth  = floor((width  + 2 * padW - kW) / dW + 1)
oheight = floor((height + 2 * padH - kH) / dH + 1)
##################################################
print("owidth: " + str(owidth))
print("oheight: " + str(oheight))
print("\n")


# Inception module....
width = owidth
height = oheight
padW = 0
padH = 0
kW = 2
kH = 2
dW = 2
dH = 2
owidth  = floor((width  + 2 * padW - kW) / dW + 1)
oheight = floor((height + 2 * padH - kH) / dH + 1)
##################################################
print("owidth: " + str(owidth))
print("oheight: " + str(oheight))
print("\n")

width = owidth
height = oheight
padW = 0
padH = 0
kW = 5
kH = 5
dW = 1
dH = 1
owidth  = floor((width  + 2 * padW - kW) / dW + 1)
oheight = floor((height + 2 * padH - kH) / dH + 1)
nOutputPlane = 2
##################################################
width = owidth
height = oheight
padW = 0
padH = 0
kW = 2
kH = 2
dW = 2
dH = 2
owidth  = floor((width  + 2 * padW - kW) / dW + 1)
oheight = floor((height + 2 * padH - kH) / dH + 1)
##################################################
print("owidth: " + str(owidth))
print("oheight: " + str(oheight))
print("\n")