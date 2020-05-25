#-------------------------------------------------------------------------------
# Name:        pySaliencyMapDefs
# Purpose:     Definitions for class pySaliencyMap
#
# Author:      Akisato Kimura <akisato@ieee.org>
#
# Created:     April 24, 2014
# Copyright:   (c) Akisato Kimura 2014-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------

# parameters for computing optical flows using the Gunner Farneback's algorithm
farne_pyr_scale = 0.5
farne_levels = 3
farne_winsize = 15
farne_iterations = 3
farne_poly_n = 5
farne_poly_sigma = 1.2
farne_flags = 0

# parameters for detecting local maxima
default_step_local = 16

# feature weights
weight_intensity   = 1
weight_color       = 1
weight_orientation = 1
weight_motion      = 1

# coefficients of Gabor filters
GaborKernel_0 = [\
    [ 0, 0],\
    [ 0, 0]\
]
GaborKernel_45 = [\
    [  0,  0],\
    [  0,  0]\
]
GaborKernel_90 = [\
    [  0,  0],\
    [  0,  0]\
]
GaborKernel_135 = [\
    [ 0, 0],\
    [ 0, 0]\
]
