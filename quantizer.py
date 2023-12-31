import numpy as np
import cupy as cp


def stochastic_rounding(x):
    # takes the integer part of the number
    x_int = x.astype(cp.int32)
    # takes the fractional part
    x_frac = cp.abs(x - x_int)

    # generate a random number
    rng = cp.random.random(x_int.shape)

    # if the frac is grater... for positive cases
    rounded_pos = cp.where(x_frac > rng, x_int + 1, x_int)

    # if the grac is greate... for negative cases
    rounded_neg = cp.where(x_frac > rng, x_int - 1, x_int)

    # select the rounded according to the signal
    rounded = cp.where(x < 0, rounded_neg, rounded_pos)
    
    return rounded


def quantize(x, stochastic_round = True, stochastic_zero = True):
    """ exponentiation and quantization function """

    # just to avoid numerical problems
    eps = 1e-8

    # extract the signal
    s = cp.sign(x)

    # takes the abs
    abs_x = cp.abs(x)

    cliped_abs_x = cp.where(abs_x < eps, eps, abs_x) # clip the min value of abs. (this is just for avoid numercal problems)
    cliped_abs_x = cp.where(cliped_abs_x > 1, 1, cliped_abs_x) # clip the max value of DN 

    # gets the exponent with base 2
    exp = cp.log2(cliped_abs_x)

    # round to nearest and cast to int (use stochastic rounding)
    if stochastic_round:
        round_exp = stochastic_rounding(exp)
    else:
        round_exp = (cp.round(exp)).astype(cp.int32)

    
    if stochastic_zero:
        ###################
        # stochastic zero
        # detect underflow
        underflow = cp.where(round_exp < -7, 1, 0)
        # clip expoent in -7
        clip_exp = cp.where(underflow, -7, round_exp)    
        # randomize the signal
        s = cp.where(cp.logical_and(cp.random.random(round_exp.shape) < 0.5, underflow), -s, s) 
        # convert to float32 again
        qx = s * cp.power(2., clip_exp)
        ###################
    else:
        ###################
        # fixed zero    
        # detect underflow
        underflow = cp.where(round_exp < -7, 1, 0)
        
        # convert to float32 again
        qx = s * cp.power(2., round_exp)
        
        # fixed zero
        qx = cp.where(underflow, 0, qx)

    return qx





def quantize_po2(x):
    """ x must be positive and greater than 1e-38"""

    eps = cp.array(1e-38)
    
    # clip the value to be quantized to avoid numerical under/overflow
    cx = cp.where(x < eps, eps, x)    

    qx = cp.power(2., cp.ceil(cp.log2(cx)))
    
    return qx # bypass