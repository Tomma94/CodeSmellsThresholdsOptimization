import argparse

def valid_int_positive(value):
    try:
        ivalue = int(value)
        if ivalue < 1:
            raise ValueError
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not integer or it out of the allowed range (>0)")


def valid_float_range(value):
    try:
        fvalue = float(value)
        if fvalue >= 1 or fvalue <= 0 :
            raise ValueError
        return fvalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not float or it out of the allowed range (0-1)")


def valid_int_positive_zero(value):
    try:
        ivalue = int(value)
        if ivalue < 0:
            raise ValueError
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not integer or it out of the allowed range (>=0)")
