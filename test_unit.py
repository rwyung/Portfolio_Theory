# test_unit.py


#from math import abs

def test_equal(name,f,d):
    if f == d:
        return(f"Unit test {name} Passed")
    else:
        return(f"Unit test {name} Failed")

def test_abs_tol(name,f,d,tol):
    if abs(f - d) <= tol:
        return(f"Unit test {name} Passed; within {tol:0.05f}")
    else:
        return(f"Unit test {name} Failed; not within {tol:0.05f}")

def test_rel_tol(name,f,d,tol):
    if abs(f/d -1) <= tol:
        return(f"Unit test {name} Passed; within {tol:0.05f}")
    else:
        return(f"Unit test {name} Failed; not within {tol:0.05f}")

