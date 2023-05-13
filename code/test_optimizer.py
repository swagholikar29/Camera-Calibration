import numpy as np
from scipy.optimize import least_squares
def fun_rosenbrock(x):
    return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])
x0_rosenbrock = np.array([2, 2])
res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
print(res_1.x)


from scipy.spatial.transform import Rotation as R

r = R.from_rotvec([0, 0, np.pi])
r = R.from_matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
print(r.as_mrp().shape)
#r1 = R.from_mrp(r.as_mrp())
#print(r1.as_matrix())
#
#def myFun(arg1, arg2, arg3):
#    print("arg1:", arg1)
#    print("arg2:", arg2)
#    print("arg3:", arg3)
#
#
## Now we can use *args or **kwargs to
## pass arguments to this function :
#args = ("Geeks", "for", "Geeks")
#myFun(*args)
#
#kwargs = {"arg1": 1, "arg2": np.eye(3), "arg3": "Geeks"}
#myFun(**kwargs)
