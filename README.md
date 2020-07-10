# Fitting

Computes best-fit cylinders, spheres, and planes. The best fit algorithm is least squares, min max, constrained least squares, or constrained min max. Using this library looks like this:

`fit.fitCylinder(nominalCylinder, surfacePoints, optimize.solveLeastSquares)`

The first parameter is the nominal shape. It is a Python dictionary. You can understand the expected fields and their expected format by reading fit.py. For the vector field, ensure the resulting vector is normalized. For the materialSign field, use +1.0 for outer features, and -1.0 for inner features. The second parameter is the surface points, which should be an Nx3 array in the numpy format. The third parameter is the solver you want to use, which can be found in the optimize.py file.

The sign of the deviations follows the following convention:
* The fitted shape has the same parity as the nominal surface. That means if the nominal surface is inner (a hole), then the fitted surface is also inner (a hole).
* Surface points that are external to the imaginary material of the fitted shape have positive deviations.
* Constrained fits are external to the actual material; that means the measured points are all internal to the imaginary material of the fitted shape.
*	Therefore, the deviations for an external-to-material constrained fit are all negative.
