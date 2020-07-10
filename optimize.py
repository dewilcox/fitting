
# author: Daniel E. Wilcox daniel.wilcox@hexagon.com 
# company: Hexagon Manufacturing Intelligence
# copyright 2020
# distributed under the GPLv3 https://www.gnu.org/licenses/gpl-3.0.en.html



from __future__ import division, print_function
import numpy
import scipy.optimize


entropy = 0.1 * numpy.array([-0.419, -0.392, 0.811, 0.908, 0.341, -0.835, 0.699, 0.877, 0.184, 0.835, -0.326, 0.390, 0.354, 0.653, 0.228, -0.690, -0.290, 0.279, -0.121, 0.185])



def computeNumericDerivative(function, x, epsilon = 1e-5):
  assert(len(x.shape) == 1)

  def computeDerivativeOfIndex(i):
    delta = numpy.zeros((x.size,))
    delta[i] = epsilon
    y_p2 = function(x + 2*delta)
    y_p1 = function(x + delta)
    y_m1 = function(x - delta)
    y_m2 = function(x - 2*delta)
    return (-y_p2 + 8*y_p1 - 8*y_m1 + y_m2) / (12.0 * epsilon)

  return numpy.stack(list(map(computeDerivativeOfIndex, range(x.size))), axis=-1)


def checkDerivativeCore(vectorFunction, functionJacobian, x0, differentiate=computeNumericDerivative, delta=1e-4):
  assert(len(x0.shape) == 1)
  jacobian = functionJacobian(x0)
  numericJacobian = differentiate(vectorFunction, x0)
  if jacobian.shape != numericJacobian.shape:
    print('jacobian shape = ' + str(jacobian.shape))
    print('numeric jacobian shape = ' + str(numericJacobian.shape))
  absoluteError = numpy.amax(numpy.abs(jacobian - numericJacobian))
  # I've checked and made sure numpy.linalg.norm works well with mpmath objects
  fractionalError = numpy.linalg.norm(jacobian - numericJacobian) / ( 1e-16 + numpy.linalg.norm(jacobian) )
  if absoluteError >= delta and fractionalError >= delta:
    print('absoluteError = ', absoluteError)
    print('fractionalError = ', fractionalError)
    print('delta = ', delta)
    print('jacobian = ', jacobian)
    print('numericJacobian = ', numericJacobian)
    assert(fractionalError < delta or absoluteError < delta)


def checkDerivative(vectorFunction, functionJacobian, x0):
  assert(len(x0.shape) == 1)
  x1 = x0 + entropy[:x0.size]
  checkDerivativeCore(vectorFunction, functionJacobian, x0)
  checkDerivativeCore(vectorFunction, functionJacobian, x1)



def __optimize(optimizationProblem, x0):

  def objective(x):
    return optimizationProblem['objective'](x)

  def objective_dx(x):
    return optimizationProblem['objective_dx'](x)

  def equalityConstraint(x):
    return optimizationProblem['equalityConstraint'](x)

  def equalityConstraint_dx(x):
    return optimizationProblem['equalityConstraint_dx'](x)

  def inequalityConstraint(x):
    return optimizationProblem['inequalityConstraint'](x)

  def inequalityConstraint_dx(x):
    return optimizationProblem['inequalityConstraint_dx'](x)

  # check the derivatives of things
  checkDerivative(objective, objective_dx, x0)

  # what are the bounds?
  bounds = [(optimizationProblem['lowerBounds'][i], optimizationProblem['upperBounds'][i]) for i in range(x0.size)]

  # construct the keyword arguments
  kwargs = {
    'fprime': objective_dx,
    'bounds': bounds,
    'acc': 1e-16,
    'iprint': 0,
  }

  # is there an equality constraint?
  if 'equalityConstraint' in optimizationProblem:
    kwargs['f_eqcons'] = equalityConstraint
    kwargs['fprime_eqcons'] = equalityConstraint_dx
    checkDerivative(equalityConstraint, equalityConstraint_dx, x0)

  # is there an inequality constraint?
  if 'inequalityConstraint' in optimizationProblem:
    kwargs['f_ieqcons'] = inequalityConstraint
    kwargs['fprime_ieqcons'] = inequalityConstraint_dx
    checkDerivative(inequalityConstraint, inequalityConstraint_dx, x0)

  # solve the optimization problem
  bestX = scipy.optimize.fmin_slsqp(objective, x0, **kwargs)

  # all done
  return bestX




def optimizeWithMultiStart(optimizationProblem, x0):

  assert(len(x0.shape) == 1)
  n = x0.size
  x0_list = [x0, x0+entropy[:n], x0+entropy[n:(2*n)], x0+entropy[(2*n):(3*n)]]

  bestX_list = [__optimize(optimizationProblem, this_x0) for this_x0 in x0_list]
  objectives = [optimizationProblem['objective'](x) for x in bestX_list]
  for i in range(len(x0_list)):
    x = bestX_list[i]
    if 'equalityConstraint' in optimizationProblem:
      equalityConstraint = optimizationProblem['equalityConstraint'](x)
      if numpy.amax(numpy.abs(equalityConstraint)) > 1e-7:
        objectives[i] = 1e99
    if 'inequalityConstraint' in optimizationProblem:
      inequalityConstraint = optimizationProblem['inequalityConstraint'](x)
      if numpy.abs(numpy.amin(inequalityConstraint)) > 1e-7:
        objectives[i] = 1e99
    if numpy.any(numpy.logical_not(numpy.isfinite(x))):
      objectives[i] = 1e99
    if numpy.any(numpy.logical_or(x < optimizationProblem['lowerBounds'] - 1e-8, x > optimizationProblem['upperBounds'] + 1e-8)):
      objectives[i] = 1e99

  objectives = numpy.array(objectives)
  assert(numpy.amin(objectives) < 1e99)

  return bestX_list[numpy.argmin(objectives)]



def solveLeastSquares(deviationsProblem, x0):

  def objective(x):
    devs = deviationsProblem['deviations'](x)
    return 0.5 * numpy.sum(devs**2)

  def objective_dx(x):
    devs = deviationsProblem['deviations'](x)
    devs_dx = deviationsProblem['deviations_dx'](x)
    return numpy.dot(devs, devs_dx)

  # create the optimization problem
  optimizationProblem = {
    'objective': objective,
    'objective_dx': objective_dx,
    'lowerBounds': deviationsProblem['lowerBounds'],
    'upperBounds': deviationsProblem['upperBounds'],
  }

  # optimize and return
  return optimizeWithMultiStart(optimizationProblem, x0)




def solveMinMax(deviationsProblem, x0):
  assert(len(x0.shape) == 1)

  def objective(x):
    return x[-1]

  def objective_dx(x):
    result = numpy.zeros((x.size,))
    result[-1] = 1.0
    return result

  def inequalityConstraint(x):
    devs = deviationsProblem['deviations'](x[:-1])
    N = devs.size
    result = numpy.zeros((2*N,))
    result[:N] = x[-1] - devs
    result[N:] = x[-1] + devs
    return result

  def inequalityConstraint_dx(x):
    devs_dx = deviationsProblem['deviations_dx'](x[:-1])
    N = devs_dx.shape[0]
    result = numpy.zeros((2*N,x.size))
    result[:, -1] = 1.0
    result[:N, :-1] = -devs_dx
    result[N:, :-1] = devs_dx
    return result

  # create the optimization problem
  optimizationProblem = {
    'objective': objective,
    'objective_dx': objective_dx,
    'inequalityConstraint': inequalityConstraint,
    'inequalityConstraint_dx': inequalityConstraint_dx,
    'lowerBounds': numpy.concatenate((deviationsProblem['lowerBounds'], [-10.0])),
    'upperBounds': numpy.concatenate((deviationsProblem['upperBounds'], [1000.0])),
  }

  startX0 = numpy.zeros((x0.size+1,))
  startX0[:-1] = x0

  # optimize and return
  augmentedResult = optimizeWithMultiStart(optimizationProblem, startX0)
  return augmentedResult[:-1]




def solveConstrainedLeastSquares(deviationsProblem, x0):

  def objective(x):
    devs = deviationsProblem['deviations'](x)
    return 0.5 * numpy.sum(devs**2)

  def objective_dx(x):
    devs = deviationsProblem['deviations'](x)
    devs_dx = deviationsProblem['deviations_dx'](x)
    return numpy.dot(devs, devs_dx)

  def inequalityConstraint(x):
    return -deviationsProblem['deviations'](x)

  def inequalityConstraint_dx(x):
    return -deviationsProblem['deviations_dx'](x)

  # create the optimization problem
  optimizationProblem = {
    'objective': objective,
    'objective_dx': objective_dx,
    'inequalityConstraint': inequalityConstraint,
    'inequalityConstraint_dx': inequalityConstraint_dx,
    'lowerBounds': deviationsProblem['lowerBounds'],
    'upperBounds': deviationsProblem['upperBounds'],
  }

  # optimize and return
  return optimizeWithMultiStart(optimizationProblem, x0)




def solveConstrainedMinMax(deviationsProblem, x0):
  assert(len(x0.shape) == 1)

  def objective(x):
    return x[-1]

  def objective_dx(x):
    result = numpy.zeros((x.size,))
    result[-1] = 1.0
    return result

  def inequalityConstraint(x):
    devs = deviationsProblem['deviations'](x[:-1])
    N = devs.size
    result = numpy.zeros((2*N,))
    result[:N] = -devs
    result[N:] = x[-1] + devs
    return result

  def inequalityConstraint_dx(x):
    devs_dx = deviationsProblem['deviations_dx'](x[:-1])
    N = devs_dx.shape[0]
    result = numpy.zeros((2*N,x.size))
    result[:N, :-1] = -devs_dx
    result[N:, -1] = 1.0
    result[N:, :-1] = devs_dx
    return result

  # create the optimization problem
  optimizationProblem = {
    'objective': objective,
    'objective_dx': objective_dx,
    'inequalityConstraint': inequalityConstraint,
    'inequalityConstraint_dx': inequalityConstraint_dx,
    'lowerBounds': numpy.concatenate((deviationsProblem['lowerBounds'], [-10.0])),
    'upperBounds': numpy.concatenate((deviationsProblem['upperBounds'], [1000.0])),
  }

  startX0 = numpy.zeros((x0.size+1,))
  startX0[:-1] = x0

  # optimize and return
  augmentedResult = optimizeWithMultiStart(optimizationProblem, startX0)
  return augmentedResult[:-1]
