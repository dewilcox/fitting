
# author: Daniel E. Wilcox daniel.wilcox@hexagon.com 
# company: Hexagon Manufacturing Intelligence
# copyright 2020
# distributed under the GPLv3 https://www.gnu.org/licenses/gpl-3.0.en.html



from __future__ import division, print_function
import numpy



def makeOrthogonalVector(a):
  minIndex = numpy.argmin(numpy.abs(a))
  b = 0.0 * a
  b[minIndex] = 1.0
  result = numpy.cross(a, b)
  return result / numpy.linalg.norm(result)



def computeCylinderDeviations(cylinder, data):
  assert(len(data.shape) == 2)
  assert(data.shape[1] == 3)
  assert(numpy.abs(cylinder['materialSign']) == 1)
  delta = data - cylinder['center']
  Delta = delta - numpy.sum(delta * cylinder['vector'], axis=1)[:, None] * cylinder['vector']
  DeltaNorm = numpy.linalg.norm(Delta, axis=1)
  d = cylinder['materialSign']*(DeltaNorm - 0.5 * cylinder['size'])
  assert(len(d.shape) == 1)
  return d



def fitCylinder(nominal, data, solve):
  assert(len(data.shape) == 2)
  assert(data.shape[1] == 3)
  assert(numpy.abs(nominal['materialSign']) == 1)

  # shift the center from the center-of-mass convention
  meanData = numpy.mean(data, axis=0)
  nominalMean = nominal['center'] - nominal['vector'] * numpy.dot(nominal['vector'], nominal['center'] - meanData)

  # construct the coordinate system
  zHat = nominal['vector']
  xHat = makeOrthogonalVector(zHat)
  yHat = numpy.cross(zHat, xHat)

  def deviations(x):
    center = nominalMean + x[0] * xHat + x[1] * yHat
    axis = nominal['vector'] + x[2] * xHat + x[3] * yHat
    axis[:] /= numpy.linalg.norm(axis)
    radius = 0.5 * nominal['size'] + x[4]

    delta = data - center
    Delta = delta - numpy.sum(delta * axis, axis=1)[:, None] * axis
    DeltaNorm = numpy.linalg.norm(Delta, axis=1)
    d = nominal['materialSign']*(DeltaNorm - radius)
    assert(len(d.shape) == 1)
    return d

  def deviations_dx(x):
    center = nominalMean + x[0] * xHat + x[1] * yHat
    rawVector = nominal['vector'] + x[2] * xHat + x[3] * yHat
    rawVectorNorm = numpy.linalg.norm(rawVector)
    axis = rawVector / rawVectorNorm

    delta = data - center
    P = numpy.eye(3) - numpy.outer(axis, axis)
    Delta = numpy.dot(delta, P)
    DeltaNorm = numpy.linalg.norm(Delta, axis=1)

    deviations_dDelta = nominal['materialSign'] * Delta / DeltaNorm[:, None]
    deviations_ddelta = numpy.dot(deviations_dDelta, P)
    deviations_daxis = -delta * numpy.dot(deviations_dDelta, axis)[:, None] - deviations_dDelta * numpy.dot(delta, axis)[:, None]

    gradient = numpy.zeros((DeltaNorm.size, 5))
    gradient[:, 0] = -numpy.dot(deviations_ddelta, xHat)
    gradient[:, 1] = -numpy.dot(deviations_ddelta, yHat)
    deviations_drawVector = numpy.dot(deviations_daxis, P/rawVectorNorm)
    gradient[:, 2] = numpy.dot(deviations_drawVector, xHat)
    gradient[:, 3] = numpy.dot(deviations_drawVector, yHat)
    gradient[:, 4] = -nominal['materialSign']
    return gradient

  deviationsStructure = {
    'lowerBounds': numpy.array([-1, -1, -0.5, -0.5, -0.2 * nominal['size']]),
    'upperBounds': numpy.array([ 1,  1,  0.5,  0.5,  0.2 * nominal['size']]),
    'deviations': deviations,
    'deviations_dx': deviations_dx,
  }
  bestX = solve(deviationsStructure, numpy.zeros((5,)))

  result = {}
  result['center'] = nominalMean + bestX[0] * xHat + bestX[1] * yHat
  result['vector'] = nominal['vector'] + bestX[2] * xHat + bestX[3] * yHat
  result['vector'][:] /= numpy.linalg.norm(result['vector'])
  result['size'] = nominal['size'] + 2.0 * bestX[4]
  result['materialSign'] = nominal['materialSign']
  result['length'] = nominal['length']
  result['deviations'] = deviations(bestX)
  delta = data - result['center']
  z = numpy.dot(delta, result['vector'])[:, None]
  Delta = delta - result['vector'] * z
  DeltaHat = Delta / numpy.linalg.norm(Delta, axis=1)[:, None]
  surfaceNormals = result['materialSign'] * DeltaHat
  result['surfaceVectors'] = surfaceNormals
  result['dataPoints'] = data

  # convert the center to the halfway-between-the-end-planes convention
  midZ = 0.5 * (numpy.amin(z) + numpy.amax(z))
  result['center'] = result['center'] + midZ * result['vector']

  # double-check the halfway-between-the-end-planes convention
  delta = data - result['center']
  z = numpy.dot(delta, result['vector'])[:, None]
  assert(numpy.abs(numpy.amin(z) + numpy.amax(z)) < 1e-12 * (numpy.abs(numpy.amin(z)) + numpy.abs(numpy.amax(z))))

  # all done
  return result






def computeSphereDeviations(sphere, data):
  assert(len(data.shape) == 2)
  assert(data.shape[1] == 3)
  assert(numpy.abs(sphere['materialSign']) == 1)
  delta = data - sphere['center']
  deltaNorm = numpy.linalg.norm(delta, axis=1)
  d = sphere['materialSign']*(0.5 * sphere['size'] - deltaNorm)
  assert(len(d.shape) == 1)
  return d




def fitSphere(nominal, data, solve):
  assert(len(data.shape) == 2)
  assert(data.shape[1] == 3)
  assert(numpy.abs(nominal['materialSign']) == 1)

  def deviations(x):
    center = nominal['center'] + x[:3]
    radius = 0.5 * nominal['size'] + x[3]
    delta = data - center
    deltaNorm = numpy.linalg.norm(delta, axis=1)
    d = nominal['materialSign']*(deltaNorm - radius)
    assert(len(d.shape) == 1)
    return d

  def deviations_dx(x):
    center = nominal['center'] + x[:3]
    delta = data - center
    deltaNorm = numpy.linalg.norm(delta, axis=1)
    deviations_ddelta = nominal['materialSign'] * delta / deltaNorm[:, None]
    gradient = numpy.zeros((deltaNorm.size,4), dtype=object)
    gradient[:, :3] = -deviations_ddelta
    gradient[:, 3] = -nominal['materialSign']
    return gradient

  deviationsStructure = {
    'lowerBounds': numpy.array([-1, -1, -1, -0.2 * nominal['size']]),
    'upperBounds': numpy.array([ 1,  1,  1,  0.2 * nominal['size']]),
    'deviations': deviations,
    'deviations_dx': deviations_dx,
  }
  bestX = solve(deviationsStructure, numpy.zeros((4,)))

  result = {}
  result['center'] = nominal['center'] + bestX[:3]
  result['size'] = nominal['size'] + 2 * bestX[3]
  result['materialSign'] = nominal['materialSign']
  result['deviations'] = deviations(bestX)
  delta = data - result['center']
  deltaHat = delta / numpy.linalg.norm(delta, axis=1)[:, None]
  surfaceNormals = result['materialSign'] * deltaHat
  result['surfaceVectors'] = surfaceNormals
  result['dataPoints'] = data

  assert(numpy.amax(numpy.abs(result['deviations'])) < 0.125 * nominal['size'])

  return result






def fitPlane(nominal, data, solve):
  assert(len(data.shape) == 2)
  assert(data.shape[1] == 3)

  zHat = nominal['vector']
  yHat = numpy.cross(zHat, nominal['xHat'])
  yHat[:] /= numpy.linalg.norm(yHat)
  xHat = numpy.cross(yHat, zHat)
  
  def deviations(x):
    center = nominal['center'] + x[0] * zHat
    vector = nominal['vector'] + x[1] * xHat + x[2] * yHat
    vector[:] /= numpy.linalg.norm(vector)
    delta = data - center
    d = numpy.dot(delta, vector)
    assert(len(d.shape) == 1)
    return d

  def deviations_dx(x):
    center = nominal['center'] + x[0] * zHat
    rawVector = nominal['vector'] + x[1] * xHat + x[2] * yHat
    rawVectorNorm = numpy.linalg.norm(rawVector)
    vector = rawVector / rawVectorNorm
    P = numpy.eye(3) - numpy.outer(vector, vector)
    delta = data - center
    d_drawVector = numpy.dot(delta, P/rawVectorNorm)

    dGradient = numpy.zeros((data.shape[0], 3), dtype=object)
    dGradient[:, 0] = numpy.dot(-vector, zHat)
    dGradient[:, 1] = numpy.dot(d_drawVector, xHat)
    dGradient[:, 2] = numpy.dot(d_drawVector, yHat)

    return dGradient

  deviationsStructure = {
    'lowerBounds': numpy.array([-1, -1, -1]),
    'upperBounds': numpy.array([ 1,  1,  1]),
    'deviations': deviations,
    'deviations_dx': deviations_dx,
  }
  bestX = solve(deviationsStructure, numpy.zeros((3,)))

  result = {}
  result['center'] = nominal['center'] + bestX[0] * zHat
  result['vector'] = nominal['vector'] + bestX[1] * xHat + bestX[2] * yHat
  result['vector'] /= numpy.linalg.norm(result['vector'])
  result['xHat'] = numpy.cross(yHat, result['vector'])
  result['xHat'] /= numpy.linalg.norm(result['xHat'])
  result['deviations'] = deviations(bestX)
  result['surfaceVectors'] = 0.0 * data + result['vector'][None, :]
  result['dataPoints'] = data

  # convert the center to the center-of-mass convention
  meanData = numpy.mean(data, axis=0)
  result['center'] = meanData + result['vector'] * numpy.dot(result['vector'], result['center'] - meanData)

  return result

