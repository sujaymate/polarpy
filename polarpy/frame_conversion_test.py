#!/use/bin/env python

import numpy as np
from astropy.coordinates import SkyCoord

RA_X = 0
Dec_X = 0

RA_Z = 0
Dec_Z = 90

RA_S = 45
Dec_S = 45


def celestial_to_local_tangent(RA_X, Dec_X, RA_Z, Dec_Z, RA_S, Dec_S):

    # create skycoord objects and convert to the cartesian frame
    X = SkyCoord(RA_X, Dec_X, frame='icrs', unit='deg', obstime='J2000').cartesian
    Z = SkyCoord(RA_Z, Dec_Z, frame='icrs', unit='deg', obstime='J2000').cartesian
    S = SkyCoord(RA_S, Dec_S, frame='icrs', unit='deg', obstime='J2000').cartesian

    # Get the instrument Y axis
    Y = Z.cross(X)

    # Matrix to go from XYZ to J2000 frame
    R_XYZ_J2000 = np.array([X.get_xyz().value, Y.get_xyz().value, Z.get_xyz().value]).T

    # Compute source theta, phi
    # Compute the projection on XYZ
    ux = X.dot(S).value
    uy = Y.dot(S).value
    uz = Z.dot(S).value

    # Compute the theta,phi
    theta = np.arccos(uz)
    phi = np.arctan2(uy, ux)
    if phi < 0:
        phi += 2*np.pi

    # Matrix to go from NED to XYZ
    R_NED_XYZ = np.array([[-np.cos(theta) * np.cos(phi), -np.sin(phi), -np.sin(theta) * np.cos(phi)],
                          [-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
                          [np.sin(phi), 0, -np.cos(theta)]])

    # Matrix to go from NED to J2000
    R_NED_J2000 = np.matmul(R_XYZ_J2000, R_NED_XYZ)

    # Compute the PA offset. This is basically azimuth of NED Z-axis in J2000
    Z_NED_J2000 = np.matmul(R_NED_J2000, [0, 0, 1])
    psi = np.arctan2(Z_NED_J2000[1], Z_NED_J2000[0])

    # Return always between 0 to 360
    if psi < 0:
        psi += 2*np.pi

    return psi


psi = celestial_to_local_tangent(RA_X, Dec_X, RA_Z, Dec_Z, RA_S, Dec_S)
print(np.rad2deg(psi % np.pi))
