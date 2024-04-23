import numpy as np
from threeML.utils.OGIP.response import InstrumentResponse
from astropy.io import fits


class POLARData(object):

    def __init__(self, polar_events, polar_specrsp=None, polar_polrsp=None, reference_time=0.):
        """
        container class that converts raw POLAR fits data into useful python
        variables

        This can build both the polarimetric and spectral data
        
        :param polar_events: path to polar event file
        :param polar_specrsp: path to polar spectral responce file
        :param polar_polrsp: path to polar polarimetric responce file
                             it will use SABOUNDS to bin you SA data in 'polar_events'
                             if 'NONE', we assume 'SA' data is already binned
        :param reference_time: reference time of the events (in SECOND)

        """

        with fits.open(polar_specrsp) as hdu_spec:

            # This gets the spectral response
            mc_low = hdu_spec['MATRIX'].data.field('ENERG_LO')
            mc_high = hdu_spec['MATRIX'].data.field('ENERG_HI')
            ebounds = np.append(mc_low, mc_high[-1])
            matrix = hdu_spec['MATRIX'].data.field('MATRIX')
            matrix = matrix.transpose()

            # build the POLAR response
            mc_energies = np.append(mc_low, mc_high[-1])
            self._rsp = InstrumentResponse(matrix=matrix, ebounds=ebounds, monte_carlo_energies=mc_energies)

        with fits.open(polar_events) as hdu_evt:

            # open the event file

            # extract the pedestal corrected ADC channels
            # which are non-integer and possibly
            # less than zero
            
            # Extract mission and instrument info
            self._mission = hdu_evt['POLEVENTS'].header['TELESCOP']
            self._instrument = hdu_evt['POLEVENTS'].header['INSTRUME']

            pha = hdu_evt['POLEVENTS'].data.field('ENERGY')

            # non-zero ADC channels are invalid
            idx = pha >= 0
            #pha = pha[idx]

            idx2 = (pha <= ebounds.max()) & (pha >= ebounds.min())

            pha = pha[idx2 & idx]

            # get the dead time fraction
            self._dead_time_fraction = (hdu_evt['POLEVENTS'].data.field('DEADFRAC'))[idx & idx2]

            # get the arrival time, in SECOND
            self._time = (hdu_evt['POLEVENTS'].data.field('TIME'))[idx & idx2] - reference_time

            # digitize the ADC channels into bins
            # these bins are preliminary

            # now do the scattering angles

            scattering_angles = hdu_evt['POLEVENTS'].data.field('SA')

            # clear the bad scattering angles
            idx = scattering_angles != -1

            self._scattering_angle_time = (hdu_evt['POLEVENTS'].data.field('TIME'))[idx] - reference_time
            self._scattering_angle_dead_time_fraction = (hdu_evt['POLEVENTS'].data.field('DEADFRAC'))[idx]
            self._scattering_angles = scattering_angles[idx]

        # bin the ADC channels
        self._binned_pha = np.digitize(pha, ebounds)

        # bin the scattering_angles

        if polar_polrsp is not None:

            with fits.open(polar_polrsp) as hdu_pol:
                samin = hdu_pol['SABOUNDS'].data.field('SA_MIN')
                samax = hdu_pol['SABOUNDS'].data.field('SA_MAX')
                scatter_bounds = np.append(samin, samax[-1])

            self._scattering_bins = scatter_bounds
            self._binned_scattering_angles = np.digitize(self._scattering_angles, scatter_bounds)

        else:

            self._scattering_bins = None
            self._binned_scattering_angles = None


    @property
    def mission(self):
        return self._mission

    @property
    def instrument(self):
        return self._instrument
    
    @property
    def pha(self):
        return self._binned_pha

    @property
    def time(self):
        return self._time

    @property
    def dead_time_fraction(self):
        return self._dead_time_fraction

    @property
    def rsp(self):
        return self._rsp

    @property
    def n_channels(self):

        return len(self._rsp.ebounds) - 1

    @property
    def scattering_angles(self):

        return self._binned_scattering_angles

    @property
    def scattering_angle_time(self):

        return self._scattering_angle_time

    @property
    def scattering_angle_dead_time_fraction(self):
        return self._scattering_angle_dead_time_fraction

    @property
    def n_scattering_bins(self):

        return len(self._scattering_bins) - 1

    @property
    def scattering_edges(self):

        return self._scattering_bins
