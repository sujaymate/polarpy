import numpy as np
from threeML.utils.OGIP.response import InstrumentResponse
import h5py
from astropy.io import fits


class POLARData(object):

    def __init__(self, polar_events, polar_specrsp=None, polar_polrsp=None, input_format='fits' , reference_time=0.):
        """
        container class that converts raw POLAR fits data into useful python
        variables

        This can build both the polarimetric and spectral data
        
        :param polar_events: path to polar event file
                             if 'hdf5', 'polar_events' contains 'specrsp' already
        :param polar_specrsp: path to polar spectral responce file
                             if 'hdf5', no need to put 'polar_specrsp'
        :param polar_polrsp: path to polar polarimetric responce file
                             if need to use bins' defination in 'polrsp' to bin you scatter_angle
        :param input_format:  input files's format, either 'fits' or 'hdf5'
                             specify that for the current two formats of polar data
        :param reference_time: reference time of the events (tunix)

        """

        if input_format == 'fits':

            with fits.open(polar_specrsp) as hdu_spec:

                # This gets the spectral response
                mc_low = hdu_spec['MATRIX'].data.field('ENERG_LO')
                mc_high = hdu_spec['MATRIX'].data.field('ENERG_HI')
                ebounds = np.append(mc_low, mc_high[-1])
                matrix = hdu_spec['MATRIX'].data.field('MATRIX')
                matrix = matrix.transpose()

            with fits.open(polar_events) as hdu_evt:

                # open the event file

                # extract the pedestal corrected ADC channels
                # which are non-integer and possibly
                # less than zero

                pha = hdu_evt['EVENTS'].data.field('ENERGY')

                # non-zero ADC channels are invalid
                idx = pha >= 0
                #pha = pha[idx]

                idx2 = (pha <= ebounds.max()) & (pha >= ebounds.min())

                pha = pha[idx2 & idx]

                # get the dead time fraction
                self._dead_time_fraction = (hdu_evt['EVENTS'].data.field('DRATIO'))[idx & idx2]

                # get the arrival time, in tunix of the events
                self._time = (hdu_evt['EVENTS'].data.field('TUNIX'))[idx & idx2] - reference_time

                # digitize the ADC channels into bins
                # these bins are preliminary

                # now do the scattering angles

                scattering_angles = hdu_evt['EVENTS'].data.field('SANGLE')

                # clear the bad scattering angles
                idx = scattering_angles != -1

                self._scattering_angle_time = (hdu_evt['EVENTS'].data.field('TUNIX'))[idx] - reference_time
                self._scattering_angle_dead_time_fraction = (hdu_evt['EVENTS'].data.field('DRATIO'))[idx]
                self._scattering_angles = scattering_angles[idx]


            # bin the scattering_angles

            if polar_polrsp is not None:

                with fits.open(polar_polrsp) as hdu_pol:

                    scatter_bounds = hdu_pol['INSAVALS'].data.field('SA_IN')

                self._scattering_bins = scatter_bounds
                self._binned_scattering_angles = np.digitize(self._scattering_angles, scatter_bounds)

            else:

                self._scattering_bins = None
                self._binned_scattering_angles = None


        if input_format == 'h5':

            with h5py.File(polar_events, 'r') as f:

                # This gets the spectral response
                rsp_grp = f['rsp']

                matrix = rsp_grp['matrix'][()]
                ebounds = rsp_grp['ebounds'][()]
                mc_low = rsp_grp['mc_low'][()]
                mc_high = rsp_grp['mc_high'][()]

                # open the event file

                # extract the pedestal corrected ADC channels
                # which are non-integer and possibly
                # less than zero
                pha = f['energy'][()]

                # non-zero ADC channels are invalid
                idx = pha >= 0
                #pha = pha[idx]

                idx2 = (pha <= ebounds.max()) & (pha >= ebounds.min())

                pha = pha[idx2 & idx]

                # get the dead time fraction
                self._dead_time_fraction = (f['dead_ratio'][()])[idx & idx2]

                # get the arrival time, in tunix of the events
                self._time = (f['time'][()])[idx & idx2] - reference_time

                # digitize the ADC channels into bins
                # these bins are preliminary

                # now do the scattering angles

                scattering_angles = f['scatter_angle'][()]

                # clear the bad scattering angles
                idx = scattering_angles != -1

                self._scattering_angle_time = (f['time'][()])[idx] - reference_time
                self._scattering_angle_dead_time_fraction = (f['dead_ratio'][()])[idx]
                self._scattering_angles = scattering_angles[idx]

            # bin the scattering_angles

            if polar_polrsp is not None:

                with h5py.File(polar_polrsp, 'r') as f:

                    scatter_bounds = f['bins'][()]

                self._scattering_bins = scatter_bounds
                self._binned_scattering_angles = np.digitize(self._scattering_angles, scatter_bounds)

            else:

                self._scattering_bins = None
                self._binned_scattering_angles = None

        # build the POLAR response

        mc_energies = np.append(mc_low, mc_high[-1])

        self._rsp = InstrumentResponse(matrix=matrix, ebounds=ebounds, monte_carlo_energies=mc_energies)

        # bin the ADC channels

        self._binned_pha = np.digitize(pha, ebounds)

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
