
import numpy
from scipy.special import jn
import scipy.linalg
import scipy.optimize
import warnings


class WaveGuide:
    r""" This class can be used to decompose in-duct sounds into acoustic modes. It can be used for experimental data
    and for numeric data.
    """

    def __init__(self, dimensions, **kwargs):
        r"""
            Parameters for initialization:

            Parameters
            ----------
            dimensions : array_like
                The dimensions of the waveguide.
                    - For a circular duct, *dimensions* is (radius, ).
                    - For  a rectangular duct, *dimension* is (dimension x, dimension y).
                    - For any other shape, *dimensions* can be specified in the same way (dimension 1, dimension 2, ...).
            temperature : float, optional
                Temperature in Kelvin of the medium inside the waveguide. Defaults to T 293.15.
            M : float, optional
                Bulk Mach_number in :math:`z_+` direction. Defaults to 0.
            flip_flow : boolean, optional
                If *True*, it changes the flow-direction from :math:`z_+` to :math:`z_-` direction. A negative Mach
                number has the same effect. Defaults to *False*.
            damping : {"no", "kirchoff","dokumaci"}, optional
                Choose one of the pre-defined acoustic dispersion models. Defaults to "no".
                    - "no": no predefined dissipation is used. This should be used to implement custom dissipation models.
                    - "kirchoff":  `Kirchoff's thermo-viscous dissipation <https://onlinelibrary.wiley.com/doi/abs/10.1002/andp.18682100602>`_ is used.
                    - "dokumaci": `Dokumaci's thermo-viscous dissipation <https://www.sciencedirect.com/science/article/pii/S0022460X14004921>`_ is used. Useful for higher order modes and waveguides with flow.
                    - "stinson": `Stinson's thermo-viscous dissipation <https://asa.scitation.org/doi/10.1121/1.400379>`_ is used.
            distance : float
                The distance between the decomposition cross section and the first microphone. Defaults to 0.
            cross_section : {"circular","rectangular","custom"}, optional
                Choose one of the pre-defined duct profiles. Defaults to "circular". If set to "custom" the methods
                :meth:`.WaveGuide.get_c`, :meth:`WaveGuide.get_psi` , and :meth:`.WaveGuide.get_wavenumber` should be
                customized.
            f_max : float, optional
                If set, all propagating modes up to the frequency `f_max` [Hz] are pre-computed and decompositions run
                faster. Defaults to 1000.
            gas_constant: float, optional
                The ideal gas constant of the medium inside the waveguide. Defaults to 287.053072.
            dynamic_viscosity : float, optional
                The dynamic viscosity of the medium inside the waveguide. Defaults to 10.13e-6.
            pressure : float, optional
                The static pressure of the medium inside the  waveguide. Defaults to 101.000.
            heat_capacity : float, optional
                The heat capacity ratio of the medium inside the waveguide. Defaults to 1.401.
            thermal_conductivity : float, optional
                The thermal conductivity of the medium inside the waveguide. Defaults to 0.02587.
            eigenvalue : functional, optional
                Uses an external function to compute the eigenvalues. If not specified, the cross section specific
                function is used. If a custom cross section is specified, the eigenvalue defaults to 1.
            modeshape : functional, optional
                Uses an external function to compute the mode shapes. If not specified, the cross section specific
                function is used. If a custom cross section is specified, the mode shapee defaults to 1.
            wavenumber : functional, optional
                Uses an external function to compute the wavenumber. If not specified, the cross section specific
                function is used. If a custom cross section is specified, the wavenumber defaults to the wavenumber
                for a circular duct.
            normalization : functional, optional
                Uses an external function to compute the normalization factor for the mode shapes. If not specified,
                the cross section specific function is used. If a custom cross section is specified, the mode shapee
                normalization defaults to 1.
            """

        # Define parameters of the test domain
        self.temperature = kwargs.get("temperature", 293.15)
        self.M = kwargs.get("M", 0)
        self.flip_flow = 1
        if "flip_flow" in kwargs.keys() and kwargs["flip_flow"] == True:
            self.flip_flow = -1

        # Acoustic dissipation
        self.damping = kwargs.get("damping", "no")

        # Duct Geometry
        self.dimensions = dimensions
        self.cross_section = kwargs.get("cross_section", "circular")
        self.f_max = kwargs.get("f_max", 100)

        # Define the parameter of the air
        self.gas_constant = kwargs.get("gas_constant", 287.053072)
        self.mu = kwargs.get("dynamic_viscosity", 18.13e-6)
        self.p0 = kwargs.get("pressure", 101000)
        self.gamma = kwargs.get("heat_capacity", 1.401)
        self.kth = kwargs.get("thermal_conductivity", 0.02587)
        # Compute specific heat constant
        self.Cp = self.gamma / (self.gamma - 1) * self.gas_constant
        # Compute density
        self.rho = self.p0 / self.gas_constant / self.temperature
        # Calculate the speed of sound
        self.speed_of_sound = numpy.sqrt(self.gamma * self.gas_constant * self.temperature)

        self.microphone_group = [[], []]
        # Axial position of the microphones
        self.microphone_position = []
        # Rectangular duct: Pos1 = x, pos2 = y of the microphone
        # Circular duct: Pos1 = angle, pos2 = radial position of the microphone
        self.microphone_pos1 = []
        self.microphone_pos2 = []
        # Allocate the cut on mode to 1, will be recalculated later
        self.cuton_mode = 1;
        self.ref_angle = 0;
        # Allocate the distance between the microphones and the decomposition cross section. Important for loss model.
        self.distance = kwargs.get("distance",0)
        self.frequency = 0
        self._link_functions(**kwargs)
        self.get_kappa = numpy.vectorize(self.get_kappa)
        # Set the eigenvalues for the first propagating mode (plane wave)
        self.kappa = self._init_eigenvalue_matrix(0, 0)
        # Set the moce vecotrs and the eigenvalues for all modes that can propagate in the expected frequency range
        self.mode_vector, self.kappa = self._init_modes()

    def _link_functions(self, **kwargs):
        """
        Links the cross section specific and problem specific functions to the class.

        Parameters
        ----------
        eigenvalue : functional, optional
            Uses an external function to compute the eigenvalues. If not specified, the cross section specific
            function is used. If a custom cross section is specified, the eigenvalue defaults to 1.
        modeshape : functional, optional
            Uses an external function to compute the mode shapes. If not specified, the cross section specific
            function is used. If a custom cross section is specified, the mode shapee defaults to 1.
        wavenumber : functional, optional
            Uses an external function to compute the wavenumber. If not specified, the cross section specific
            function is used. If a custom cross section is specified, the wavenumber defaults to the wavenumber
            for a circular duct.
        normalization : functional, optional
            Uses an external function to compute the normalization factor for the mode shapes. If not specified,
            the cross section specific function is used. If a custom cross section is specified, the mode shapee
            normalization defaults to 1.
        """

        damping = None
        if self.damping == "kirchoff":
            damping = self.K0_kirchoff
        if self.damping == "dokumaci":
            damping = self.K0_dokumaci

        if self.cross_section == "circular":
            eigenvalue = kwargs.get("eigenvalue", self.get_eigenvalue_circular)
            modeshape = kwargs.get("modeshape", self.get_psi_circular)
            wavenumber = kwargs.get("wavenumber", self.get_wavenumber)
            mode_norm = kwargs.get("normalization", self.get_c_circular)
            if self.damping == "stinson":
                damping = self.K0_stinson_circular

        elif self.cross_section == "rectangular":
            eigenvalue = kwargs.get("eigenvalue", self.get_eigenvalue_rect)
            modeshape = kwargs.get("modeshape", self.get_psi_rect)
            wavenumber = kwargs.get("wavenumber", self.get_wavenumber)
            mode_norm = kwargs.get("normalization", self.get_c_rect)
            if self.damping == "stinson":
                damping = self.K0_stinson_rect

        else:
            eigenvalue = kwargs.get("eigenvalue", self.get_eigenvalue)
            modeshape = kwargs.get("modeshape", self.get_psi)
            wavenumber = kwargs.get("wavenumber", self.get_wavenumber)
            mode_norm = kwargs.get("normalization", self.get_c)

        if damping is not None:
            self.get_K0 = damping

        self.get_wavenumber = wavenumber
        self.get_eigenvalue = eigenvalue
        self.get_psi = modeshape
        self.get_c = mode_norm

    def _init_eigenvalue_matrix(self, m, n):
        """
        Initializes a matrix that contains the eigenvalues for all propagating modes.

        Parameters
        ----------
        m : integer
            Mode-order in the first direction. If the waveguide is circular, m indicates the circumferential mode-order.
            The plane wave has the order 0.
        n : integer
            Mode-order in the second direction. If the waveguide is circular, n indicates the radial mode-order.
            The plane wave has the order 0.

        Returns
        -------
        numpy.ndArray of the dimension m x n that contains the eigenvalues for all modes up to the mode-order (m,n).
        """

        ematrix = numpy.zeros((m + 2, n + 2))

        for mOrder in range(m + 2):
            for nOrder in range(n + 2):
                ematrix[mOrder, nOrder] = self.get_eigenvalue(mOrder, nOrder)

        return ematrix

    def _init_modes(self):
        """
        Finds the order of the (m,n)-modes regarding their cut-on frequencies.

        Returns
        -------
        (mode_vector, kappa): tuple
            - mode_vector : numpy.ndArray, containing tuples (m,n) of the modes, in the order of their cut-on frequency.
            - kappa :  numpy.ndArray of the dimension m x n that contains the eigenvalues for all modes up to the mode
                order (m,n).
        """

        mode_vector = []
        # At least the plane wave ( (0,0)- mode ) must be cut-on
        maxm = 0
        maxn = 0
        # Find the first m-mode-order that is cut-off
        while (numpy.imag(self.get_wavenumber(maxm, 0, self.f_max, sign=-1, dissipative=False)) == 0 and
               numpy.imag(self.get_wavenumber(maxm, 0, self.f_max, sign=+1, dissipative=False)) == 0):
            maxm += 1
        # Find the first n mode-order that is cut-off
        while (numpy.imag(self.get_wavenumber(0, maxn, self.f_max, sign=-1, dissipative=False)) == 0 and
               numpy.imag(self.get_wavenumber(0, maxn, self.f_max, sign=+1, dissipative=False)) == 0):
            maxn += 1
        # Create a matrix that contains all cut-on (and some cut-off) eigenvalues for the modes
        ematrix = self._init_eigenvalue_matrix(maxm, maxn)
        kappa = numpy.copy(ematrix)
        # Iterate through all modes as long as the cut-off mode with the smallest eigenvalue is found
        currentm, currentn = numpy.unravel_index(numpy.argmin(ematrix, axis=None), ematrix.shape)
        while (numpy.imag(self.get_wavenumber(currentm, currentn, self.f_max, sign=-1, dissipative=False)) == 0 and
               numpy.imag(self.get_wavenumber(currentm, currentn, self.f_max, sign=+1, dissipative=False)) == 0):
            # If the duct is circular, the mode-order can be positive and negative
            if self.cross_section == "circular" and not currentm == 0:
                mode_vector.append([-1 * currentm, currentn])
            mode_vector.append([currentm, currentn])
            ematrix [currentm, currentn] = numpy.Inf
            currentm, currentn = numpy.unravel_index(numpy.argmin(ematrix, axis=None), ematrix.shape)

        return (numpy.array(mode_vector), kappa)

    def get_domainvalues(self):
        """
        Returns the characteristic properties of the waveguide and the medium.

        Returns
        -------
        dict
            The characteristic properties {"density", "dynamic_viscosity", "specific_heat", "heat_capacity",
            "thermal_conductivity", "speed_of_sound", "Mach_number", "radius", "bulk-viscosity"} of the waveguide and
            the medium.
        """

        return {"density": self.rho, "dynamic_viscosity": self.mu, "specific_heat": self.Cp,
                "heat_capacity": self.gamma, "thermal_conductivity": self.kth, "speed_of_sound": self.speed_of_sound,
                "Mach_number": self.M * self.flip_flow, "radius": self.get_radius(), "bulk-viscosity": 0.6 * self.mu}

    def set_distance(self, d):
        """
        Sets the distance between the first microphone and the decomposition cross section.

        Parameters
        ----------
        d : float
            Distance in [m] between the first microphone and the decomposition cross section.
        """
        # Subtract the old distance and add the new distance
        self.microphone_position = self.microphone_position + d - self.distance
        self.distance = d

    def get_radius(self):
        """
        Returns the radius or an equivalent measure for the waveguide.

        Returns
        -------
        float
            If the waveguide's cross section is "circular", the radius is returned. If the waveguide is "rectangular",
            the hydraulic radius is returned. Otherwise, dimension[0] is returned.
        """

        if self.cross_section == "circular":
            radius = self.dimensions[0]
        # If the duct is rectangular, return the hydraulic radius.
        elif self.cross_section == "rectangular":
            radius = self.dimensions[0] * self.dimensions[1] / (self.dimensions[0] + self.dimensions[1])
        else:
            radius = self.dimension[0]

        return radius

    def set_temperature_pressure(self, t=None, p0=None):
        """
        Sets the temperature and pressure in the waveguide. Recalculates the speed of sound, the density, and the
        Mach_number.

        Parameters
        ----------
        t : float
            Temperature in Kelvin of the medium inside the waveguide.
        p0 : float, optional
            Static pressure in Pa of the medium inside the waveguide.
        """

        # Update temperature and pressure
        if p0 is not None:
            self.p0 = p0
        if t is not None:
            self.temperature = t
        # recompute the properties of the medium
        speed_of_sound_updated = numpy.sqrt(self.gamma * self.gas_constant * self.temperature)
        self.M *= self.speed_of_sound / speed_of_sound_updated
        self.rho = self.p0 / self.gas_constant / self.temperature
        self.speed_of_sound = speed_of_sound_updated

    def set_flip_flow(self, flip_flow):
        """
        Set the flow direction. Standard flow direction is in :math:`P_+` direction.

        Parameters
        ----------
        flip_flow : bool
            If flip_flow is *True*, the flow direction is in :math:`P_-` direction (towards the test component).
        """

        if flip_flow:
            self.flip_flow = -1
        else:
            self.flip_flow = 1

    def read_microphonefile(self, filename, cylindrical_coordinates=False, **kwargs):
        """
        Reads a file that contains the microphone position. The dimensions are [m] or [deg].
        The file must have the following structure:
            - For Circular duct:

                ===  ===  ==
                z1   r1   :math:`\Phi` 1
                z2   r2   :math:`\Phi` 2
                ...  ...  ...
                zm   rm   :math:`\Phi` m
                ===  ===  ==

            - For other ducts:
                ===  ===  ==
                z1   x1       :math:`y` m
                z2   x2       :math:`y` m
                ...  ...      ...
                zm   xm       :math:`y` m
                ===  ===  ==

        Parameters
        ----------
        filename : str
            Full Path to the file that contains the microphone data.
        cylindrical_coordinates : bool, optional
            If *True* the circumferential position is converted from deg. to radians.
        kwargs : additional parameters
            Will be passed to `numpy.loadtxt <https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html>`_\.
            Please refer to the numpy documentation for information.
        """

        self.microphone_position = numpy.loadtxt(filename, **kwargs)
        # Transform deg to radians
        if cylindrical_coordinates:
            self.microphone_position[:, 2] *= numpy.pi / 180
        self.microphone_position[:, 0] += self.distance

    def set_microphone_positions(self, posz, pos1, pos2, cylindrical_coordinates = False):
        """
        Sets the positions for the pressure probes. The dimensions are [m] or [deg].

        Parameters
        ----------
        posz : array_like
            Axial positions of the pressure probes.
        pos1 : array_like
            Position of the pressure probe in the first dimension. For waveguides with circular cross section, this
            is the radial position in meters.
        pos2 : array_like
            Position of the pressure probe in the second dimension. For waveguides with circular cross sections, this
            is the circumferential position in deg.
        cylindrical_coordinates : bool, optional
            If *True* the circumferential position is converted from deg. to radians.
        """

        self.microphone_position = numpy.zeros((len(posz), 3))
        self.microphone_position[:, 0] = posz
        self.microphone_position[:, 1] = pos1
        self.microphone_position[:, 2] = pos2
        if cylindrical_coordinates:
            self.microphone_position[:, 2] *= numpy.pi / 180
        self.microphone_position[:, 0] += self.distance

    def get_eigenvalue(self, m, n, **kwargs):
        """
        Placeholder for the eigenvalue of the (m,n)-mode. When the object is initialized, this function may be
        overwritten either by one of the predefined eigenvalue functions :meth:`.WaveGuide.get_eigenvalue_circular`
        (circular waveguide) and :meth:`.WaveGuide.get_eigenvalue_rect` (rectangular waveguide),
        or a custom function.

        Parameters
        ----------
        m : int
            Mode-order in the first direction. If the waveguide is circular, m indicates the circumferential mode-order.
            The plane wave has the order 0.
        n : integer
            Mode-order in the second direction. If the waveguide is circular, n indicates the radial mode-order.
            The plane wave has the order 0.
        kwargs : additional arguments

        Returns
        -------
        complex
            The eigenvalue of the (m,n)-mode.
        """

        return 1

    def get_eigenvalue_circular(self, m, n, **kwargs):
        r"""
        Returns the eigenvalue of the (m,n)-mode for a circular duct.

        .. math::
            \kappa_{mn} = \frac{R_n(J'_m)}{r}

        where :math:`J'_m` is the derivative of the Bessel function of first kind and order m, and :math:`R_n` is the
        n-th zero.

        Parameters
        ----------
        m : int
            Mode-order in the circumferential direction. The plane wave has the order 0.
        n : integer
            Mode-order in the radial direction. The plane wave has the order 0.
        kwargs : additional arguments

        Returns
        -------
        float or complex
            The eigenvalue of the (m,n)-mode, divided by the radius.
        """
        # here, we need to correct the fact that the scipy value for the 0-0 value is missing.
        bessel_der_zero = scipy.special.jnp_zeros(m, n + 1)
        if m == 0:
            bessel_der_zero = numpy.append([0], bessel_der_zero)

        return bessel_der_zero[n] / self.dimensions[0]

    def get_eigenvalue_rect(self, m, n, **kwargs):
        r"""
        Returns the eigenvalue of the (m,n)-mode for a rectangular duct.

        .. math ::
            \kappa_{mn} = \pi (\frac{n}{a} + \frac{m}{b})

        Parameters
        ----------
        m : int
            Mode-order in x-direction. The plane wave has the order 0.
        n : integer
            Mode-order in y-direction. The plane wave has the order 0.
        kwargs : additional arguments

        Returns
        -------
        float or complex
            The eigenvalue of the (m,n)-mode.
        """

        return numpy.pi * (n / self.dimensions[0] + m / self.dimensions[1])

    def get_kappa(self, m, n):
        """
        Returns the eigenvalues of the (m,n)-mode. This method picks the eigenvalues from a pre-calculated eigenvalue
        matrix, which makes computations in larger data sets more efficient. For the computation of the eigenvalue,
        see :meth:`.WaveGuide.get_eigenvalue` .

        Parameters
        ----------
        m : int
            Mode-order in the circumferential direction. The plane wave has the order 0.
        n : integer
            Mode-order in the radial direction. The plane wave has the order 0.

        Returns
        -------
        float or complex
            The eigenvalue of the (m,n)-mode.
        """

        # if kappa was not computed for m and n yet, compute it
        m, n = abs(m), abs(n)
        if m >= self.kappa.shape[0] or n >= self.kappa.shape[1]:
            self.kappa = self._init_eigenvalue_matrix(m, n)

        return self.kappa[m, n]

    def get_c(self, m, n, **kwargs):
        """
        Placeholder for the normalization of the mode shapee of the (m,n)-mode.
        When the object is initialized, this function may be overwritten either by one of the predefined
        normalization functions :meth:`.WaveGuide.get_c_circular` (circular waveguide) and
        :meth:`.WaveGuide.get_c_rect` (rectangular waveguide), or a custom function. The predefined functions
        normalize the mode shapes to be orthonormal, i.e.,

        .. math ::
            \int_{A} \psi_{mn} \psi_{m'n'} dA = \delta_{mm'} \delta_{nn'}

        and :math:`\delta` is the Kronecker delta.

        Parameters
        ----------
        m : int
            Mode-order in the first direction. If the waveguide is circular, m indicates the circumferential mode-order.
            The plane wave has the order 0.
        n : integer
            Mode-order in the second direction. If the waveguide is circular, n indicates the radial mode-order.
            The plane wave has the order 0.
        kwargs : additional arguments.

        Returns
        -------
        float or complex
            The normalization factor of the (m,n)-mode.
        """

        return 1

    def get_c_rect(self, m, n, **kwargs):
        r"""
        Return the normalization of the (m,n)-mode for a rectangular duct. Modes are normalized to be orthonormal.

        .. math ::
            C_{mn} = \frac{\sqrt{a b}}{2}

        Parameters
        ----------
        m : int
            Mode-order in x-direction. The plane wave has the order 0.
        n : integer
            Mode-order in y-direction. The plane wave has the order 0.
        kwargs : additional arguments

        Returns
        -------
        float or complex
            The eigenvalue of the (m,n)-mode.
        """

        return numpy.sqrt(self.dimensions[0] * self.dimensions[1]) / 2

    def get_c_circular(self, m, n, **kwargs):
        """
        Return the normalization of the (m,n)-mode for a circular duct. Modes are normalized to be orthonormal.

        .. math ::
            C_{mn} = \sqrt{A * (J_m(r\kappa)^2 - J_{m-1}(r\kappa) * J_{m+1}(r\kappa))}

        where :math:`J_m` is the Bessel function of 1 Kind and Order *m*, and :math:`\kappa` is the eigenvalue, see
        :meth:`.WaveGuide.get_eigenvalue` .

        Parameters
        ----------
        m : int
            Mode-order in the circumferential direction. The plane wave has the order 0.
        n : integer
            Mode-order in the radial direction. The plane wave has the order 0.
        kwargs : additional arguments

        Returns
        -------
        complex
            The normalization-factor of the (m,n)-mode.
        """

        k_r = self.dimensions[0] * self.get_kappa(m, n)
        A = self.dimensions[0] ** 2 * numpy.pi

        return numpy.sqrt(A * (numpy.square(jn(m, k_r)) - jn(m - 1, k_r) * jn(m + 1, k_r)))

    def get_psi(self, m, n, pos1, pos2, **kwargs):
        """
        Placeholder for the normalized mode shapes of the (m,n)-mode.  When the object is initialized, this function
        may be overwritten either by one of the predefined normalization functions :meth:`.WaveGuide.get_psi_circular`
        (circular waveguide) and :meth:`.WaveGuide.get_psi_rect` (rectangular waveguide), or a custom function.

        Parameters
        ----------
        m : int or array_like
            Mode-order in the first direction. If the waveguide is circular, m indicates the circumferential mode-order.
            The plane wave has the order 0.
        n : int or array_like
            Mode-order in the second direction. If the waveguide is circular, n indicates the radial mode-order.
            The plane wave has the order 0.

        pos1 : float or array_like
            Position in the first direction.
        pos2 : float or array_like
            Position in the second direction.
        kwargs : additional arguments

        Returns
        -------
        complex
            The eigenvalue of the (m,n)-mode.
        """

        return numpy.ones((len(pos1),), dtype=numpy.complex)

    def get_psi_circular(self, m, n, r, phi, **kwargs):
        r"""
        Return the normalized mode shapee of the (m,n)-mode for a circular duct. Modes are normalized to be
        orthonormal, see :meth:`.WaveGuide.get_c`\.

        .. math::
            \Psi_{mn} = \frac{J_m(R\kappa_{mn}) \mathbf{e}^{\mathbf{i} m \phi}}{C_{mn}}

        where :math:`\kappa` is the eigenvalue, see :meth:`.WaveGuide.get_eigenvalue` .

        Parameters
        ----------
        m : int
            Mode-order in the circumferential direction. The plane wave has the order 0.
        n : integer
            Mode-order in the radial direction. The plane wave has the order 0.
        r : float or array_like
            Radial-coordinate.
        phi : float or array_like
            Circumferential-coordinate.
        kwargs : additional arguments

        Returns
        -------
        complex
            The normalized mode shapee of the (m,n)-mode at position (pos1,pos2).
        """

        return 1 / self.get_c(m, n) * jn(m, r * self.get_kappa(m, n)) * numpy.exp(m * phi * 1j)

    def get_psi_rect(self, m, n, x, y):
        r"""
        Return the normalized mode shapee of the (m,n)-mode for a rectangular duct. Modes are normalized to be
        orthonormal, see :meth:`.WaveGuide.get_c`\.

        .. math::
            \Psi_{mn} = \frac{\cos(\pi m x/a)*\cos(\pi n y/b)}{C_{mn}}

        Parameters
        ----------
        m : int
            Mode-order in x-direction. The plane wave has the order 0.
        n : integer
            Mode-order in y-direction. The plane wave has the order 0.
        x : float or array_like
            X - coordinate.
        y: float or array_like
            Y - coordinate.
        kwargs : additional arguments

        Returns
        -------
        float or complex
            The eigenvalue of the (m,n)-mode at the position (x,y).
        """

        return (1 / self.get_c(m, n)
                * numpy.cos(numpy.pi * m * x / self.dimensions[0])
                * numpy.cos(numpy.pi * n * y / self.dimensions[1]))

    def get_wavenumber(self, m, n, f, **kwargs):
        r"""
        Compute the wavenumber of the (m,n)-mode at the frequency f [Hz].

        .. math ::
            k_{mn\pm} = \pm K_0 \frac{\omega}{c} \frac{\sqrt{1-(\kappa_{mn}c/\omega)^2(1-M)}\mp M}{1-M^2}

        Here, :math:`K_0` is the dissipation factor (see :meth:`.WaveGuide.get_K0`\),
        :math:`\omega` is the angular frequency, :math:`c` is the speed of sound,
        :math:`\kappa_{mn}` is the eigenvalue (see :meth:`.WaveGuide.get_eigenvalue`\), and
        :math:`M` is the Mach_number.

        .. Note ::
            The flow is assumed to move in :math:`z_+` direction. For flow towards
            math:`z_-`, you can either use a negative Mach_number or call :meth:`.WaveGuide.set_flip_flow`
            with flipFlow = *True*.

        Parameters
        ----------
        m : int or array_like
            Mode-order in the first direction. If the waveguide is circular, m indicates the circumferential mode-order.
            The plane wave has the order 0.
        n : int or array_like
            Mode-order in the second direction. If the waveguide is circular, n indicates the radial mode-order.
            The plane wave has the order 0.
        f : float
            Frequency [Hz].
        dissipative : bool, optional
            If *True*, the wavenumber is corrected with the dissipation model. If *False*, :math:`K_0` is set to 1.
            Defaults to *True*.
        sign : 1 or -1
            The direction of the wave propagation. 1 means :math:`z_+` direction, -1 means :math:`z_-` direction.
            Set to 1 for :math:`k_+` and  -1 for :math:`k_-`\. Defaults to 1.

        Returns
        -------
        complex or array_like
            The wavenumber(s) of the (m,n)-mode at the frequency f [Hz].
        """

        sign = kwargs.get("sign", 1)

        omega = 2 * numpy.pi * f
        # This guarantees the correct sign for the wavenumber. See Rienstra, Introduction to Duct Acoustics.
        root = numpy.sqrt(numpy.array(numpy.square(
            self.get_kappa(m, n) * self.speed_of_sound / omega) * (1 - numpy.square(self.M)) - 1, dtype=numpy.complex))
        root *= -1j

        wave_number = (sign * omega / self.speed_of_sound
                       * (root - self.M * sign * self.flip_flow) / (1 - numpy.square(self.M)))

        if kwargs.get("dissipative", True):
            k0 = self.get_K0(m, n, f, **kwargs)
        else:
            k0 = 1

        return k0 * wave_number

    def find_cuton(self, f, **kwargs):
        """
        Returns the number of cut-on modes for the specified frequency `f` Hz.

        Parameters
        ----------
        f : float
            Frequency in Hz.
        kwargs: additional arguments

        Returns
        -------
        int
            Number of cut-on modes. Symmetric modes are counted as two modes, i.e., after the first cut-on in a circular
            duct, three modes can propagate.
        """

        # If the number of modes for the frequency f has not been computed it needs to be recomputed.
        if f > self.f_max:
            self.f_max = f
            self.mode_vector, self.kappa = self._init_modes()
        # Compute the cut-on in upstream adn downstream direction
        cuton1 = (self.mode_vector.shape[0]
                  - numpy.count_nonzero(numpy.imag(self.get_wavenumber(self.mode_vector[:, 0],
                                                                       self.mode_vector[:, 1],
                                                                       f, sign=1, dissipative=False))))
        cuton2 = (self.mode_vector.shape[0]
                  - numpy.count_nonzero(numpy.imag(self.get_wavenumber(self.mode_vector[:, 0],
                                                                       self.mode_vector[:, 1],
                                                                       f, sign=-1, dissipative=False))))
        return max(cuton1, cuton2)

    def get_T(self, m, n, f, z, **kwargs):
        """
        Returns the mode-propagation :math:`T_{i\pm}(z)`\.

        .. math ::
            T{mn\pm} = e^{-\mathbf{i}k_{mn\pm} z}


        Parameters
        ----------
        m : int or array_like
            Mode-order in the first direction. If the waveguide is circular, m indicates the circumferential mode-order.
            The plane wave has the order 0.
        n : int or array_like
            Mode-order in the second direction. If the waveguide is circular, n indicates the radial mode-order.
            The plane wave has the order 0.
        f : float
            Frequency [Hz].
        kwargs: additional arguments

        Returns
        -------

        """
        sign = kwargs.get("sign",1)
        k = self.get_wavenumber(m, n, f, sign=sign)

        return numpy.exp(-1j * k * z)

    def _calculate_modalmatrix(self, f, cuton_mode, **kwargs):
        """
        Returns the modal matrix that can be used for the mode decomposition.

        Parameters
        ----------
        f : float
            Frequency in Hz.
        cuton_mode : int
            Number of cut-on modes.
        kwargs : additional arguments

        Returns
        -------
        numpy.ndArray
            The modal matrix with the dimensions (Microphones x (2 x cuton_mode))
        """

        modalmatrix = numpy.zeros((len(self.microphone_position[:, 0]), 2 * cuton_mode), dtype=numpy.complex)

        for modes in range(cuton_mode):
            m, n = self.mode_vector[modes]

            posz = self.microphone_position[:, 0]
            pos1 = self.microphone_position[:, 1]
            pos2 = self.microphone_position[:, 2]

            modalmatrix[:, modes] = self.get_psi(m, n, pos1, pos2) * self.get_T(m, n, f, posz, sign=1)
            modalmatrix[:, modes + cuton_mode] = self.get_psi(m, n, pos1, pos2) * self.get_T(m, n, f, posz, sign=-1)

        return modalmatrix

    def get_modalmatrix(self, f, **kwargs):
        """
        Returns the modal matrix that can be used for the mode decomposition. Takes flow-effects, temperature, and
        pressure into account.

        .. math::
            p = M p_\pm

        where :math:`p_\pm` is a row vector containing the complex mode amplitudes of all cut-on modes in :math:`z_+` and
        :math:`z_-` direction and :math:`p` is a row vector with measured pressure values in the frequency domain.

        Parameters
        ----------
        f : float
            Frequency in Hz.
        Mach_number : float, optional
            Mach_number of the medium inside the waveguide. Defaults to the value stored in self.M. If specified, it
            overwrites the
            value stored in self.M.
        t : float, optional
            Temperature of the medium inside the waveguide. Defaults to the value stored in self.t. If specified, it
            overwrites the value stored in self.t.
        Ps : float, optional
            Static pressure of the medium inside the waveguide. Defaults to the value stored in self.p0. If specified,
            it overwrites the value stored in self.p0.

        Returns
        -------
        numpy.ndArray
            The modal matrix with the dimensions (Microphones x (2 x cuton_mode))
        """
        # We need to update temperature, Mach_number, and Pressure before we compute the modal-matrix
        if "Mach_number" in kwargs.keys():
            self.M = kwargs["Mach_number"]
        if "t" in kwargs.keys() or "Ps" in kwargs.keys():
            self.set_temperature_pressure(kwargs.get("t", self.temperature), kwargs.get("Ps", self.p0))
        cuton_mode = self.find_cuton(f)
        # From that, we calculate the modal matrix
        modalmatrix = self._calculate_modalmatrix(f, cuton_mode)

        return modalmatrix

    def get_K0(self, m, n, f, **kwargs):
        """
        Placeholder for the dissipation function for the wavenumber. When the object is initiated, this function may
        be overwritten either by one of the predefined dissipation functions :meth:`WaveGuide.getK0_kirchoff`,
        :meth:`WaveGuide.getK0_dokumaci`, meth:`WaveGuide.get_K0_stinso_rect`,
        meth:`WaveGuide.get_K0_stinson_cricular` or a custom function.

        Parameters
        ----------
        m : int or array_like
            Mode-order in the first direction. If the waveguide is circular, m indicates the circumferential mode-order.
            The plane wave has the order 0.
        n : int or array_like
            Mode-order in the second direction. If the waveguide is circular, n indicates the radial mode-order.
            The plane wave has the order 0.
        f : float, optional
            Frequency in Hz.
        kwargs : additional arguments

        Returns
        -------
        complex
            The dissipation factor of the (m,n)-mode at the frequency f [Hz].
        """

        return 1

    def K0_kirchoff(self, m, n, f, **kwargs):
        """
        Dissipation function for the wavenumber based on
        `Kirchoff's thermo-viscous dissipation <https://onlinelibrary.wiley.com/doi/abs/10.1002/andp.18682100602>`_
        in waveguides without flow.

        Parameters
        ----------
        m : int or array_like
            Mode-order in the first direction. If the waveguide is circular, m indicates the circumferential mode-order.
            The plane wave has the order 0.
        n : int or array_like
            Mode-order in the second direction. If the waveguide is circular, n indicates the radial mode-order.
            The plane wave has the order 0.
        f : float, optional
            Frequency in Hz.
        kwargs : additional arguments

        Returns
        -------
        complex
            The dissipation factor of the (m,n)-mode at the frequency f [Hz].
        """
        omega = f * 2 * numpy.pi
        s = self.get_radius() * numpy.sqrt(self.rho * omega / self.mu)
        xi = self.mu * self.Cp / self.kth

        return (1 + complex(1 - 1j) / (numpy.sqrt(2) * s) * (1 + (self.gamma - 1) / xi)
                - 1j / (s * s) * (1 + (self.gamma - 1) / xi - self.gamma / 2 * (self.gamma - 1) / (xi * xi)))

    def K0_dokumaci(self, m, n, f, **kwargs):
        """
        Dissipation function for the wavenumber based on
        `Dokumaci's thermo-viscous dissipation <https://www.sciencedirect.com/science/article/pii/S0022460X14004921>`_
        in waveguides with flow and higher-order modes.

        Parameters
        ----------
        m : int or array_like
            Mode-order in the first direction. If the waveguide is circular, m indicates the circumferential mode-order.
            The plane wave has the order 0.
        n : int or array_like
            Mode-order in the second direction. If the waveguide is circular, n indicates the radial mode-order.
            The plane wave has the order 0.
        f : float, optional
            Frequency in Hz.
        kwargs : additional arguments

        Returns
        -------
        complex
            The dissipation factor of the (m,n)-mode at the frequency f [Hz].
        """
        # We get all needed properties of the testdomain for the model
        values = self.get_domainvalues()
        values["f"] = f
        values["sign"] = kwargs.get("sign", 1)
        # We extract the convection part from the wavenumber, as dokumacis solution for the dissipation includes
        # convection
        convection = (-1 * values["sign"]
                      * self.get_wavenumber(m,n,f,dissipative=False) * self.speed_of_sound/(2*numpy.pi*f))
        # We use Kirchoffs solution as a guess
        dissipation_guess = self.K0_kirchoff(m, n, f, **kwargs) * convection


        dissipation = scipy.optimize.fsolve(self._Dokumaci,
                                            [numpy.real(dissipation_guess), numpy.imag(dissipation_guess)],
                                            (values, m), xtol=1e-12)

        dissipation = (dissipation[0] - 1j * dissipation[1])
        # Again, we have to divide by the convection, as, later in our wavenumber, convection will be multiplied
        return dissipation/convection

    def _Fs(self, nu, omega, **kwargs):
        """
        Function needed by stinsons model for rectangular ducts
        Parameters
        ----------
        nu : float
            Parameter from stinsons model
        omega : float
            Angular frequency
        elements : int
            Number of elements used for the series expansion. Higher values give more accurate results but take longer
            to compute. Defaults to 150.
        kwargs : additional arguments

        Returns
        -------
        complex
            Fs for Stinson's model.

        """
        a = self.dimensions[0]/2
        b = self.dimensions[1]/2


        seriesElements = kwargs.get("elements", 150)
        Y = 4 * 1j * omega / nu / a ** 2 / b ** 2
        series = 0
        for k in range(seriesElements):
            alfak = ((k + 1 / 2) * numpy.pi / a) ** 2
            for n in range(seriesElements):
                betan = ((n + 1 / 2) * numpy.pi / b) ** 2
                series += 1 / (alfak * betan * (alfak + betan + 1j * omega / nu))
        return series * Y

    def _Fc(self, nu, omega, **kwarg):
        """
        Function needed by stinsons model for cirular ducts
        Parameters
        ----------
        nu : float
            Parameter from stinsons model
        omega : float
            Angular frequency
        kwargs : additional arguments

        Returns
        -------
        complex
            Fc for Stinson's model.

        """
        r = self.get_radius()
        Y = -1j * omega / nu
        G = scipy.special.jve(1, r * numpy.power(Y, 0.5)) / scipy.special.jve(0, r * numpy.power(Y, 0.5))

        return 1 - 2 * Y ** (-1 / 2) * G / r

    def K0_stinson_rect(self, m, n, f, **kwargs):
        """
        Dissipation function for the wavenumber based on
        `Stinson's thermo-viscous dissipation <https://asa.scitation.org/doi/10.1121/1.400379>`_ in rectangular
        waveguides without flow.

        Parameters
        ----------
        m : int or array_like
            Mode-order in x-direction. The plane wave has the order 0.
        n : int or array_like
            Mode-order in y-direction. The plane wave has the order 0.
        f : float, optional
            Frequency in Hz.
        elements: int, optional
            Number of elements used for the series expension. Higher values give more accurate results, but take longer
            to compute. Defaults to 150.
        kwargs : additional arguments

        Returns
        -------
        complex
            The dissipation factor of the (m,n)-mode at the frequency f [Hz].
        """

        omega = 2 * numpy.pi * f
        # compute gas properties
        v = self.mu / self.rho
        vp = self.kth / self.rho / self.Cp

        Fs_g = self._Fs(vp / self.gamma, omega)
        Fs_o =  self._Fs(v, omega)
        dissipation = numpy.sqrt(numpy.array(-(self.gamma - (self.gamma - 1) * Fs_g) / Fs_o, dtype=complex)) * -1j

        return dissipation


    def K0_stinson_circular(self, m, n, f, **kwargs):
        """
        Dissipation function for the wavenumber based on
        `Stinson's thermo-viscous dissipation <https://asa.scitation.org/doi/10.1121/1.400379>`_ in circular
        waveguides without flow.

        Parameters
        ----------
        m : int or array_like
            Mode-order in the circumferential direction. The plane wave has the order 0.
        n : int or array_like
            Mode-order in the radial direction. The plane wave has the order 0.
        f : float, optional
            Frequency in Hz.
        kwargs : additional arguments

        Returns
        -------
        complex
            The dissipation factor of the (m,n)-mode at the frequency f [Hz].
        """

        omega = 2 * numpy.pi * f
        # compute gas properties
        v = self.mu / self.rho
        vp = self.kth / self.rho / self.Cp
        Fc_g = self._Fc(vp / self.gamma, omega)
        Fc_o = self._Fc(v, omega)
        dissipation = numpy.sqrt(numpy.array(-(self.gamma - (self.gamma - 1) * Fc_g) / Fc_o, dtype=complex)) * -1j
        return dissipation

    def _J(self, m, v):
        """
        Returns a scaled version of the 1st Bessel function of order m for v.

        Parameters
        ----------
        m : integer
            Order of Bessel function.
        v : float
            Argument for bessel function.

        Returns
        -------
        float
            The 1st Bessel function of order m for v

        """
        # As the equation only has quotients of the bessel functions, we can used the scaled version in order
        # to avoid infinite results.
        return scipy.special.jve(m, v)

    def _JP(self, m, v):
        """
        Returns an exponentially scaled 1st Bessel function of order m for v.

        Parameters
        ----------
        m : integer
            Order of Bessel function.
        v : float
            Argument for bessel function.

        Returns
        -------
        float
            An exponentially scaled 1st Bessel function of order m for v.

        """
        # as the equation only has quotients of the bessel functions, we can used the scaled version in order
        # to avoid infinite results. Since a scaled version of the derivative is not implemented, we compute it from the
        # scaled version of the bessel function.
        return 0.5 * (scipy.special.jve(m - 1, v) - scipy.special.jve(m + 1, v))

    def _Dokumaci(self, kguess, options, m):
        """
        Computes the thermo-viscous dissipation based on Dokumaci's dissipation model for flow ducts and higher-order
        modes.

        Parameters
        ----------
        kguess : tuple
            Guess for the wavenumber (real(k), imag(k)).
        options : dict
            contains the parameters of the decomposition domain:
                {Mach_number, radius, speed_of_sound, density, dynamic_viscosity, specific_heat, thermal_conductivity,
                 bulk-viscosity, heat_capacity}

        Returns
        -------
        tuple
            the dissipation factor, split in its real and imaginary part (real(K0),imag(K0)).
        """

        omega = options["f"] * 2 * numpy.pi
        M = -1 * options["Mach_number"] * options["sign"]
        r = options["radius"]
        c = options["speed_of_sound"]
        rho = options["density"]
        mu = options["dynamic_viscosity"]
        Cp = options["specific_heat"]
        k_therm = options["thermal_conductivity"]
        mu_bulk = options["bulk-viscosity"]
        gamma = options["heat_capacity"]
        K = kguess[0] + 1j * kguess[1]
        k0 = omega / c;
        # Equation 27
        lambda0 = k0 * numpy.sqrt(-numpy.square(K) + 1j * (rho * c * (1 - K * M)) / (mu * k0))
        # equation 12
        H0 = rho * c * Cp * r / k_therm;
        D0 = rho * c * r / (4.0 * mu / 3 + mu_bulk)
        # equation 11
        C0 = 1j * numpy.power(k0 * r * (1 - K * M), 3)
        C2 = 0.5 * 1j * k0 * r * (1 - K * M) * (1 - 1j * k0 * r * (1 / D0 + gamma / H0) * (1 - K * M))
        C4 = 1 / H0 * (1 - 1j * k0 * r * gamma * (1 - K * M) / D0)
        # equation 15
        K12 = 1 / numpy.power(k0 * r, 2) / (C2 / C0 + numpy.sqrt(numpy.power(C2 / C0, 2) - C4 / C0))
        K22 = 1 / numpy.power(k0 * r, 2) / (C2 / C0 - numpy.sqrt(numpy.power(C2 / C0, 2) - C4 / C0))
        # equation 16
        alpha1 = numpy.sqrt(numpy.square(k0) * (K12 - numpy.square(K)))
        alpha2 = numpy.sqrt(numpy.square(k0) * (K22 - numpy.square(K)))

        # Equation 21
        xi1 = rho * Cp * r * ((1 - K * M) / (1j * k0 * r * K12) + gamma / H0)
        xi2 = rho * Cp * r * ((1 - K * M) / (1j * k0 * r * K22) + gamma / H0)



        fc_sub1 = (numpy.square(lambda0 * r) * (lambda0 * r * self._JP(m, lambda0 * r) / self._J(m, lambda0 * r))
                   * (alpha1 * r * self._JP(m, alpha1 * r) / self._J(m, alpha1 * r)
                      - xi2 / xi1 * alpha2 * r *self._JP(m, alpha2 * r) / self._J(m, alpha2 * r)))

        fc_sub2 = ((1 - xi2 / xi1)
                   * (numpy.power(k0 * r * K * (lambda0 * r * self._JP(m, lambda0 * r) / self._J(m, lambda0 * r)), 2)
                      - numpy.power(m, 2) * (numpy.square(lambda0 * r) + numpy.power(k0 * r * K, 2))))

        result = (fc_sub1 + fc_sub2);

        return [result.real, result.imag]

    def decompose(self, data, f_col, probe_col, case_col=None, Mach_col=None, temperature_col=None, Ps_col=None):
        """
        Decompose sound fields into modal components. The function can process several frequencies and test-cases at the
        same time. It returns :math:`p_\pm` , which contains all complex mode amplitudes of the propagating modes in
        :math:`z_+` and :math:`p_-` direction.

        .. math::
             p_\pm = M^{-1} p

        Parameters
        ----------
        data : numpy.ndArray (complex)
            All data that is needed for the decomposition. The columns in the srray are the relevant parameters
            for the decomposition (pressure values, frequencies, temperatures, Mach-numbers).
            The rows are measurements of different frequencies and test-cases. A measurement of a single frequency
            for a test with three microphones would have at following format:

            ==============  ====  ====  ====
            Frequency [Hz]  Mic1  Mic2  Mic3
            ==============  ====  ====  ====
            f1              P1    P2    P3
            ==============  ====  ====  ====

            A measurement of multiple frequencies
            for a test with three microphones would have at following format:

            ==============  ====  ====  ====
            Frequency [Hz]  Mic1  Mic2  Mic3
            ==============  ====  ====  ====
            f1              P11   P21   P31
            f2              P12   P22   P32
            ...             ...   ...   ...
            fn              P1n   p2n   p3n
            ==============  ====  ====  ====

            A measurement of a multiple frequencies with multiple tests per frequencies
            with three microphones would have at following format:

            ==============  ====  ====  ====  ====
            Frequency [Hz]  Case  Mic1  Mic2  Mic3
            ==============  ====  ====  ====  ====
            f1              0     P110  P210  P310
            f1              1     P111  P211  p311
            ...             ...   ...   ...   ...
            fn              m     p1nm  p2nm  p3nm
            ==============  ====  ====  ====  ====

            The method can process any number of testcases, frequencies, and microphones.

        f_col : integer
            Column in *data* in which the frequency is stored. Starting with 0.
        probe_col : array_like
            Columns in *data* in which the pressure at the different probe locations is stored. Starting with 0.
            The order of the probes must be the same as for the microphone positions.
        case_col : integer, optional
            Column in *data* in which the case number is stored. Important for tests with multiple loudspeakers.
            Starting with 0. Defaults to None.
        Mach_col : integer, optional
            Column in *data* in which the Mach-number is stored. Starting with 0. Defaults to None.
        temperature_col : integer, optional
            Column in *data* in which the temperature is stored. Starting with 0. Defaults to None.
        Ps_col : integer, optional
            Column in *data* in which the static pressure is stored. Starting with 0. Defaults to None.

        Returns
        -------
        tuple
            Returns (*decomposed_fields*, *headers*). *decomposed_fields* (numpy.ndArray) contains the complex mode
            amplitudes (as columns) for the different test cases (as rows). *headers* (list) contains the names of the
            columns in *decomposed_fields*.
        """


        # We need to check that pressure for each probe is set.
        if not len(probe_col) == self.microphone_position.shape[0]:
            raise ValueError("The number of columns does not fit the number of specified microphone positions.")

        # compute the maximum number of modes
        self.f_max = numpy.max(numpy.abs(data[:, f_col]))
        mode_vector, _ = self._init_modes()

        # create the headers
        headers = []
        for modes in mode_vector:
            headers.append("(" + str(modes[0]) + "," + str(modes[1]) + ") plus Direction")
        for modes in mode_vector:
            headers.append("(" + str(modes[0]) + "," + str(modes[1]) + ") minus Direction")
        headers += ["f", "Mach_number", "temperature", "Ps", "condition number", "case"]

        # Loop through all rows (different test cases) in the data array and decompose the field
        decomposed_fields = numpy.zeros((data.shape[0], len(headers)), dtype=complex)
        for f in range(data.shape[0]):
            # Copy the data relevant for the decomposition, that the user can use that data for plotting and filtering.
            decomposed_fields[f, 2 * len(mode_vector)] = numpy.abs(data[f, f_col])
            decomposed_fields[f, 2 * len(mode_vector) + 1] = self.M
            decomposed_fields[f, 2 * len(mode_vector) + 2] = self.temperature
            decomposed_fields[f, 2 * len(mode_vector) + 3] = self.p0
            decomposed_fields[f, 2 * len(mode_vector) + 4] = 0

            # The parameters is data that is overwritten by the user, for example the Mach_number. We need to update
            # The parameters before we run the decomposition.
            parameters = {}

            if Mach_col is not None:
                parameters["Mach_number"] = numpy.abs(data[f, Mach_col])
                decomposed_fields[f, 2 * len(mode_vector) + 1] = numpy.abs(data[f, Mach_col])
            if temperature_col is not None:
                parameters["t"] = numpy.abs(data[f, temperature_col])
                decomposed_fields[f, 2 * len(mode_vector) + 2] = numpy.abs(data[f, temperature_col])
            if Ps_col is not None:
                parameters["Ps"] = numpy.abs(data[f, Ps_col])
                decomposed_fields[f, 2 * len(mode_vector) + 3] = numpy.abs(data[f, Ps_col])
            if case_col is not None:
                decomposed_fields[f, 2 * len(mode_vector) + 5] = numpy.abs(data[f, case_col])
            # Here comes the code for the decomposition
            modalmatrix = self.get_modalmatrix(numpy.abs(data[f, f_col]), **parameters)
            inverted_modalmatrix = scipy.linalg.pinv(modalmatrix)
            conditionNumber = numpy.linalg.cond(modalmatrix)
            decomposed_fields[f, 2 * len(mode_vector) + 4] = conditionNumber
            # High condition numbers should give a warning.
            if conditionNumber > 15:
                warnings.warn("The Modal analysis is ill-conditioned for some of the frequencies.")

            decomposedFreq = numpy.dot(inverted_modalmatrix, data[f, probe_col].T)
            # As the decomposed_fields consider all modes that propagate at the highest frequency,
            # we need to assign the modes for frequencies with a lower number of modes manually
            decomposed_fields[f, 0:len(decomposedFreq) // 2] = decomposedFreq[0:decomposedFreq.shape[0] // 2]
            decomposed_fields[f, len(mode_vector):decomposedFreq.shape[0] // 2 + len(mode_vector)] = \
                decomposedFreq[decomposedFreq.shape[0] // 2:decomposedFreq.shape[0]]

        return decomposed_fields, headers
