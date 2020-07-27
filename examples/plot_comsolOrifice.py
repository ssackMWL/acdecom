
"""
How to post-process simulation data
==================================================

In this example we compute the scattering through an orifice plate in a circular duct with flow. The data is extracted
from two Comsol Multiphysics simulations with a similar setup as in
`this study <https://www.sciencedirect.com/science/article/abs/pii/S0022460X17306752?via%3Dihub>`_\.
The geometry is the same as in `this study <https://www.sciencedirect.com/science/article/abs/pii/S0022460X09002995>`_\.
"""

# %%
# .. image:: ../../image/orifice.png
#    :width: 800
#
# 1. Initialization
# -----------------
# First, we import the packages needed for this example.
import numpy
import matplotlib.pyplot as plt
import acdecom
# %%
# The orifice is mounted inside a circular test duct with a radius of 15 mm.
# The highest frequency of interest is 4000 Hz. The bulk Mach-number is 0.042 and the temperature is 295 Kelvin.
section = "circular"
r = 0.015  # m
f_max = 4000  # Hz
M = 0.042
t = 295  # Kelvin
# %%
# The graphic in the beginning of this examples shows a snapshot of the pressure field from the simulation.
# The "bubbles" downstream from the orifice plate are the vortex shed by the acoustic field. When we setup the
# decomposition domain, we want to be outside the shedding region, to avoid detecting *artificial modes* that do not
# belong to the acoustic field. Therefore, we will only use pressure values that are at least 10 duct diameters away
# from the orifice.

distance = 20.*r  # m

# %%
# To analyze the measurement data, we create objects for the US and the DS test ducts.
td_US = acdecom.WaveGuide(dimensions=(r, ), cross_section=section, f_max=f_max, damping="no",
                          M=M, temperature=t, flip_flow=True)
td_DS = acdecom.WaveGuide(dimensions=(r, ), cross_section=section, f_max=f_max, damping="no",
                          M=M, temperature=t)
# %%
#
# .. note::
#   The standard flow direction is in :math:`P_+` direction. On the inlet side, the Mach-number therefore must be
#   either set negative or the argument *flipFlow* must be set to *True*.
#.. note::
#   We use *no* dissipation model. In the numerical simulation, we have turned off the boundary layer effects and we
#   do not need to take them into account for our post-processing.

# %%
# 2. Data Preparation
# -------------------
# We will post-process the direct data output using Comsol Multiphysics. The solution is two-dimensional and
# axi-symmetric. Using Comsol, we exported the two files
# `comsolOrificeCase1.txt <https://github.com/ssackMWL/acdecom/blob/master/examples/data/comsolOrifice1.txt>`_
# and `comsolOrificeCase2.txt <https://github.com/ssackMWL/acdecom/blob/master/examples/data/comsolOrifice1.txt>`_\.
# They contain the same geometry. However, the orifice was either excited with an acoustic plane wave from the upstream
# direction or from the downstream direction. We will post-process both files in order to extract the sound scattering
# in both directions.
#
# First, we print the header of the files to get a cleaner view of their structure.

with open("data/comsolOrifice1.txt","r") as pressurefile:
    for i in range(11):
        print(pressurefile.readline())

# %%
# There are a few header lines starting with *%*, followed by the pressure data for different frequencies.
# The first and second columns are the r and z coordinates of the grid points. The remaining columns hold the pressure
# at those positions for different frequencies. The frequencies range from 500 Hz to 4000 Hz in steps of 50 Hz.
# We can create variables that will be useful later in the study.

z_col = 1
r_col = 0
f = numpy.arange(500,4001,50)
header = "%"

# %%
# .. note::
#   There is no column for the circumferential coordinate. The reason is, that we did a two dimensional, axi-symmetric
#   simulation that does not require a circumferential coordinate.
#
# We read both simulation files.

pressure_1 = numpy.loadtxt("data/comsolOrifice1.txt", dtype=complex, comments=header)
pressure_2 = numpy.loadtxt("data/comsolOrifice2.txt", dtype=complex, comments=header)

# %%
# We delete all positions that are too close to the orifice plate. Furthermore, we split the simulation in
# upstream (*negative z*) and downstream (*positive z*) sides.

pressure_1us = pressure_1[pressure_1[:, z_col] < -distance, :]
pressure_1ds = pressure_1[pressure_1[:, z_col] > distance, :]

pressure_2us = pressure_2[pressure_2[:, z_col] < -distance, :]
pressure_2ds = pressure_2[pressure_2[:, z_col] > distance, :]

# %%
# We can check how many grid points we have on the upstream and the downstream side.

print("Probes on US side ", pressure_1us.shape[0], ". Probes on DS side: ", pressure_1ds.shape[0])

# %%
# Generally, we can use all grid points for post processing. However, this is not the best method
# if we have a large grid with many points. Instead, we will use a random choice of points.

number_of_probes = 100
mics_rows_US = numpy.random.choice(pressure_1us.shape[0], size=number_of_probes, replace=False)
mics_rows_DS = numpy.random.choice(pressure_1ds.shape[0], size=number_of_probes, replace=False)

# %%
# We extract the coordinates of the random grid points.

z_DS = numpy.abs(pressure_1ds[mics_rows_DS, z_col])  # m
r_DS = numpy.abs(pressure_1ds[mics_rows_DS, r_col])   # m
phi_DS = numpy.zeros((number_of_probes, ))  # deg

z_US = numpy.abs(pressure_1us[mics_rows_US, z_col])  # m
r_US = numpy.abs(pressure_1us[mics_rows_US, r_col])   # m
phi_US = numpy.zeros((number_of_probes, ))  # deg

# %%
# We assign the random grid points as microphones to the object.

td_US.set_microphone_positions(z_US, r_US, phi_US)
td_DS.set_microphone_positions(z_DS, r_DS, phi_DS)

# %%
# In order to decompose the simulation data, we must express it in a format that can be processed by
# :meth:`.WaveGuide.decompose`.
#
# First, we remove all grid points except the random points that we have chosen for the decomposition. Furthermore,
# we remove the two columns that contain the coordinates of the grid points.

pressure_1us = numpy.delete(pressure_1us[mics_rows_US, :], [z_col,r_col], axis=1)
pressure_1ds = numpy.delete(pressure_1ds[mics_rows_DS, :], [z_col,r_col], axis=1)

pressure_2us = numpy.delete(pressure_2us[mics_rows_US, :], [z_col,r_col], axis=1)
pressure_2ds = numpy.delete(pressure_2ds[mics_rows_DS, :], [z_col,r_col], axis=1)

# %%
# Next, we add a new row that contains the frequencies (similar to the line in the header of our file), and one
# more that contains the test case number. This row is has the value *0* for case 1, and *1* for case 2.
# Subsequently, we transpose the array, to have the data format required by :meth:`.WaveGuide.decompose`.

pressure_1us = numpy.vstack([pressure_1us, f, numpy.zeros((f.shape[0],))]).T
pressure_2us = numpy.vstack([pressure_2us, f, numpy.ones((f.shape[0],))]).T

pressure_1ds = numpy.vstack([pressure_1ds, f, numpy.zeros((f.shape[0],))]).T
pressure_2ds = numpy.vstack([pressure_2ds, f, numpy.ones((f.shape[0],))]).T

# %%
# Finally, we combine the two cases at the upstream and the downstream sides to create two large data sets.

pressure_US = numpy.vstack([pressure_1us, pressure_2us])
pressure_DS = numpy.vstack([pressure_1ds, pressure_2ds])

# %%
# The pressure at the different grid points is stored in the first *number_of_probes* columns.
# The frequency is stored in the second last column; and the case number is stored in the last column. We create
# variables that we will use later in this study.

mic_col = range(0, number_of_probes)
frequ_col = -2
case_col = -1

# %%
# 3. Decomposition
# ----------------
#
# With the pre-processed data, we can run the decomposition.

decomp_us, headers_us = td_US.decompose(pressure_US, frequ_col, mic_col, case_col=case_col)

decomp_DS, headers_DS = td_DS.decompose(pressure_DS, frequ_col, mic_col, case_col=case_col)

# %%
# 4. Further Post-processing
# --------------------------
# We can print the *headersDS* to see the names of the columns of the arrays that store the decomposed sound fields.

print(headers_us)

# %%
# We use that information to extract the modal data.

minusmodes = [1]  # from headers_us
plusmodes = [0]

# %%
# Furthermore, we acquire the unique decomposed frequency points.

frequs = numpy.abs(numpy.unique(decomp_us[:, headers_us.index("f")]))
nof = frequs.shape[0]

# %%
# For each of the frequencies we compute the scattering matrix by solving a linear system of equations
# :math:`S = p_+ p_-^{-1}`, wherein :math:`S` is the scattering matrix and :math:`p_{\pm}` are matrices containing the
# acoustic modes that are placed in rows and the different test cases that are placed in columns.
#
# .. note::
#   Details for the computation of the Scattering Matrix and the procedure to measure the different test-cases can be
#   found in `this study <https://www.ingentaconnect.com/content/dav/aaua/2016/00000102/00000005/art00008>`_\.

S = numpy.zeros((2,2,nof),dtype = complex)

for fIndx, f in enumerate(frequs):
    frequ_rows = numpy.where(decomp_us[:, headers_us.index("f")] == f)
    ppm_us = decomp_us[frequ_rows]
    ppm_DS = decomp_DS[frequ_rows]
    pp = numpy.concatenate((ppm_us[:,plusmodes].T, ppm_DS[:,plusmodes].T))
    pm = numpy.concatenate((ppm_us[:,minusmodes].T, ppm_DS[:,minusmodes].T))
    S[:,:,fIndx] = numpy.dot(pp, numpy.linalg.pinv(pm))

# %%
# 5. Plot
# -------
# We can plot the transmission coefficients. Transmission coefficients higher than 1 indicate frequencies where
# amplification can occur.

amplification_us = numpy.abs(S[1, 0, :]) > 1
amplification_ds = numpy.abs(S[0, 1, :]) > 1

plt.hlines(1, frequs[0], frequs[-1], linestyles="dashed", color="grey")

plt.plot(frequs, numpy.abs(S[1, 0, :]),
         ls="-", color="#D38D7B", alpha=0.5)
plt.plot(frequs, numpy.abs(S[0, 1, :]),
         ls="-", color="#67A3C1", alpha=0.5)

plt.plot(frequs[amplification_us], numpy.abs(S[1, 0, amplification_us]),
         ls="-", color="#D38D7B", label="Transmission Upstream")
plt.plot(frequs[amplification_ds], numpy.abs(S[0, 1, amplification_ds]),
         ls="-", color="#67A3C1", label="Transmission Downstream")

plt.xlabel("Frequency [Hz]")
plt.ylabel("Scattering Magnitude")
plt.title("Sound transmission including absorption and amplification at \n an orifice plate with flow.")

plt.legend()
plt.show()

