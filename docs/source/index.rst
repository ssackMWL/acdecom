.. acdecom documentation master file, created by
   sphinx-quickstart on Fri Jun 12 16:03:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

acdecom - acoustic mode decomposition in flow ducts
====================================================

Acdecom is a Python module that makes the post-processing of acoustic data in flow ducts easy. It is developed and
maintained by the `Marcus Wallenberg Laboratory for Sound and Vibration Research <https://www.ave.kth.se/avd/mwl/the-marcus-wallenberg-laboratory-for-sound-and-vibration-research-mwl-1.34565>`_\.

**The package features:**

   - An easy-to-set-up three-step decomposition scheme for experiments and simulations
   - Predefined wavenumbers, acoustic mode shapes, and attenuation models for the most common geometries
   - Preparation of decomposition matrices based on auto-detected cut-on modes
   - High customizability



Installation
------------

The module can be easily installed using PiPy:

.. code-block:: console

   >> pip install acdecom

The source code is available at `GitHub <https://github.com/ssackMWL/acdecom>`_\.

Documentation
-------------

.. toctree::
   :maxdepth: 1

   Theory and Definitions <theory/theory>
   Examples <auto_examples/index>
   The WaveGuide Class <_autosummary/acdecom.WaveGuide>

Acknowledgement
---------------

This project was conducted as part of a project within the Competence Center for Gas Exchange (CCGEx) at KTH. The
authors would like to acknowledge the Swedish Energy Agency, Volvo Cars, Volvo GTT, Scania, BorgWarner Turbo Systems
Engineering, and Wärtsilä for their support and contributions. The authors also wish to acknowledge the financial
support of the European Commission provided in the framework of the FP7 Collaborative Project IDEALVENT
(Grant Agreement 314066).


Icons made by `iconixar <https://www.flaticon.com/free-icon/sound-wave_3225510>`_ from  `www.flaticon.com <https://www.flaticon.com/>`_\.
Some of the artwork was found on `https://www.pexels.com/ <https://www.pexels.com>`_\.

The Module
----------
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst

   acdecom
   acdecom.WaveGuide