# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack import *


class Vacuumms(CMakePackage):
    """VACUUMMS: (Void Analysis Codes and Unix Utilities for Molecular Modeling and Simulation) is a collection of research codes for the compuational analysis of free volume in molecular structures, including the generation of code for the production of high quality ray-traced images and videos."""


    homepage = "https://github.com/frankwillmore/VACUUMMS"
    url      = "https://github.com/frankwillmore/VACUUMMS"
    git      = "https://github.com/frankwillmore/VACUUMMS.git"

    maintainers = ['frankwillmore']

    version('master', branch='master')
    version('1.0.0', '08cc1cb8b8e84e7d39078fc89a4530f7')

    depends_on('libtiff')

