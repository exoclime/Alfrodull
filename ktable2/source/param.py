# ==============================================================================
# Mini module to read parameter file
# Copyright (C) 2018 Matej Malik
# ==============================================================================
# This file is part of HELIOS.
#
#     HELIOS is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     HELIOS is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You find a copy of the GNU General Public License in the main
#     HELIOS directory under <license.txt>. If not, see
#     <http://www.gnu.org/licenses/>.
# ==============================================================================

import numpy as npy
from dataclasses import dataclass, field
from source.opacity_species import OpacitySpecies
from source.opacity_species_cia import OpacitySpeciesCIA
from source.opacity_species_special import OpacitySpeciesHminus
import argparse


class Param(object):
  """ class to read in the input parameters """

  def __init__(self):
    self.format = None
    self.building = None
    self.heliosk_path = None
    self.resampling_path = None
    self.sampling_param_path = None
    self.heliosk_format = None
    self.resolution = 50.0
    self.special_limits = [0.34, 30]
    self.fastchem_path = ''
    self.cond_path = None
    self.species_path = None
    self.final_path = ''
    self.units = None
    self.condensation = None
    self.grid_spacing = None
    self.param_file = None

    self.helios_wavenumber_step = 0.01

    self.save_sampled_data = False
    self.read_sampled_data = False
    self.sampled_file_path = None


  def parseCommandLineArguments(self):

    """ reads the input file and command line options """

    # set up command line options.
    parser = argparse.ArgumentParser(description=
                                     "The following are the possible command-line parameters for HELIOS")

    parser.add_argument('-outputdir', help='output directory', required=False)
    parser.add_argument('-chemdir', help='FastChem directory', required=False)
    parser.add_argument('-save_sampling', help='save sampled opacity data in npy file', required=False)
    parser.add_argument('-read_sampling', help='read sampled opacity data from npy file', required=False)
    parser.add_argument('-param', help='parameter file', required=False)

    args = parser.parse_args()

    # read parameter file name. If none specified, use standard name.
    if args.param:
        self.param_file = args.param
    else:
        self.param_file = "param_ktable2.dat"

    if args.read_sampling:
      self.read_sampled_data = True
      self.sampled_file_path = args.read_sampling

    if args.save_sampling:
      self.save_sampled_data = True
      self.sampled_file_path = args.save_sampling

    if self.save_sampled_data & self.read_sampled_data:
      print('I cannot read and write the sampled data at the same time :-?')
      exit()

    if args.outputdir:
      self.final_path = args.outputdir

    if args.chemdir:
      self.fastchem_path = args.chemdir

    return None


  def readParameterFile(self, file_name, species):
    try:
      with open(file_name, "r", encoding='utf-8') as param_file:
        for line in param_file:
          column = line.split()

          if column:
            if column[0] == "format":
              self.format = column[2]

            elif column[0] == "path" and column[2] == "HELIOS-K":
              self.heliosk_path = column[5]

            elif column[0] == "sampling" and column[1] == "wavelength":
              if self.format == "sampling":
                  self.resolution = float(column[4])
                  try:
                    self.special_limits = [float(column[5]), float(column[6])]
                  except ValueError:
                    pass

            elif column[0] == "ktable" and column[1] == "wavelength":
              if self.format == "ktable":
                  self.resolution = float(column[4])
                  try:
                    self.special_limits = [float(column[5]), float(column[6])]
                  except ValueError:
                    pass

            elif column[0] == "path" and column[2] == "FastChem" and self.fastchem_path == '':
              self.fastchem_path = column[5]
            elif column[0] == "final" and column[1] == "output" and self.final_path == '':
              self.final_path = column[6]
            elif column[0] == "units" and column[4] == "table":
              self.units = column[6]
              if self.units not in ["MKS", "CGS"]:
                raise ValueError("Chosen units for the opacity table unknown. Please double-check entry in the parameter file.")

            elif column[0] == "species":
              if self.read_sampled_data == False:
                self.readSpeciesData(param_file, species)
            elif column[0] == "grid" and column[1] == "spacing":
                self.grid_spacing = column[3]
                if self.grid_spacing not in ["wavenumber", "loglambda"]:
                    raise ValueError("Invalid type of grid spacing")
            elif column[0] == "cadence" and column[2] == "sampling":
                self.sampling_cadence = column[4]

    except IOError:
      print("ABORT - Param file not found!")
      raise SystemExit()

    return None


  def readSpeciesData(self, param_file, species):

    for line in param_file:

      column = line.split()

      if len(column) > 0:
        if (len(column) != 7):
          print("Line for species data, starting with", column[0], "in parameter file incomplete or wrong format.")
          exit()

        species_name = column[0]
        mixing_ratio = column[3]
        fastchem_name = column[4]


        if species_name == 'H-':
          mixing_ratio_description = column[3].split('&')

          if mixing_ratio_description[0] != 'FastChem' and mixing_ratio_description[0] != 'fastchem':
            mixing_ratios = [float(mixing_ratio_description[0]), float(mixing_ratio_description.split[1])]
          else:
            mixing_ratios = 'FastChem'

          new_species = OpacitySpeciesHminus(species_name, True, False, mixing_ratios, ['H', 'e_minus'], float(column[5]), column[6])
          species.append(new_species)
          continue

        if "CIA" not in species_name:

          if mixing_ratio != 'FastChem' and mixing_ratio != 'fastchem':
            mixing_ratio = float(mixing_ratio)
            fastchem_name = species_name

          new_species = OpacitySpecies(species_name, column[1]=='yes', column[2]=='yes', mixing_ratio, fastchem_name, float(column[5]), column[6])
          species.append(new_species)

        elif "CIA" in species_name:
          mixing_ratio_description = column[3].split('&')
          molecular_weights = [float(column[5].split('&')[0]), float(column[5].split('&')[1])]

          if mixing_ratio_description[0] != 'FastChem' and mixing_ratio_description[0] != 'fastchem':
            mixing_ratios = [float(mixing_ratio_description[0]), float(mixing_ratio_description.split[1])]
            fastchem_names = [species_name + '_1', species_name + '_2']
          else:
            mixing_ratios = 'FastChem'
            fastchem_names = column[4].split('&')


          new_species = OpacitySpeciesCIA(species_name, column[1]=='yes', False, mixing_ratios, fastchem_names, molecular_weights, column[6])
          species.append(new_species)


    print('Found the following opacity species in the parameter file:\n')

    for s in species:
      if s.absorbing == True: s.scanDirectory()

    return None
