
from dataclasses import dataclass, field
import numpy as np
import os
from source.opacity_data import OpacityData
from source.chemical_composition import ChemicalComposition


@dataclass
class OpacitySpecies:
  name : str = field(init=True)
  absorbing : bool = field(init=True)
  scattering : bool = field(init=True)
  mixing_ratio : float = field(init=True)
  fastchem_symbol : str = field(init=True)
  molecular_weight : float = field(init=True)
  directory : str = field(init=True)

  nb_pressures : int = field(default=0, init=False)
  nb_temperatures : int = field(default=0, init=False)

  max_temperature : float = field(default=0.0, init=False)
  min_temperature : float = field(default=0.0, init=False)
  max_pressure : float = field(default=0.0, init=False)
  min_pressure : float = field(default=0.0, init=False)

  log_opacities : bool = field(default=False, init=False)

  #opacity_data : list[object] = field(default_factory=list, init=False, repr=False)
  opacity_data : [OpacityData] = field(default_factory=list, init=False, repr=False)


  #retrieves the opacity for a given temperature & pressure at the sampling indices
  #here it simply calls the interpolation routine
  def getOpacity(self, temperature, pressure_log, mean_molecular_weight, sampling_indices, wavelength):

    if len(self.opacity_data) == 0: return None

    return self.interpolateOpacity(temperature, pressure_log, sampling_indices)



  def scanDirectory(self):
    #print("Scanning directory", self.directory)
    #scan the directory for all 'bin' files and extract pressures and temperatures
    for file_name in os.listdir(self.directory):
      if file_name.endswith(".bin"):

        #print("found", file_name)

        pressure_string = file_name.split('.')[0].split("_")[-1]
        temperature_string = file_name.split('.')[0].split("_")[-2]
        file_prefix = file_name.split('.')[0].split("_")[1] + '_' + file_name.split('.')[0].split("_")[2]

        temperature = float(temperature_string)
        log_pressure = float(pressure_string[1] + '.' + pressure_string[2:] + pressure_string[2:] + pressure_string[2:] + pressure_string[2:])

        if pressure_string[0] == 'n':
          log_pressure *= -1

        log_pressure += 6 #unit conversion from bar to dyn cm-2

        self.opacity_data.append(OpacityData(self.directory+file_name, temperature, log_pressure))

    #safety check
    if (len(self.opacity_data) == 0):
      print('No opacity files for species', self.name, 'found in folder', self.directory)
      exit()

    #sort the data in ascending temperatures
    #and for each temperature in ascending pressures
    self.opacity_data = sorted(self.opacity_data, key = lambda x: (x.temperature, x.log_pressure))


    #Note: the following assumes that the temperature-pressure grid the opacities are tabulated at
    #is regular and rectangular, i.e. that each temperature has the same number of pressure points

    #number of unique temperatures and pressures
    for i in self.opacity_data:
      if i.temperature == self.opacity_data[0].temperature:
        self.nb_pressures += 1

      if i.log_pressure == self.opacity_data[0].log_pressure:
        self.nb_temperatures += 1

    self.max_temperature = self.opacity_data[-1].temperature
    self.min_temperature = self.opacity_data[0].temperature

    self.max_pressure = self.opacity_data[-1].log_pressure
    self.min_pressure = self.opacity_data[0].log_pressure

    print(self.name)
    print('opacity grid:')
    print('  number of files:',len(self.opacity_data))
    print("  number of temperatures:", self.nb_temperatures, " number of pressures:", self.nb_pressures)
    print("  temperature min:", self.min_temperature, "max:", self.max_temperature)
    print("  pressure min:", self.min_pressure, "max:", self.max_pressure, '\n')

    return None



  def interpolateOpacity(self, temperature, pressure, sampling_indices):

    interpol_temperature = np.copy(temperature)
    interpol_pressure = np.copy(pressure)

    #restrict the interpolation to the borders of the tabulated data
    if interpol_temperature > self.max_temperature: interpol_temperature = self.max_temperature
    if interpol_temperature < self.min_temperature: interpol_temperature = self.min_temperature

    if interpol_pressure > self.max_pressure: interpol_pressure = self.max_pressure
    if interpol_pressure < self.min_pressure: interpol_pressure = self.min_pressure


    interpolation_points = self.findInterpolationPoints(interpol_temperature, interpol_pressure)

    #debug output in case of problems
    # if len(interpolation_points) == 0:
    #   print("interpolation values for T =", interpol_temperature, ", p =", pressure, "not found")
    # else:
    #   print("Found the following interpolation points for T=", interpol_temperature, "and p=",interpol_pressure)
    #   print(interpolation_points[0].temperature, interpolation_points[0].log_pressure)
    #   print(interpolation_points[1].temperature, interpolation_points[1].log_pressure)
    #   print(interpolation_points[2].temperature, interpolation_points[2].log_pressure)
    #   print(interpolation_points[3].temperature, interpolation_points[3].log_pressure)

    interpolated_data = self.interpolateData(interpolation_points, interpol_temperature, interpol_pressure, sampling_indices)

    return interpolated_data



  #finds the four closest data points in the p-T grid for the desired interpolation temperature and pressure
  #the function assumes that the grid is rectangular, i.e. that the number of pressure points for each temperature are equal
  #returns a list of OpacityData objects for the four points:
  #the first two are the two different temperatures for the frist pressure
  #the second two are the two temperatures for the second pressure
  #some of the points might be identical if the interpolation temperature and pressure coincide with grid points
  #function returns empty list when the interpolation temperature or pressure are out of range of the grid
  def findInterpolationPoints(self, interpol_temperature, interpol_pressure):

    #out-of-range check, should actually never happen
    if (interpol_pressure > self.opacity_data[-1].log_pressure or interpol_pressure < self.opacity_data[0].log_pressure
      or interpol_temperature > self.opacity_data[-1].temperature or interpol_temperature < self.opacity_data[0].temperature):
      return []


    #first, find the two grid temperatures T1, T2 that are T1 < Tint and T2 > Tint or T1 = T2 = Tint
    for i in range(0, len(self.opacity_data), self.nb_pressures):
      if self.opacity_data[i].temperature == interpol_temperature:
        temperature_index1 = i
        temperature_index2 = i
        break
      elif self.opacity_data[i].temperature < interpol_temperature and self.opacity_data[i+self.nb_pressures].temperature > interpol_temperature:
        temperature_index1 = i
        temperature_index2 = i+self.nb_pressures
        break


    #next, find the two grid pressures p1, p2 that are p1 < p_int and p2 > p_int or p1 = p2 = p_int
    for i in range(0, self.nb_pressures):
      if self.opacity_data[i].log_pressure == interpol_pressure:
        pressure_index1 = i
        pressure_index2 = i
        break
      elif self.opacity_data[i].log_pressure < interpol_pressure and self.opacity_data[i+1].log_pressure > interpol_pressure:
        pressure_index1 = i
        pressure_index2 = i+1
        break


    #create the list of OpacityData objects
    interpolation_points = [self.opacity_data[temperature_index1 + pressure_index1], self.opacity_data[temperature_index2+pressure_index1],
                            self.opacity_data[temperature_index1 + pressure_index2], self.opacity_data[temperature_index2+pressure_index2]]

    return interpolation_points



  #does the actual interpolatiin in temperature and pressure based on the four grid points
  #it will first perform two temperature interpolations and then an interpolation in pressure of the results
  #uses linear interpolation in each case
  def interpolateData(self, data_points, interpol_temperature, interpol_pressure, sampling_indices):

    #just in case...
    if len(data_points) < 4:
      return None

    #check if the opacity has been read in already
    for data in data_points:
      if data.opacity.size == 0:
        data.readOpacityFile(sampling_indices, self.log_opacities)


    #lambda function for a simple linear interpolation
    #note: it has *no* safeguard for the case x2 = x1
    linearInterpolation = lambda x1, x2, y1, y2, x : y1 + (y2 - y1) * (x - x1)/(x2 - x1)


    #temperature interpolation for the first presssure value
    if data_points[0] == data_points[1]:
      interpolation1 = np.copy(data_points[0].opacity)
    else:
      interpolation1 = linearInterpolation(data_points[0].temperature, data_points[1].temperature, data_points[0].opacity, data_points[1].opacity, interpol_temperature)


    #temperature interpolation for the second pressure value
    if data_points[2] == data_points[3]:
      interpolation2 = np.copy(data_points[2].opacity)
    else:
      interpolation2 = linearInterpolation(data_points[2].temperature, data_points[3].temperature, data_points[2].opacity, data_points[3].opacity, interpol_temperature)


    #now we interpolate in pressure
    if data_points[0] == data_points[2]:
      interpolated_data = interpolation1
    else:
      interpolated_data = linearInterpolation(data_points[0].log_pressure, data_points[2].log_pressure, interpolation1, interpolation2, interpol_pressure)


    #create an OpacityData object and return it
    opacity_data = OpacityData("", interpol_temperature, interpol_pressure)
    opacity_data.opacity = interpolated_data

    #clean up memory
    for data in data_points:
      data.opacity = np.empty(0)

    return opacity_data



  def loadAllOpacitiyData(self, sampling_indices):

    print('Reading all files for species', self.name)

    for data in self.opacity_data:
      if data.opacity.size == 0:
        data.readOpacityFile(sampling_indices, self.log_opacities)

    return None
