
from dataclasses import dataclass, field
import numpy as np
from source.opacity_data import OpacityData
from source.opacity_species import OpacitySpecies
from source import phys_const as pc


#dataclass for the CIA coefficients
#this class deals with the HITRAN CIA data, that is given in
#cross sections/molecule/molecue
#the binary files contain the cross sections in log10
@dataclass
class OpacitySpeciesCIA(OpacitySpecies):

  log_opacities : bool = field(default=True, init=False)


  #retrieves the opacity for a given temperature & pressure at the sampling indices
  #re-defines the function from the parent class
  def getOpacity(self, temperature, pressure_log, mean_molecular_weight, sampling_indices, wavelength):
    
    if len(self.opacity_data) == 0: return None

    interpolated_data = self.interpolateOpacity(temperature, pressure_log, sampling_indices)
    
    #total mass density of the gas in g/cm3
    mass_density = 10**pressure_log * mean_molecular_weight / (pc.R_UNIV * temperature)

    #convert from CIA cross sections to opacity
    #note that the cross sections and opacity are in log10
    #hence, addition instead of multiplication
    #we also convert the opacity from cm5/g2 to cm2/g by multiplying with the gas mass density
    #during the opacity mixing, this is later scaled to the mass density of the molecule by multiplying 
    #with the corresponding mass mixing ratio
    interpolated_data.opacity +=  np.log10(mass_density/(self.molecular_weight[0] * pc.AMU * self.molecular_weight[1] * pc.AMU))


    return interpolated_data

