
from dataclasses import dataclass, field
import numpy as np
from source.opacity_data import OpacityData
from source.opacity_species import OpacitySpecies
from source import phys_const as pc



#special class for the H- continuum opacity
#it has no tabulated data, the opacity is calculated on the fly
@dataclass
class OpacitySpeciesHminus(OpacitySpecies):

  #just to make sure we have the correct entries for the H- opacity
  def __post_init__(self):
    self.name = 'H-'
    self.absorbing = True
    self.scattering = False
    self.fastchem_symbol = ['H', 'e_minus']
    self.molecular_weight = [1.00794, 0.0]

    self.wavelength = np.empty(0)


  #re-definition of the scanDirectory function from the parent class
  def scanDirectory(self):

    print(self.name)
    print('opacity grid:')
    print('  no tabulated opacity...\n')

    return None



  #re-definition of the getOpacity function from the parent class
  def getOpacity(self, temperature, pressure_log, mean_molecular_weight, sampling_indices, wavelength):

    if self.wavelength.size == 0:
      self.wavelength = np.copy(wavelength) * 1e4
    
    pressure = 10**pressure_log


    #create an OpacityData object
    #note that the mixing part assumes the opacity to be in log10
    opacity_data = OpacityData("", temperature, pressure)
    opacity_data.opacity = np.log10(self.calcHminusOpac(temperature, pressure))


    return opacity_data



  def calcHminusOpac(self, temperature, pressure):

    kappa_bf = self.Hminus_boundfree(temperature)
    kappa_ff = self.Hminus_freefree(temperature)

    #the pressure here is supposed to be the electron partial pressure
    #therefore, the result is muliplied by the electon mixing ratio
    #later during the mixing process
    #we also convert the cross section to opacity
    opacity = (kappa_bf + kappa_ff) * pressure / (pc.M_H * pc.AMU)
 

    return opacity



  def Hminus_boundfree(self, temperature):
    # alpha value in the John 1988 paper is wrong. It should be alpha = c * h / k = 1.439e4 micron K
    alpha = 1.439e4
    lambda_0 = 1.6419   #photo-detachment threshold
    
    C_n = np.array([0.0, 152.519, 49.534, -118.858, 92.536, -34.194, 4.982])
  
  
    kappa_bf = np.zeros(self.wavelength.size)
    
    #indices for the wavelength region of the bound-free opacity
    bf_wavelengths = np.where((self.wavelength <= lambda_0) & (self.wavelength >=0.125))[0]


    kappa_bf[bf_wavelengths] = 1e-18 * self.wavelength[bf_wavelengths]**3 \
                                     * (1.0/self.wavelength[bf_wavelengths] - 1.0/lambda_0)**1.5 \
                                     * sum(C_n[i] * (1.0/self.wavelength[bf_wavelengths] - 1.0/lambda_0)**((float(i)-1.0)/2.0) for i in range(1,7))

    kappa_bf[bf_wavelengths] = 0.750 * temperature**(-2.5) * np.exp(alpha/lambda_0 / temperature) \
                     * (1.0 - np.exp(-alpha/self.wavelength[bf_wavelengths] / temperature)) * kappa_bf[bf_wavelengths]


    return kappa_bf



  def Hminus_freefree(self, temperature):
    A_n1 = np.array([0.0, 0.0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830])
    B_n1 = np.array([0.0, 0.0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170])
    C_n1 = np.array([0.0, 0.0, -2054.2910, 8746.5230, -13651.1050, 8624.9700, -1863.8650])
    D_n1 = np.array([0.0, 0.0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880])
    E_n1 = np.array([0.0, 0.0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880])
    F_n1 = np.array([0.0, 0.0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850])

    #for wavelengths between 0.1823 micron and 0.3645 micron
    A_n2 = np.array([0.0, 518.1021, 473.2636, -482.2089, 115.5291, 0.0, 0.0])
    B_n2 = np.array([0.0, -734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0])
    C_n2 = np.array([0.0, 1021.1775, -1977.3395, 1096.8827, -245.6490, 0.0, 0.0])
    D_n2 = np.array([0.0, -479.0721, 922.3575, -521.1341, 114.2430, 0.0, 0.0])
    E_n2 = np.array([0.0, 93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0])
    F_n2 = np.array([0.0, -6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0])


    kappa_ff = np.zeros(self.wavelength.size)

    #the indices for the two different wavelength regions
    ff_wavelengths_1 = np.where(self.wavelength >= 0.3645)[0]
    ff_wavelengths_2 = np.where((self.wavelength >= 0.1823) & (self.wavelength < 0.3645))[0]
  
    kappa_ff[ff_wavelengths_1] = 1e-29 * sum((5040.0/temperature)**((i+1)/2.0)
                                 * (self.wavelength[ff_wavelengths_1]**2 * A_n1[i] 
                                    + B_n1[i] 
                                    + C_n1[i]/self.wavelength[ff_wavelengths_1] 
                                    + D_n1[i]/self.wavelength[ff_wavelengths_1]**2 
                                    + E_n1[i]/self.wavelength[ff_wavelengths_1]**3 
                                    + F_n1[i]/self.wavelength[ff_wavelengths_1]**4) for i in range(1, 7))

    kappa_ff[ff_wavelengths_2] = 1e-29 * sum((5040.0/temperature)**((i+1)/2.0) 
                                 * (self.wavelength[ff_wavelengths_2]**2 * A_n2[i] 
                                    + B_n2[i] 
                                    + C_n2[i]/self.wavelength[ff_wavelengths_2] 
                                    + D_n2[i]/self.wavelength[ff_wavelengths_2]**2 
                                    + E_n2[i]/self.wavelength[ff_wavelengths_2]**3 
                                    + F_n2[i]/self.wavelength[ff_wavelengths_2]**4) for i in range(1, 7))

  
    return kappa_ff
