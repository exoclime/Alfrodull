
#include "atomic_add.h"
#include "physics_constants.h"

// calculates the integrated upwards and downwards fluxes
__global__ void integrate_flux_double(
        double* deltalambda, 
        double* F_down_tot, 
        double* F_up_tot, 
        double* F_net, 
        double* F_down_wg, 
        double* F_up_wg,
        double* F_dir_wg,
        double* F_down_band, 
        double* F_up_band, 
        double* F_dir_band,
        double* gauss_weight,
        int 	nbin, 
        int 	numinterfaces, 
        int 	ny
){

    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = threadIdx.z;

    while(x < nbin && y < ny && i < numinterfaces){

        while(x < nbin && y < ny && i < numinterfaces){

            F_up_tot[i] = 0;
            F_down_tot[i] = 0;

            F_dir_band[x + nbin * i] = 0;
            F_up_band[x + nbin * i] = 0;
            F_down_band[x + nbin * i] = 0;

            x += blockDim.x;
        }
        x = threadIdx.x;
        i += blockDim.z;	
    }
    __syncthreads();

    i = threadIdx.z;

    while(x < nbin && y < ny && i < numinterfaces){

        while(x < nbin && y < ny && i < numinterfaces){

            while(x < nbin && y < ny && i < numinterfaces){
                
                atomicAdd_double(&(F_dir_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_dir_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_double(&(F_up_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_up_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_double(&(F_down_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_down_wg[y + ny * x + ny * nbin * i]);
                
                x += blockDim.x;
            }
            x = threadIdx.x;
            y += blockDim.y;
        }
        y = threadIdx.y;
        i += blockDim.z;
    }
    __syncthreads();
    
    i = threadIdx.z;

    while(x < nbin && y == 0 && i < numinterfaces){
        
        while(x < nbin && y == 0 && i < numinterfaces){

            atomicAdd_double(&(F_up_tot[i]), F_up_band[x + nbin * i] * deltalambda[x]);
            atomicAdd_double(&(F_down_tot[i]), (F_dir_band[x + nbin * i] + F_down_band[x + nbin * i]) * deltalambda[x]);

            x += blockDim.x;
        }
        x = threadIdx.x;
        i += blockDim.z;	
    }
    __syncthreads();

    i = threadIdx.z;

    while(x < nbin && y < 1 && i < numinterfaces){
        F_net[i] = F_up_tot[i] - F_down_tot[i];
        i += blockDim.z;
    }
}


// calculates the direct beam flux with geometric zenith angle correction, isothermal version
__global__ void fdir_iso(
        double* 	F_dir_wg,
        double* 	planckband_lay,
        double* 	delta_tau_wg,
        double* 	z_lay,
        double 	mu_star,
        double	R_planet,
        double 	R_star, 
        double 	a,
        int		dir_beam,
        int		geom_zenith_corr,
        int 	ninterface,
        int 	nbin,
        int 	ny
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < ninterface && x < nbin && y < ny) {

        // the stellar intensity at TOA
        double I_dir = ((R_star / a)*(R_star / a)) * PI * planckband_lay[(ninterface - 1) + x * (ninterface-1+2)];

        // initialize each flux value
        F_dir_wg[y + ny * x + ny * nbin * i]  = -dir_beam * mu_star * I_dir;

        double mu_star_layer_j;

        // flux values lower that TOA will now be attenuated depending on their location
        for(int j = ninterface - 2; j >= i; j--){
            
            if(geom_zenith_corr == 1){
            mu_star_layer_j  = - sqrt(1.0 - pow((R_planet + z_lay[i])/(R_planet+z_lay[j]), 2.0) * (1.0 - pow(mu_star, 2.0)));
            }
            else{
                mu_star_layer_j = mu_star;
            }

            // direct stellar flux	
            F_dir_wg[y+ny*x+ny*nbin*i] *= exp(delta_tau_wg[y+ny*x + ny*nbin*j] / mu_star_layer_j);
        }
    }
}


// calculates the direct beam flux with geometric zenith angle correction, non-isothermal version
__global__ void fdir_noniso(
        double* 	F_dir_wg,
        double* 	Fc_dir_wg,
        double* 	planckband_lay,
        double* 	delta_tau_wg_upper,
        double* 	delta_tau_wg_lower,
        double* 	z_lay,
        double 	mu_star,
        double	R_planet,
        double 	R_star, 
        double 	a,
        int		dir_beam,
        int		geom_zenith_corr,
        int 	ninterface,
        int 	nbin,
        int 	ny
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < ninterface && x < nbin && y < ny) {

        // the stellar intensity at TOA
        double I_dir = ((R_star / a)*(R_star / a)) * PI * planckband_lay[(ninterface - 1) + x * (ninterface-1+2)];

        // initialize each flux value
        F_dir_wg[y + ny * x + ny * nbin * i]  = -dir_beam * mu_star * I_dir;

        double mu_star_layer_j;

        // flux values lower that TOA will now be attenuated depending on their location
        for(int j = ninterface - 2; j >= i; j--){

            if(geom_zenith_corr == 1){
                mu_star_layer_j  = - sqrt(1.0 - pow((R_planet + z_lay[i])/(R_planet+z_lay[j]), 2.0) * (1.0 - pow(mu_star, 2.0)));
            }
            else{
                mu_star_layer_j = mu_star;
            }
            
            double delta_tau = delta_tau_wg_upper[y+ny*x + ny*nbin*j] + delta_tau_wg_lower[y+ny*x + ny*nbin*j];
            
            // direct stellar flux
            Fc_dir_wg[y+ny*x+ny*nbin*i] = F_dir_wg[y+ny*x+ny*nbin*i] * exp(delta_tau_wg_upper[y+ny*x + ny*nbin*j] / mu_star_layer_j);
            F_dir_wg[y+ny*x+ny*nbin*i] *= exp(delta_tau / mu_star_layer_j);
        }
    }
}





// calculation of the spectral fluxes, isothermal case with emphasis on on-the-fly calculations
__global__ void fband_iso_notabu(
				 double* F_down_wg, // out
				 double* F_up_wg, // out
				 double* F_dir_wg, // in
				 double* planckband_lay, // in
				 double* w_0, // in
				 double* delta_tau_wg, // in
				 double* M_term, // in
				 double* N_term, // in
				 double* P_term, // in
				 double* G_plus, // in
				 double* G_minus, // in
				 double* g_0_tot_lay, 
				 double 	g_0,
				 int 	singlewalk, 
				 double 	Rstar, 
				 double 	a, 
				 int 	numinterfaces, 
				 int 	nbin, 
				 double 	f_factor, 
				 double 	mu_star,
				 int 	ny, 
				 double 	epsi,
				 double 	w_0_limit,
				 int 	dir_beam,
				 int 	clouds,
				 double   albedo
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nbin && y < ny) {
        
        for (int i = numinterfaces - 1; i >= 0; i--){

            if (i == numinterfaces - 1) {
                
                // flux at TOA (without direct irradiation beam)
                F_down_wg[y + ny * x + ny * nbin * i] = (1.0 - dir_beam) * f_factor * ((Rstar / a)*(Rstar / a)) * PI * planckband_lay[i + x * (numinterfaces-1+2)];
            }
            else {
                
                double w0 = w_0[y+ny*x + ny*nbin*i];
                double del_tau = delta_tau_wg[y+ny*x + ny*nbin*i];
                double M = M_term[y+ny*x + ny*nbin*i];
                double N = N_term[y+ny*x + ny*nbin*i];
                double P = P_term[y+ny*x + ny*nbin*i];
                double G_pl = G_plus[y+ny*x + ny*nbin*i];
                double G_min = G_minus[y+ny*x + ny*nbin*i];
                double g0 = g_0;	
                if(clouds == 1){
                    g0 = g_0_tot_lay[x + nbin * i];
                }
                
                if(w0 > w_0_limit){
                    // w0 = 1 solution
                    double first_fraction = (F_up_wg[y+ny*x+ny*nbin*i] - F_down_wg[y+ny*x+ny*nbin*(i+1)]) * 1.0/epsi * (1.0 - g0)*del_tau/(1.0/epsi * (1.0 - g0) * del_tau + 2.0);
                            
                    double large_bracket = (mu_star/epsi - 1.0) * F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) + (1.0/epsi * (1.0 - g0) * del_tau - mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star);
                    
                    double direct_terms = mu_star/(1.0/epsi * (1.0 - g0) * del_tau + 2.0) * large_bracket;
                            
                    F_down_wg[y+ny*x+ny*nbin*i] = F_down_wg[y+ny*x+ny*nbin*(i+1)] + first_fraction + direct_terms;

                }
                else{
                    // isothermal solution
                    double flux_terms = P * F_down_wg[y+ny*x+ny*nbin*(i+1)] - N * F_up_wg[y+ny*x+ny*nbin*i];
                    
                    double planck_terms = planckband_lay[i+x*(numinterfaces-1+2)] * (N + M - P);
                    
                    double direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min * M + G_pl * N) - F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) * P * G_min;
                    
                    F_down_wg[y+ny*x+ny*nbin*i] = 1.0 / M * (flux_terms + 2.0 * PI * epsi * planck_terms + direct_terms);
                }
            }
        }

        for (int i = 0; i < numinterfaces; i++){

            if (i == 0){
                
                double reflected_part = albedo * (F_dir_wg[y+ny*x+ny*nbin*i] + F_down_wg[y+ny*x+ny*nbin* i]);
                
                // this is the surface emission. it now comes with the emissivity e = (1 - albedo)
                double internal_part = (1.0 - albedo) * PI * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
                
                F_up_wg[y+ny*x+ny*nbin* i] = reflected_part + internal_part; // internal_part comprises the interior heat plus the surface emission
            }
            else {
                
                double w0 = w_0[y+ny*x + ny*nbin*(i-1)];
                double del_tau = delta_tau_wg[y+ny*x + ny*nbin*(i-1)];
                double M = M_term[y+ny*x + ny*nbin*(i-1)];
                double N = N_term[y+ny*x + ny*nbin*(i-1)];
                double P = P_term[y+ny*x + ny*nbin*(i-1)];
                double G_pl = G_plus[y+ny*x + ny*nbin*(i-1)];
                double G_min = G_minus[y+ny*x + ny*nbin*(i-1)];
                double g0 = g_0;
                if(clouds == 1){
                    g0 = g_0_tot_lay[x + nbin * (i-1)];
                }

                if(w0 > w_0_limit){
                    // w0 = 1 solution
                    double first_fraction = (F_down_wg[y+ny*x+ny*nbin*i] - F_up_wg[y+ny*x+ny*nbin*(i-1)]) * 1.0/epsi * (1.0-g0)*del_tau / (1.0/epsi * (1.0-g0)*del_tau + 2.0);
                    
                    double large_bracket = (mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) - (1.0/epsi * (1.0 - g0) * del_tau + mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star);
                    
                    double direct_terms = mu_star/(1.0/epsi * (1.0 - g0) * del_tau + 2.0) * large_bracket;
                    
                    F_up_wg[y+ny*x+ny*nbin*i] = F_up_wg[y+ny*x+ny*nbin*(i-1)] + first_fraction + direct_terms;
                }
                else{
                    // isothermal solution
                    double flux_terms = P * F_up_wg[y+ny*x+ny*nbin*(i-1)] - N * F_down_wg[y+ny*x+ny*nbin*i];
                    
                    double planck_terms = planckband_lay[(i-1)+x*(numinterfaces-1+2)] * (N + M - P);
                    
                    double direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min * N + G_pl * M) - F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P * G_pl;
                    
                    F_up_wg[y+ny*x+ny*nbin*i] = 1.0 / M * (flux_terms + 2.0 * PI * epsi * planck_terms + direct_terms);
                }
            }
        }
    }
}


// calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
__global__ void fband_noniso_notabu(
        double* F_down_wg, 
        double* F_up_wg, 
        double* Fc_down_wg, 
        double* Fc_up_wg,
        double* F_dir_wg,
        double* Fc_dir_wg,
        double* planckband_lay, 
        double* planckband_int,
        double* w_0_upper,
        double* w_0_lower,
        double* delta_tau_wg_upper,
        double* delta_tau_wg_lower,
        double* M_upper,
        double* M_lower,
        double* N_upper,
        double* N_lower,
        double* P_upper,
        double* P_lower,
        double* G_plus_upper,
        double* G_plus_lower,
        double* G_minus_upper,
        double* G_minus_lower,
        double* g_0_tot_lay,
        double* g_0_tot_int,
        double 	g_0,
        int 	singlewalk, 
        double 	Rstar, 
        double 	a, 
        int 	numinterfaces,
        int 	nbin, 
        double 	f_factor,
        double 	mu_star,
        int 	ny,
        double 	epsi,
        double 	w_0_limit,
        double 	delta_tau_limit,
        int 	dir_beam,
        int 	clouds,
        double   albedo,
        double*	trans_wg_upper,
        double* trans_wg_lower
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nbin && y < ny) {

        for (int i = numinterfaces - 1; i >= 0; i--){

            if (i == numinterfaces - 1) {
                F_down_wg[y + ny * x + ny * nbin * i] = (1.0 - dir_beam) * f_factor * ((Rstar / a)*(Rstar / a)) * PI * planckband_lay[i + x * (numinterfaces-1+2)];
            }
            else {
                // upper part of layer
                double w0_up = w_0_upper[y+ny*x + ny*nbin*i];
                double del_tau_up = delta_tau_wg_upper[y+ny*x + ny*nbin*i];
                double M_up = M_upper[y+ny*x + ny*nbin*i];
                double N_up = N_upper[y+ny*x + ny*nbin*i];
                double P_up = P_upper[y+ny*x + ny*nbin*i];
                double G_pl_up = G_plus_upper[y+ny*x + ny*nbin*i];
                double G_min_up = G_minus_upper[y+ny*x + ny*nbin*i];
                double g0_up = g_0;
                if(clouds == 1){
                    g0_up = (g_0_tot_lay[x + nbin * i] + g_0_tot_int[x + nbin * (i+1)]) / 2.0;
                }
                
                // lower part of layer
                double w0_low = w_0_lower[y+ny*x + ny*nbin*i];
                double del_tau_low = delta_tau_wg_lower[y+ny*x + ny*nbin*i];
                double M_low = M_lower[y+ny*x + ny*nbin*i];
                double N_low = N_lower[y+ny*x + ny*nbin*i];
                double P_low = P_lower[y+ny*x + ny*nbin*i];
                double G_pl_low = G_plus_lower[y+ny*x + ny*nbin*i];
                double G_min_low = G_minus_lower[y+ny*x + ny*nbin*i];
                double g0_low = g_0;
                if(clouds == 1){
                    g0_low = (g_0_tot_int[x + nbin * i] + g_0_tot_lay[x + nbin * i]) / 2.0;
                }

                if(w0_up > w_0_limit){
                    // w0 = 1 solution
                    double first_fraction = (Fc_up_wg[y+ny*x+ny*nbin*i] - F_down_wg[y+ny*x+ny*nbin*(i+1)]) * 1.0/epsi * (1.0-g0_up)*del_tau_up / (1.0/epsi * (1.0-g0_up)*del_tau_up + 2.0);

                    double large_bracket = (mu_star/epsi - 1.0) * F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) + (1.0/epsi * (1.0 - g0_up) * del_tau_up - mu_star/epsi + 1.0) * Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star);
                                        
                    double direct_terms = mu_star/(1.0/epsi * (1.0 - g0_up) * del_tau_up + 2.0) * large_bracket;

                    Fc_down_wg[y+ny*x+ny*nbin*i] = F_down_wg[y+ny*x+ny*nbin*(i+1)] + first_fraction + direct_terms;
                }
                else{
                    double flux_terms;
                    double planck_terms;
                    double direct_terms;

                    if(del_tau_up < delta_tau_limit){
                        // the isothermal solution
                        flux_terms = P_up * F_down_wg[y+ny*x+ny*nbin*(i+1)] - N_up * Fc_up_wg[y+ny*x+ny*nbin*i];

                        planck_terms = (planckband_int[(i+1)+x*numinterfaces] + planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (N_up + M_up - P_up);

                        direct_terms = Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_up * M_up + G_pl_up * N_up) - F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) * G_min_up * P_up;
                    }
                    else{
                        // the non-isothermal solution
                        double pgrad_up = (planckband_lay[i + x * (numinterfaces-1+2)] - planckband_int[(i + 1) + x * numinterfaces]) / del_tau_up;

                        flux_terms = P_up * F_down_wg[y+ny*x+ny*nbin*(i+1)] - N_up * Fc_up_wg[y+ny*x+ny*nbin*i];

                        planck_terms = planckband_lay[i+x*(numinterfaces-1+2)] * (M_up + N_up) - planckband_int[(i+1)+x*numinterfaces] * P_up + epsi/(1.0-w0_up*g0_up) * pgrad_up * (P_up - M_up + N_up);

                        direct_terms = Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_up * M_up + G_pl_up * N_up) - F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) * P_up * G_min_up;
                    }
                    Fc_down_wg[y+ny*x+ny*nbin*i] = 1.0 / M_up * (flux_terms + 2.0 * PI * epsi * planck_terms + direct_terms);
                }

                if(w0_low > w_0_limit){
                    // w0 = 1 solution
                    double first_fraction = (F_up_wg[y+ny*x+ny*nbin*i] - Fc_down_wg[y+ny*x+ny*nbin*i]) * 1.0/epsi * (1.0-g0_low)*del_tau_low / (1.0/epsi * (1.0-g0_low)*del_tau_low + 2.0);

                    double large_bracket = (mu_star/epsi - 1.0) * Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) + (1.0/epsi * (1.0 - g0_low) * del_tau_low - mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star);
                                                            
                    double direct_terms = mu_star/(1.0/epsi * (1.0 - g0_low) * del_tau_low + 2.0) * large_bracket;

                    F_down_wg[y+ny*x+ny*nbin*i] = Fc_down_wg[y+ny*x+ny*nbin*i] + first_fraction + direct_terms;
                }
                else{
                    double flux_terms;
                    double planck_terms;
                    double direct_terms;

                    if(del_tau_low < delta_tau_limit){
                        // isothermal solution
                        flux_terms = P_low * Fc_down_wg[y+ny*x+ny*nbin*i] - N_low * F_up_wg[y+ny*x+ny*nbin*i];

                        planck_terms = (planckband_int[i+x*numinterfaces] + planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (N_low + M_low - P_low);

                        direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_low * M_low + G_pl_low * N_low) - Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * P_low * G_min_low;
                    }
                    else{
                        // non-isothermal solution
                        double pgrad_low = (planckband_int[i + x * numinterfaces] - planckband_lay[i + x * (numinterfaces-1+2)]) / del_tau_low;

                        flux_terms = P_low * Fc_down_wg[y+ny*x+ny*nbin*i] - N_low * F_up_wg[y+ny*x+ny*nbin*i];

                        planck_terms = planckband_int[i+x*numinterfaces] * (M_low + N_low) - planckband_lay[i+x*(numinterfaces-1+2)] * P_low + epsi/(1.0-w0_low*g0_low) * pgrad_low * (P_low - M_low + N_low) ;

                        direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_low * M_low + G_pl_low * N_low) - Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * P_low * G_min_low;						
                    }
                    F_down_wg[y+ny*x+ny*nbin*i] = 1.0 / M_low * (flux_terms + 2.0*PI*epsi*planck_terms + direct_terms);
                }
            }
        }

        __syncthreads();
        
        for (int i = 0; i < numinterfaces; i++){
            
            if (i == 0){
                
                double reflected_part = albedo * (F_dir_wg[y+ny*x+ny*nbin*i] + F_down_wg[y+ny*x+ny*nbin* i]);
                
                // this is the surface emission. it now comes with the emissivity e = (1 - albedo)
                double internal_part = (1.0 - albedo) * PI * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
                
                F_up_wg[y+ny*x+ny*nbin* i] = reflected_part + internal_part; // internal_part comprises the interior heat plus the surface emission
            }
            else {
                // lower part of layer
                double w0_low = w_0_lower[y+ny*x + ny*nbin*(i-1)];
                double del_tau_low = delta_tau_wg_lower[y+ny*x + ny*nbin*(i-1)];
                double M_low = M_lower[y+ny*x + ny*nbin*(i-1)];
                double N_low = N_lower[y+ny*x + ny*nbin*(i-1)];
                double P_low = P_lower[y+ny*x + ny*nbin*(i-1)];
                double G_pl_low = G_plus_lower[y+ny*x + ny*nbin*(i-1)];
                double G_min_low = G_minus_lower[y+ny*x + ny*nbin*(i-1)];
                double g0_low = g_0;
                if(clouds == 1){
                    g0_low = (g_0_tot_int[x + nbin * (i-1)] + g_0_tot_lay[x + nbin * (i-1)]) / 2.0;
                }
                                
                // upper part of layer
                double w0_up = w_0_upper[y+ny*x + ny*nbin*(i-1)];
                double del_tau_up = delta_tau_wg_upper[y+ny*x + ny*nbin*(i-1)];
                double M_up = M_upper[y+ny*x + ny*nbin*(i-1)];
                double N_up = N_upper[y+ny*x + ny*nbin*(i-1)];
                double P_up = P_upper[y+ny*x + ny*nbin*(i-1)];
                double G_pl_up = G_plus_upper[y+ny*x + ny*nbin*(i-1)];
                double G_min_up = G_minus_upper[y+ny*x + ny*nbin*(i-1)];
                double g0_up = g_0;
                if(clouds == 1){
                    g0_up = (g_0_tot_lay[x + nbin * (i-1)] + g_0_tot_int[x + nbin * i]) / 2.0;
                }

                if(w0_low > w_0_limit){
                    // w0 = 1 solution
                    double first_fraction = (Fc_down_wg[y+ny*x+ny*nbin*(i-1)] - F_up_wg[y+ny*x+ny*nbin*(i-1)]) * 1.0/epsi * (1.0-g0_low)*del_tau_low / (1.0/epsi * (1.0-g0_low)*del_tau_low + 2.0);

                    double large_bracket = (mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) - (1.0/epsi * (1.0 - g0_low) * del_tau_low + mu_star/epsi + 1.0) * Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star);
                    
                    double direct_terms = mu_star/(1.0/epsi * (1.0 - g0_low) * del_tau_low + 2.0) * large_bracket;

                    Fc_up_wg[y+ny*x+ny*nbin*(i-1)] = F_up_wg[y+ny*x+ny*nbin*(i-1)] + first_fraction + direct_terms;
                }
                else{
                    double flux_terms;
                    double planck_terms;
                    double direct_terms;

                    if(del_tau_low < delta_tau_limit){
                        // isothermal solution
                        flux_terms = P_low * F_up_wg[y+ny*x+ny*nbin*(i-1)] - N_low * Fc_down_wg[y+ny*x+ny*nbin*(i-1)];

                        planck_terms = ( (planckband_int[(i-1)+x*numinterfaces] + planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (N_low + M_low - P_low) ) ;

                        direct_terms = Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * (G_min_low * N_low + G_pl_low * M_low) - F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P_low * G_pl_low;

                    }
                    else{
                        // non-isothermal solution
                        double pgrad_low = (planckband_int[(i-1) + x * numinterfaces] - planckband_lay[(i-1) + x * (numinterfaces-1+2)]) / del_tau_low;

                        flux_terms = P_low * F_up_wg[y+ny*x+ny*nbin*(i-1)] - N_low * Fc_down_wg[y+ny*x+ny*nbin*(i-1)];

                        planck_terms = planckband_lay[(i-1)+x*(numinterfaces-1+2)] * (M_low + N_low) - planckband_int[(i-1)+x*numinterfaces] * P_low + epsi/(1.0-w0_low*g0_low) * pgrad_low * (M_low - P_low - N_low);

                        direct_terms = Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * (G_min_low * N_low + G_pl_low * M_low) - F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P_low * G_pl_low;
                    }
                    Fc_up_wg[y+ny*x+ny*nbin*(i-1)] = 1.0 / M_low * (flux_terms + 2.0*PI*epsi*planck_terms + direct_terms);
                }

                if(w0_up > w_0_limit){
                    // w0 = 1 solution
                    double first_fraction = (F_down_wg[y+ny*x+ny*nbin*i] - Fc_up_wg[y+ny*x+ny*nbin*(i-1)]) * 1.0/epsi * (1.0-g0_up)*del_tau_up / (1.0/epsi * (1.0-g0_up)*del_tau_up + 2.0);

                    double large_bracket = (mu_star/epsi + 1.0) * Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) - (1.0/epsi * (1.0 - g0_up) * del_tau_up + mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star);
                                        
                    double direct_terms = mu_star/(1.0/epsi * (1.0 - g0_up) * del_tau_up + 2.0) * large_bracket;

                    F_up_wg[y+ny*x+ny*nbin*i] = Fc_up_wg[y+ny*x+ny*nbin*(i-1)] + first_fraction + direct_terms;
                }
                else{
                    double flux_terms;
                    double planck_terms;
                    double direct_terms;

                    if(del_tau_up < delta_tau_limit){
                        // isothermal solution
                        flux_terms = P_up * Fc_up_wg[y+ny*x+ny*nbin*(i-1)] - N_up * F_down_wg[y+ny*x+ny*nbin*i];

                        planck_terms = (planckband_int[i+x*numinterfaces] + planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (N_up + M_up - P_up);

                        direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_up * N_up + G_pl_up * M_up) - Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P_up * G_pl_up;
                    }
                    else{
                        // non-isothermal solution
                        double pgrad_up = (planckband_lay[(i-1) + x * (numinterfaces-1+2)] - planckband_int[i + x * numinterfaces]) / del_tau_up;

                        flux_terms = P_up * Fc_up_wg[y+ny*x+ny*nbin*(i-1)] - N_up * F_down_wg[y+ny*x+ny*nbin*i];

                        planck_terms = planckband_int[i+x*numinterfaces] * (M_up + N_up) - planckband_lay[(i-1)+x*(numinterfaces-1+2)] * P_up + epsi/(1.0-w0_up*g0_up) * pgrad_up * (M_up - P_up - N_up);

                        direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_up * N_up + G_pl_up * M_up) - Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P_up * G_pl_up;	
                    }
                    F_up_wg[y+ny*x+ny*nbin*i] = 1.0 / M_up * (flux_terms + 2.0*PI*epsi*planck_terms + direct_terms);
                }
            }
        }
    }
}

