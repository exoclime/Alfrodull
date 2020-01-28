

bool prepare_compute_flux(
		  double * dev_planckband_lay,  // csp, cse
		  double * dev_planckband_grid,  // pil, pii
		  double * dev_planckband_int,  // pii
		  double * dev_starflux, // pil
		  double * dev_opac_interwave,  // csp
		  double * dev_opac_deltawave,  // csp, cse
		  double * dev_F_down_tot, // cse
		  double * dev_T_lay, // it, pil, io, mmm, kil
		  double * dev_T_int, // it, pii, ioi, mmmi, kii
		  double * dev_ktemp, // io, mmm, mmmi
		  double * dev_p_lay, // io, mmm, kil
		  double * dev_p_int, // ioi, mmmi, kii
		  double * dev_kpress, // io, mmm, mmmi
		  double * dev_opac_k, // io
		  double * dev_opac_wg_lay, // io
		  double * dev_opac_wg_int, // ioi
		  double * dev_opac_scat_cross, // io
		  double * dev_scat_cross_lay, // io
		  double * dev_scat_cross_int, // ioi
		  double * dev_meanmolmass_lay, // mmm
		  double * dev_meanmolmass_int, // mmmi
		  double * dev_opac_meanmass, // mmm, mmmi
		  double * dev_opac_kappa, // kil, kii
		  double * dev_entr_temp, // kil, kii
		  double * dev_entr_press, // kil, kii
		  double * dev_kappa_lay, // kil
		  double * dev_kappa_int, // kii
		  const int & ninterface, // it, pii, mmmi, kii
		  const int & nbin, // csp, cse, pil, pii, io
		  const int & nlayer, // csp, cse, pil, io, mmm, kil
		  const int & iter_value, // cse // TODO: check what this is for. Should maybe be external
		  const int & real_star, // pil
		  const int & npress, // io, mmm, mmmi
		  const int & ntemp, // io, mmm, mmmi
		  const int & ny, // io
		  const int & entr_npress, // kii, kil
		  const int & entr_ntemp, // kii, kil		  
		  const double & fake_opac, // io
		  
		  const double & T_surf, // csp, cse, pil
		  const double & surf_albedo, // cse
		  const int & dim, // pil, pii
		  const int & step, // pil, pii
		  const bool & use_kappa_manual, // ki
		  const double & kappa_manual_value, // ki	     
		  const bool & iso, // pii
		  const bool & correct_surface_emissions,
		  const bool & interp_and_calc_flux_step
		  
			  );


// calculates the integrated upwards and downwards fluxes
void integrate_flux(
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
		    );

void wrap_integrate_flux(long deltalambda_, // double*
			 long F_down_tot_, // double *
			 long F_up_tot_, // double *
			 long F_net_,  // double *
			 long F_down_wg_,  // double *
			 long F_up_wg_, // double *
			 long F_dir_wg_, // double *
			 long F_down_band_,  // double *
			 long F_up_band_,  // double *
			 long F_dir_band_, // double *
			 long gauss_weight_, // double *
			 int 	nbin, 
			 int 	numinterfaces, 
			 int 	ny,
			 int block_x,
			 int block_y,
			 int block_z,
			 int grid_x,
			 int grid_y,
			 int grid_z
			 );

bool wrap_calculate_transmission_iso(
				      long 	trans_wg,
        long 	delta_tau_wg,
        long 	M_term,
        long 	N_term,
        long 	P_term,
        long 	G_plus,
        long 	G_minus,
        long 	delta_colmass,
        long 	opac_wg_lay,
        long cloud_opac_lay,
        long 	meanmolmass_lay,
        long 	scat_cross_lay,
        long 	cloud_scat_cross_lay,
        long  w_0,
        long 	g_0_tot_lay,
        double   g_0,
        double 	epsi,
        double 	mu_star,
        int 	scat,
        int 	nbin,
        int 	ny,
        int 	nlayer,
        int 	clouds,
        int 	scat_corr
					      );

 bool wrap_calculate_transmission_noniso(
        long trans_wg_upper,
        long trans_wg_lower,
        long delta_tau_wg_upper,
        long delta_tau_wg_lower,
        long M_upper,
        long M_lower,
        long N_upper,
        long N_lower,
        long P_upper,
        long P_lower,
        long G_plus_upper,
        long G_plus_lower,
        long G_minus_upper,
        long G_minus_lower,
        long delta_col_upper,
        long delta_col_lower,
        long opac_wg_lay,
        long opac_wg_int,
        long cloud_opac_lay,
        long cloud_opac_int,		
        long meanmolmass_lay,
        long meanmolmass_int,
        long scat_cross_lay,
        long scat_cross_int,
        long cloud_scat_cross_lay,
        long cloud_scat_cross_int,		
        long w_0_upper,
        long w_0_lower,
        long 	g_0_tot_lay,
        long 	g_0_tot_int,
        double	g_0,
        double 	epsi,
        double 	mu_star,
        int 	scat,
        int 	nbin,
        int 	ny,
        int 	nlayer,
        int 	clouds,
        int 	scat_corr
					 );


bool wrap_direct_beam_flux(long 	F_dir_wg,
			   long 	Fc_dir_wg,
			   long 	planckband_lay,
			   long 	delta_tau_wg,
			   long 	delta_tau_wg_upper,
			   long 	delta_tau_wg_lower,
			   long 	z_lay,
			   double 	mu_star,
			   double	R_planet,
			   double 	R_star, 
			   double 	a,
			   int		dir_beam,
			   int		geom_zenith_corr,
			   int 	ninterface,
			   int 	nbin,
			   int 	ny,
			   bool iso
			   );




bool wrap_populate_spectral_flux_noniso(
					      long F_down_wg, 
					      long F_up_wg, 
					      long Fc_down_wg, 
					      long Fc_up_wg,
					      long F_dir_wg,
					      long Fc_dir_wg,
					      long planckband_lay, 
					      long planckband_int,
					      long w_0_upper,
					      long w_0_lower,
					      long delta_tau_wg_upper,
					      long delta_tau_wg_lower,
					      long M_upper,
					      long M_lower,
					      long N_upper,
					      long N_lower,
					      long P_upper,
					      long P_lower,
					      long G_plus_upper,
					      long G_plus_lower,
					      long G_minus_upper,
					      long G_minus_lower,
					      long g_0_tot_lay,
					      long g_0_tot_int,
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
					      long	trans_wg_upper,
					      long trans_wg_lower
					);

bool wrap_populate_spectral_flux_iso(
				     long F_down_wg, 
        long F_up_wg, 
        long F_dir_wg, 
        long planckband_lay,
        long w_0,
        long delta_tau_wg,
        long M_term,
        long N_term,
        long P_term,
        long G_plus,
        long G_minus,
        long g_0_tot_lay,
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
				     );

void init_alfrodull();
void deinit_alfrodull();
