function [structV] = param_init(list, filter_type, id_method,...
			 list_ref_elem, num_iter, var_bool)
% Description of the function param_init.m:
%
% This function returns the intialized parameters for a particular problem
% to be solved, including all the definitios related  to the subzones (SZ).
%  The input parameters of the function are the following:
% 
%	 *  list: specifies the index of the type of behaviour  which we will
%	 address).
%	 *  filter_type: specifies the type of filtering regarding the eventual
%	 displacement field.
%	 *  id_method: specifies the type of identification method addressed
%	 such as direct, adjoint, and modified-ECE.
%	 *  list_ref_elem: type of reference elements: 
%
%		- Constant. Hexagonal elements with 1 node (1).
%		- Linear. Hexagonal elements  with 8 nodes (2).
%		- Quadratic 20 nodes. Hexagonal elements with 20 nodes (3).
%		- Quadratic 27 nodes. Hexagonal elements with 27 nodes (4).
%	 
%	 *  num_iter: number of Gaussian points used for the numerical integration 
%	 of the stiffness (K) and mass matrix (M).
%	 *  var_bool: Boolean variable defining the grouping of the mass matrix
%	 (true: grouping; false: no groping).
%
% The output paramter is an array of struct elements in which the information 
% is stored. The fields of the first two elements (structV{1} and structV{2})
% of this array are:
% 
%	 *  Number of behaviour parameters, structV{number}.behav_param
%	    (numeros_parametres_comportement).
%	 *  Number of mass paramteres, structV{number}.mass_param
%	    (numeros_parametres_masse).
%	 *  Number of parameters to identify, structV{number}.id_param
%	    (numeros_parametres_a_identifier).
%	 *  Number of fixed parametes, structV{number}.fixed_param
%	    (numeros_parametres_fixes). 
%	 *  Initialized parameters, structV{number}.init. This is a 1x3 row vector
%	    which contains three parameters:  lambda_r0 + i*lambda_i0 (Lame's first
%	    parameter), mu_r0 + i*mu_i0 (Lame's second parameter), and rho_0
%	    (mass density).
%
% The other elements of the array are described below:
%
%	 *  structV{3} specifies the type of filtering (Gaussian, no_filtering),
%	 the identification method employed (MECE, FEMU_AFC, FEMU_DFC) and the
%	 mechanical pulse of the system.
%	 *  structV{4} stores the values of list_ref_elem, num_iter, and var_bool.
%	 *  structV{5} defines the FEM meshing structure for the direct method
%	 (DFC). These are defined proportionally to the size of the measurement
%	 zone. In adittion, it specifies the subzone lengths (Lx_SZ, Ly_SZ, and 
%	 Lz_SZ) for phase and stress elements, as well as the overlapping of
%	 subzones (0: no overlapping, 1: overlapping compensated by one stress
%	 element).
%	 *  structV{6} defines the tolerance value regarding the position
%	 (tol_LDC), the convergence parameters related to the material identifica-
%	 tion (tol_LDC, iter_max_LDC), the  phase threshold of elements that are not
%	 deformed enough (phase_threshold), the  ponderation parameter of 
%	 measurement/behaviour (kappa), and number of degrees of freedom per node
%	 (DOFs_per_node).
%	 *  structV{7} specifies the essential boundary conditions (BC) of the 
%	 problem (no BC with MECE and FEMU_AFC).
%	 *  structV{8} specifies the stress and phase reference elements that are
%	 associated with structV{4}.num_iter{1} and structV{4}.num_iter{2}, respec-
%	 tively.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% File directory in which images are stored

name  = '/users/bionanonmri/nohra/Documents/MATLAB/MRI/donnees_dep.don';
list = creation_LdC_anisotrope_repere_global();

%% Behaviour definitions

% Initialziation of parameters related to the material behaviour

for i=1:length(idx)

    structV{i} = list{i};
    
    switch idx{i}  % cases 1 and 7 are defined only!
        case 1 % linear isotropic elastic (normalized)
	    lambda_0 = 40000.; % [Pa]
	    mu_0 = 1800.; % [Pa]
	    rho_0 = 1020.; % [kg/m3]
            structV{i}.init = [lambda_0 mu_0 rho_0];
        case 2 % linear isotropic elastic quasi-incompressible
            lambda_0 = 40000.; % [Pa]
            mu_0 = 1800.; % [Pa]
            rho_0 = 1020.; % [kg/m3]
	    structV{i}.init = [lambda_0 mu_0 rho_0];
        case 3 % linear isotropic elastic incompressible
            mu_0 = 1800.; % [Pa]
            rho_0 = 1020.; % [kg/m3]
	    structV{i}.init = [mu_0 rho_0];
        case 4 % linear elastic cubic
            lambda_0 = 40000.; % [Pa]
            mu_0 = 1800.; % [Pa]
	    G_0 = 0.; % [Pa]
	    theta_x_0 = 0.; % [°]
	    theta_y_0 = 0.; % [°]
	    theta_z_0 = 0.; % [°]
            rho_0 = 1020.; % [kg/m3]
	    structV{i}.init = [lambda_0 mu_0 G_0 theta_x_0 theta_y_0 theta_z_0];
        case 5 % linear isotropic elastic transverse
	    C_11_0 = 0.; % [Pa]
	    C_33_0 = 0.; % [Pa]
	    C_44_0 = 0.; % [Pa]
	    C_12_0 = 0.; % [Pa]
	    C_13_0 = 0.; % [Pa]
            theta_x_0 = 0.; % [°]
            theta_y_0 = 0.; % [°]
            theta_z_0 = 0.; % [°]
	    rho_0 = 1020.; % [kg/m3]
	    structV{i}.init = [C_11_0 C_33_0 C_44_0 C_12_0 C_13_0 theta_x_0...
				 theta_y_0  theta_z_0];
        case 6 % linear elastic orthotropic
	    C_11_0 = 0.; % [Pa]
	    C_22_0 = 0.; % [Pa]
 	    C_33_0 = 0.; % [Pa]
	    C_44_0 = 0.; % [Pa]
	    C_55_0 = 0.; % [Pa]
	    C_66_0 = 0.; % [Pa]
	    C_12_0 = 0.; % [Pa]
	    C_13_0 = 0.; % [Pa]
	    C_23_0 = 0.; % [Pa]
	    theta_x_0 = 0.; % [°]
	    theta_y_0 = 0.; % [°]
	    theta_z_0 = 0.; % [°]
            rho_0 = 1020.; % [kg/m3
            structV{i}.init = [C_11_0 C_22_0 C_33_0 C_44_0 C_55_0 C_66_0...
				 C_12_0 C_13_0 C_23_0 theta_x_0 theta_y_0 theta_z_0];
        case 7 % linear isotropic viscoelastic
            lambda_r0 = 40000.; % [Pa]
	    lambda_i0 = 180.; % [Pa]
            mu_r0 = 1800.; % [Pa]
	    mu_i0 = 180.; % [Pa]
            rho_0 = 1020.; % [kg/m3]
            structV{i}.init = [(lambda_r0+1i*lambda_i0) (mu_r0+1i*mu_i0) rho_0];
        case 8 % linear isotropic viscoelastic quasi-incompressible
	    lambda_r0 = 40000.; % [Pa]
            lambda_i0 = 180.; % [Pa]
            mu_r0 = 1800.; % [Pa]
            mu_i0 = 180.; % [Pa]
            rho_0 = 1020.; % [kg/m3]
            structV{i}.init = [(lambda_r0+1i*lambda_i0) (mu_r0+1i*mu_i0) rho_0];
        case 9 % linear isotropic viscoelastic incompressible
	    mu_r0 = 1800.; % [Pa]
            mu_i0 = 180.; % [Pa]
            rho_0 = 1020.; % [kg/m3]
            structV{i}.init = [(mu_r0+1i*mu_i0) rho_0];
	case 10 % linear viscoelastic cubic
            lambda_r0 = 40000.; % [Pa]
            lambda_i0 = 180.; % [Pa]
            mu_r0 = 1800.; % [Pa]
            mu_i0 = 180.; % [Pa]
            G_r0 = 0.; % [Pa]
            G_i0 = 180.; % [Pa]
            theta_x_0 = 0.; % [°]
            theta_y_0 = 0.; % [°]
            theta_z_0 = 0.; % [°]
            rho_0 = 1020.; % [kg/m3]
	    structV{i}.init = [(lambda_r0+1i*lambda_i0) (mu_r0+1i*mu_i0)...
				 (G_r0+1i*G_i0) theta_x_0 theta_y_0 theta_z_0];
	case 11 % linear isotropic viscoelastic transverse
            C_11_r0 = 0.; % [Pa]
            C_11_i0 = 0.; % [Pa]
            C_33_r0 = 0.; % [Pa]
            C_33_i0 = 0.; % [Pa]
            C_44_r0 = 0.; % [Pa]
            C_44_i0 = 0.; % [Pa]
            C_12_r0 = 0.; % [Pa]
            C_12_i0 = 0.; % [Pa]
            C_13_r0 = 0.; % [Pa]
            C_13_i0 = 0.; % [Pa]
            theta_x_0 = 0.; % [°]
            theta_y_0 = 0.; % [°]
            theta_z_0 = 0.; % [°]
            rho_0 = 1020.; % [kg/m3]
            structV{i}.init = [(C_11_r0+1i*C_11_i0) (C_33_r0+1i*C_33_i0)...
				 (C_44_r0+1i*C_44_i0) (C_12_r0+1i*C_12_i0)...
				 (C_13_r0+1i*C_13_i0) theta_x_0 theta_y_0  theta_z_0];
	case 12 % linear viscoelastic orthotropic
            C_11_r0 = 0.; % [Pa]
            C_11_i0 = 0.; % [Pa]
            C_22_r0 = 0.; % [Pa]
            C_33_i0 = 0.; % [Pa]
            C_44_r0 = 0.; % [Pa]
            C_55_i0 = 0.; % [Pa]
            C_66_r0 = 0.; % [Pa]
            C_12_i0 = 0.; % [Pa]
            C_13_r0 = 0.; % [Pa]
            C_23_i0 = 0.; % [Pa]
            theta_x_0 = 0.; % [°]
            theta_y_0 = 0.; % [°]
            theta_z_0 = 0.; % [°]
            rho_0 = 1020.; % [kg/m3]
            structV{i}.init = [(C_11_r0+1i*C_11_i0) (C_22_r0+1i*C_22_i0)...
				 (C_33_r0+1i*C_33_i0) (C_44_r0+1i*C_44_i0)...
                                 (C_55_r0+1i*C_55_i0) (C_66_r0+1i*C_66_i0)...
				 (C_12_r0+1i*C_12_i0) (C_13_r0+1i*C_13_i0)...
				 (C_23_r0+1i*C_23_i0) theta_x_0 theta_y_0 theta_z_0];
    end

	structV{i}.behav_param = [1 2]; 
	structV{i}.mass_param = [3];
	structV{i}.id_param = [1 2];
	structV{i}.fixed_param = [3];

end

%% Other definitions

if strcmp(filter_type,'Gaussian') == 1 
    structV{3} = struct('type','Gaussian','Sigma_x',1,'R_x',3,'Sigma_y',...
        1,'R_y',3,'Sigma_z',1,'R_z',3);
else
    structV{3} = struct('type',filter_type);
end

structV{3}.('id_method') =  id_method;

freq_mec = 20; % Solicitation frequency [Hz]

structV{3}.('mech_pulse') = 2*pi*freq_mec; % Mechanical pulse [rad/s]

structV{4} = struct('ref_elem_phase', list_ref_elem{1}, 'ref_elem_sig',...
		      list_ref_elem{2}, 'num_int_K', num_iter{1}, 'num_int_M',...
			num_iter{2}, 'mass_group', var_bool);

structV{4}.('overlap') = 0.5;

structV{5} = struct('sig_x',20,'sig_y',21,'sig_z',22,'pha_x',3,'pha_y',4,...
                'pha_z',5);

[structV{5}.('Lx_SZ'), structV{5}.('Ly_SZ'), structV{5}.('Lz_SZ')]...
								 = deal(2*5+1);

structV{6} = struct('tol_pos',10000.,'tol_LDC',1e-5,'iter_max_LDC',20,...
	       'phase_threshold',0.1,'kappa',10000,'DOFs_per_node',3);

% Essential boundary conditions (BC)

if ( strcmp( structV{3}.id_method,'MECE' ) == 1 )
    structV{7} = struct('type','no_BC');
elseif ( strcmp( structV{3}.id_method,'FEMU_AFC' ) == 1 )
    structV{7} = struct('type','no_BC');
elseif ( strcmp( structV{3}.id_method,'FEMU_DFC' ) == 1 )
    %structV{7} = struct('type','measurement','type_interpolation_U',...
		% 'linear','type_extrapolation_U','linear');
    structV{7} = struct('type','filtred_measurement','filtred_parameters',...
	struct('type','Gaussian','Sigma_x',1,'R_x',3,'Sigma_y',1,...
		'R_y',3,'Sigma_z',1,'R_z',3));
end

% Creation of reference elements 

liste_elem_ref = creation_elem_ref;

% Definition of phase and stress elements

structV{8} = struct('P_el', liste_elem_ref{n_elem_pha},...
		 'S_el', liste_elem_ref{n_elem_sig}); 

end
