%% Part 1:

% lambda_r0 = 40000.; % [Pa]
% lambda_i0 = 0.; % [Pa]
% mu_r0 = 1800.; % [Pa]
% mu_i0 = 180.; % [Pa]
% rho_0 = 1020.; % [kg/m3]
% 
% vec_param_initialisation = [(lambda_r0+1i*lambda_i0) (mu_r0+1i*mu_i0) rho_0];
% 
% vec_numeros_parametres_comportement = [1 2];
% vec_numeros_parametres_masse = [3];
% vec_numeros_parametres_a_identifier = [1 2];
% vec_numeros_parametres_fixes = [3];
% 
% struct_param_comportement_a_identifier.vec_numeros_parametres_comportement = vec_numeros_parametres_comportement;
% struct_param_comportement_a_identifier.vec_numeros_parametres_masse = vec_numeros_parametres_masse;
% struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier = vec_numeros_parametres_a_identifier;
% struct_param_comportement_a_identifier.vec_numeros_parametres_fixes = vec_numeros_parametres_fixes;
% struct_param_comportement_a_identifier.vec_param_initialisation = vec_param_initialisation;
% 
% clear lambda_r0 lambda_i0 mu_r0 mu_i0 rho_0 vec_numeros_parametres_comportement vec_numeros_parametres_masse vec_numeros_parametres_a_identifier vec_numeros_parametres_fixes;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% n_LdC_normalisation = 1;
% struct_param_comportement_normalisation = liste_LdC{n_LdC_normalisation};
% 
% lambda_r0 = 40000.; % [Pa]
% mu_r0 = 1800.; % [Pa]
% rho_0 = 1020.; % [kg/m3]
% 
% vec_param_initialisation = [lambda_r0 mu_r0 rho_0];
% 
% vec_numeros_parametres_comportement = [1 2];
% vec_numeros_parametres_masse = [3];
% vec_numeros_parametres_a_identifier = [1 2];
% vec_numeros_parametres_fixes =  [3];
% 
% struct_param_comportement_normalisation.vec_numeros_parametres_comportement = vec_numeros_parametres_comportement;
% struct_param_comportement_normalisation.vec_numeros_parametres_masse = vec_numeros_parametres_masse;
% struct_param_comportement_normalisation.vec_numeros_parametres_a_identifier = vec_numeros_parametres_a_identifier;
% struct_param_comportement_normalisation.vec_numeros_parametres_fixes = vec_numeros_parametres_fixes;
% struct_param_comportement_normalisation.vec_param_initialisation = vec_param_initialisation;
% 
% clear lambda_r0 mu_r0 rho_0 vec_numeros_parametres_comportement vec_numeros_parametres_masse vec_numeros_parametres_a_identifier vec_numeros_parametres_fixes;

%% Part 2: