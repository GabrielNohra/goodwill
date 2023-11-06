function [struct] = param_init(list, id, type)
% paramInitialization: this function returns the intialized parameters for
% a particular problem to be solved. The input parameters of the function 
% are the following:
% 
%   *  id: specifies the type of behaviour which we will address.
%
%   *  list: specifies the behaviours we could address: elastic linear
%      isotropic materials, elastic linear isotropic quasi incompressible
%      materials, ...
%
%   *  type: it defines whether it is an initialization (1) or a
%      normalization (0) of parameters.
% 
%   Detailed explanation goes here

%% Definition of the behaviour we are going to address...

struct = list(id);

%% Initialization of parameters related to the behaviour

lambda_r0 = 40000.; % [Pa]
lambda_i0 = 0.; % [Pa]
mu_r0 = 1800.; % [Pa]
mu_i0 = 180.; % [Pa]
rho_0 = 1020.; % [kg/m3]

init = [(lambda_r0+1i*lambda_i0*type) (mu_r0+1i*mu_i0*type) rho_0];

vec_numeros_parametres_comportement = [1 2];
vec_numeros_parametres_masse = [3];
vec_numeros_parametres_a_identifier = [1 2];
vec_numeros_parametres_fixes = [3];

struct{1}.vec_numeros_parametres_comportement = vec_numeros_parametres_comportement;
struct{1}.vec_numeros_parametres_masse = vec_numeros_parametres_masse;
struct{1}.vec_numeros_parametres_a_identifier = vec_numeros_parametres_a_identifier;
struct{1}.vec_numeros_parametres_fixes = vec_numeros_parametres_fixes;
struct{1}.vec_param_initialisation = init;

end

