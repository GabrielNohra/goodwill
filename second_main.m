%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% NOMENCLATURE OF THIS CODE %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% idParam: structura variable related to the studied case.
% nnParam: structure variable related to the normalized behaviour of materials.
% filterType: eventual filtrage of displacements.
% idType: type of identification method.
%
%
%
%
%
%
%
%                                                                               /

idx = {'1','7'}; % {normalized, studied case}
filter_type = 'no_filtering';
id_method = 'MECE';
list_ref_elem = {1, 4}; % {phase, stress}
int_points = {6, 6}; % {stiffness Gaussian points, mass Gaussian points}
var_bool = false;

init_struct = param_init(idx, filter_type, id_method, list_ref_elem,...
		 int_points, var_bool);















