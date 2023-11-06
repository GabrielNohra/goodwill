function [ L_tot, struct_grille_mes, mat_X_mes_3D, mat_I_mes_3D,...
 mat_U_mes_3D ] = untitled5(mat_pos_mes,mat_U_mes,facteur_tolerance_position)
% Description of the function untitled5.m
%
%   Detailed explanation goes here
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

mat_data = load(nom_fichier_deplacement);
mat_pos_mes = mat_data(:,1:3)';
mat_U_mes = mat_data(:,4:2:9)'+1i*mat_data(:,5:2:9)';

nb_dim = size(mat_U_mes,1);

%% Total mesh length (1D)

L_tot = max(mat_pos_mes') - min(mat_pos_mes');
tolerance_position = max(L_tot)/facteur_tolerance_position;

%% Reconstruction of measurement mesh (regular)

vec_mes = diff(sort(mat_pos_mes,2),1,2);
vec_mes(vec_mes == 0) = NaN;
d_mes = min(vec_dx,[],2);

min_mes = min(mat_pos_mes');

vec = (round(bsxfun(@rdivide, mat_pos_mes - repmat(min(mat_pos_mes'),size(mat_pos_mes,2),1)', d_mes))+1);

struct_grille_mes = struct('dx',d_mes(1,:),'dy',d_mes(2,:),'dz',d_mes(3,:),'x_min',min_mes(:,1),...
	'y_min',min_mes(:,2),'z_min',min_mes(:,3),'facteur_tolerance_position',...
									facteur_tolerance_position);

%% Conversion of enumeration => np

n_mes = max(vec,[],2);

vec_n_mes = 1+(vec_i_mes-1)+ni_mes*((vec_j_mes-1)+nj_mes*(vec_k_mes-1));

for i=1:size(vec,1)
       
    [mat_I_tempo_3D, mat_X_tempo_3D, mat_U_tempo_3D] = deal(...
					nan(ni_mes,nj_mes,nk_mes));
 
    [mat_I_tempo_3D(vec_n_mes), mat_X_tempo_3D(vec_n_mes),...
	 mat_U_tempo_3D(vec_n_mes)] = struct('x', num2cell([vec(i,:),...
					mat_pos_mes(i,:),mat_U_mes(i,:)])).x
    
    [mat_I_mes_3D(:,:,:,i), mat_X_mes_3D(:,:,:,i),...
	 mat_U_tempo_3D(:,:,:,i)] = struct('x', num2cell([mat_I_tempo_3D,...
						mat_X_tempo_3D,...
						mat_U_tempo_3D])).x

end

R_a_conserver = round(0.8*0.5*min([ni_mes nj_mes nk_mes]));

for n_dim = 1:nb_dim
    mat_i_mes_3D_local = squeeze(mat_I_mes_3D(:,:,:,1));
    mat_j_mes_3D_local = squeeze(mat_I_mes_3D(:,:,:,2));
    mat_k_mes_3D_local = squeeze(mat_I_mes_3D(:,:,:,3));
    vec_n_a_supprimer = find ( sqrt((mat_i_mes_3D_local(:)-ni_mes/2).^2+(mat_j_mes_3D_local(:)-nj_mes/2).^2+(mat_k_mes_3D_local(:)-nk_mes/2).^2) > R_a_conserver );
    mat_X_mes_3D_local = squeeze(mat_X_mes_3D(:,:,:,n_dim));
    mat_U_mes_3D_local = squeeze(mat_U_mes_3D(:,:,:,n_dim));
    mat_X_mes_3D_local(vec_n_a_supprimer) = nan;
    mat_U_mes_3D_local(vec_n_a_supprimer) = nan;
    mat_X_mes_3D(:,:,:,n_dim) = mat_X_mes_3D_local;
    mat_U_mes_3D(:,:,:,n_dim) = mat_U_mes_3D_local;
end

clear mat_i_mes_3D_local mat_j_mes_3D_local mat_k_mes_3D_local vec_n_a_supprimer mat_X_mes_3D_local mat_U_mes_3D_local;

t_fin = cputime;
disp(['      ' num2str(t_fin-t_ini) ' s']);
disp(' ');

%% Creation de maillage

t_ini = cputime;
[liste_elem_pha,mat_pos_maillage_pha,mat_pos_pha,mat_n_pha,liste_elem_sig,mat_pos_maillage_sig,mat_pos_sig,mat_n_sig] = creation_maillages_EF(mat_X_mes_3D,elem_pha_ref,elem_sig_ref,n_integration_K,n_integration_M,struct_parametres_maillage_EF,struct_grille_mes);


toc;

end
