clearvars; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%INITIALISATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('INITIALISATION');
disp(' ');

%% *** FICHIER ENTREE ***

nom_fichier_deplacement = '/users/bionanonmri/nohra/Documents/MATLAB/MRI/donnees_dep.don';
liste_LdC = creation_LdC_anisotrope_repere_global();

%% function 1: param_init.m (avalaiable at Part 1 - untitled2.m)

struct_param_comportement_a_identifier = liste_LdC{7};
%struct_param_comportement_normalisation = liste_LdC{1};

 lambda_r0 = 40000.; % [Pa]
 lambda_i0 = 0.; % [Pa]
 mu_r0 = 1800.; % [Pa]
 mu_i0 = 180.; % [Pa]
 rho_0 = 1020.; % [kg/m3]
 
 vec_param_initialisation = [(lambda_r0+1i*lambda_i0) (mu_r0+1i*mu_i0) rho_0];
 
 vec_numeros_parametres_comportement = [1 2];
 vec_numeros_parametres_masse = [3];
 vec_numeros_parametres_a_identifier = [1 2];
 vec_numeros_parametres_fixes = [3];
 
 struct_param_comportement_a_identifier.vec_numeros_parametres_comportement = vec_numeros_parametres_comportement;
 struct_param_comportement_a_identifier.vec_numeros_parametres_masse = vec_numeros_parametres_masse;
 struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier = vec_numeros_parametres_a_identifier;
 struct_param_comportement_a_identifier.vec_numeros_parametres_fixes = vec_numeros_parametres_fixes;
 struct_param_comportement_a_identifier.vec_param_initialisation = vec_param_initialisation;
 
 clear lambda_r0 lambda_i0 mu_r0 mu_i0 rho_0 vec_numeros_parametres_comportement vec_numeros_parametres_masse vec_numeros_parametres_a_identifier vec_numeros_parametres_fixes;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 n_LdC_normalisation = 1;
 struct_param_comportement_normalisation = liste_LdC{n_LdC_normalisation};
 
 lambda_r0 = 40000.; % [Pa]
 mu_r0 = 1800.; % [Pa]
 rho_0 = 1020.; % [kg/m3]
 
 vec_param_initialisation = [lambda_r0 mu_r0 rho_0];
 
 vec_numeros_parametres_comportement = [1 2];
 vec_numeros_parametres_masse = [3];
 vec_numeros_parametres_a_identifier = [1 2];
 vec_numeros_parametres_fixes =  [3];
 
 struct_param_comportement_normalisation.vec_numeros_parametres_comportement = vec_numeros_parametres_comportement;
 struct_param_comportement_normalisation.vec_numeros_parametres_masse = vec_numeros_parametres_masse;
 struct_param_comportement_normalisation.vec_numeros_parametres_a_identifier = vec_numeros_parametres_a_identifier;
 struct_param_comportement_normalisation.vec_numeros_parametres_fixes = vec_numeros_parametres_fixes;
 struct_param_comportement_normalisation.vec_param_initialisation = vec_param_initialisation;
 
 clear lambda_r0 mu_r0 rho_0 vec_numeros_parametres_comportement vec_numeros_parametres_masse vec_numeros_parametres_a_identifier vec_numeros_parametres_fixes;

%% --- Autres definitions

struct_filtrage_deplacement = struct('type','sans'); % Filtrage eventuel des deplacements
%struct_filtrage_deplacement = struct('type','Gaussien','Sigma_x',1,'R_x',3,'Sigma_y',1,'R_y',3,'Sigma_z',1,'R_z',3);


L_x_sub_zone = (2*5+1); % en nombre d'elements de contrainte dans la direction "x"
L_y_sub_zone = (2*5+1); % en nombre d'elements de contrainte dans la direction "y"
L_z_sub_zone = (2*5+1); % en nombre d'elements de contrainte dans la direction "z"

taux_recouvrement_sub_zones_par_MAJ_materielle = 0.5; % 0 : sub-zones disjointes, 1 : sub-zones chevauchantes decalees d'un seul element de contrainte

% Type de methode d'identification:

type_identification = 'MERC';
% type_identification = 'FEMU_AFC'; % no_BC
% type_identification = 'FEMU_DFC'; % BC

f_mec = 35; % Defintion de la frequence de sollicitation [Hz]

% Definitions des tailles de sub-zones

% L_x_sub_zone = (2*3+1); % en nombre d'elements de contrainte dans la direction "x"
% L_y_sub_zone = (2*3+1); % en nombre d'elements de contrainte dans la direction "y"
% L_z_sub_zone = (2*3+1); % en nombre d'elements de contrainte dans la direction "z"

L_x_sub_zone = (2*5+1); % en nombre d'elements de contrainte dans la direction "x"
L_y_sub_zone = (2*5+1); % en nombre d'elements de contrainte dans la direction "y"
L_z_sub_zone = (2*5+1); % en nombre d'elements de contrainte dans la direction "z"

% Defintion du nombre de sub-zones a identifier pour une mise a jour materielle

%taux_recouvrement_sub_zones_par_MAJ_materielle = 0; % 0 : sub-zones disjointes, 1 : sub-zones chevauchantes decalees d'un seul element de contrainte
taux_recouvrement_sub_zones_par_MAJ_materielle = 0.5; % 0 : sub-zones disjointes, 1 : sub-zones chevauchantes decalees d'un seul element de contrainte

%% --- %C% Facteur de relaxation [0: totale, 1: nul] a imposer si _ident devient < 0 %C%

% alpha_relaxation_lambda = 1.; % pas de relaxation : on garde la valeur identifiee
% alpha_relaxation_mu_r = 1.; % relax forte mais plus efficace
% alpha_relaxation_mu_i = 1.; % relax forte mais plus efficace

%% --- Numero element de reference: constant (1) ou lineaire (2) ou quad 20 noeuds (3) ou quad 27 noeuds (4) ; retravailler les (1)

n_elem_pha = 1; % 1, 2, 3 ou 4
n_elem_sig = 4; % 1, 2, 3 ou 4
%n_elem_mes = 3;

%% --- Type d'integration pour les elements de la matrice de raideur "K" et la matrice de masse "M"

%n_integration_K = 7;
%n_integration_M = 7;

n_integration_K = 6;
n_integration_M = 6;

%% --- Booleen pour savoir si on "lump" la matrice de masse

test_lump = false;
%test_lump = true;

%% --- Parametres pour la definition du maillage "element-finis" (EF) pour le calcul direct => phases et contrainte

% % Grandeurs definies comme des proportion de la taille de la zone de mesure

% %dx_moins = 0.25;
% %dx_plus = 0.25;
% %dy_moins = 0.25;
% %dy_plus = 0.25;
% %dz_moins = 0.25;
% %dz_plus = 0.25;

% % alpha_bord = 0.05;

% % dx_moins = alpha_bord/(1-2*alpha_bord);
% % dx_plus = alpha_bord/(1-2*alpha_bord);
% % dy_moins = alpha_bord/(1-2*alpha_bord);
% % dy_plus = alpha_bord/(1-2*alpha_bord);
% % dz_moins = alpha_bord/(1-2*alpha_bord);
% % dz_plus = alpha_bord/(1-2*alpha_bord);

% % nb_elem_pha_x = 1;
% % nb_elem_pha_y = 1;
% % nb_elem_pha_z = 1;
% % nb_elem_sig_x = 10;
% % nb_elem_sig_y = 11;
% % nb_elem_sig_z = 12;

% alpha_bord = 0.;

% dx_moins = alpha_bord/(1-2*alpha_bord);
% dx_plus = alpha_bord/(1-2*alpha_bord);
% dy_moins = alpha_bord/(1-2*alpha_bord);
% dy_plus = alpha_bord/(1-2*alpha_bord);
% dz_moins = alpha_bord/(1-2*alpha_bord);
% dz_plus = alpha_bord/(1-2*alpha_bord);

%nb_elem_sig_x = 10;
%nb_elem_sig_y = 11;
%nb_elem_sig_z = 12;

nb_elem_sig_x = 20;
nb_elem_sig_y = 21;
nb_elem_sig_z = 22;

%nb_elem_pha_x = 1;
%nb_elem_pha_y = 1;
%nb_elem_pha_z = 1;

nb_elem_pha_x = 3;
nb_elem_pha_y = 4;
nb_elem_pha_z = 5;

%nb_elem_pha_x = nb_elem_sig_x;
%nb_elem_pha_y = nb_elem_sig_y;
%nb_elem_pha_z = nb_elem_sig_z;

%% --- Facteur pour calculer la tolerance sur la position : tolerance = dimension_maxi/facteur

facteur_tolerance_position = 10000.;

%% --- %C% Parametres du solveur iteratif "gmres" --> � changer selon le solveur %C%

% nb_iter_restart_gmres = 20;
% tolerance_gmres = 1e-6;
% nb_iter_gmres = 500;

%% --- %C% Methode pour la mise a jour des proprietes des phases %C%

% test_methode_MAJ_pha = 0; % 0 : resolution en moyenne par phase, 1 : moindres carres, 2: moyenne par phase + valeurs imposees a la frontiere, 3: moyenne par phase + gradients proprietes nuls a la frontiere, 4: moyenne par subdivision de phase
% %test_methode_MAJ_pha = 1;
% %test_methode_MAJ_pha = 2;
% %test_methode_MAJ_pha = 3;
% %test_methode_MAJ_pha = 4;

%% --- Parametres de convergence sur l'identification materielle

tolerance_LDC = 1e-5;
nb_iter_LDC_max = 50;
% tolerance_LDC = [1e-3 5e-4 5e-4 5e-4 3e-4 5e-4 3e-4 1e-3 1e-3 1e-3]; % adapter � la boucle des it pyramidales
% if ( strcmp(var_fixe,'mu_r') == 1 )
%  tolerance_LDC = tolerance_LDC/4; % si on identifie lb, les facteurs auront la meme influence sur le crit
% end
% nb_iter_LDC_max = 40;
% nb_iter_LDC_save = 5;

%% --- Seuil permettant de definir les elements de phase "pas assez deformes" pour pouvoir y identifier des proprietes

seuil_NRJ = 0.1; % 0 : pas de filtre

%% --- Parametre de ponderation mesure/comportement de l'ERCM

%kappa = 1;
%kappa = 7;
kappa = 10000;

%% --- Initialisation des indices de mesures de la positivite de A

% n_i_A = 0; % indice pour la positivite des matrices de cpt identifiees
% indice_A_nonpos = []; % contient les informations sur les matrices non-definies positives

%% --- Parametres fsolve (identification materielle)

% eps_norme_x = 1;
% TolX = 1.;
% TolFun = 1e-3;
% MaxIterations = 100;

%% --- Nombre de DDL par noeud

nb_DDL_par_noeud = 3;

%% *** DEFINITIONS ***

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%DEFINITIONS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% --- Definition de la pulsation

omega_mec = 2*pi*f_mec;

%% --- Definition du type de conditions aux limites a imposer

if ( strcmp(type_identification,'MERC') == 1 )
    struct_CL = struct('type','sans');
elseif ( strcmp(type_identification,'FEMU_AFC') == 1 )
    struct_CL = struct('type','sans');
elseif ( strcmp(type_identification,'FEMU_DFC') == 1 )
    %struct_CL = struct('type','mesure','type_interpolation_U','linear','type_extrapolation_U','linear');
    struct_CL = struct('type','mesure_filtree','parametres_filtrage',struct('type','Gaussien','Sigma_x',1,'R_x',3,'Sigma_y',1,'R_y',3,'Sigma_z',1,'R_z',3));
end

%% --- Definition de la structure contenant tous les parametres de definition du maillage EF

struct_parametres_maillage_EF = struct('nb_elem_pha_x',nb_elem_pha_x,'nb_elem_pha_y',nb_elem_pha_y,'nb_elem_pha_z',nb_elem_pha_z,'nb_elem_sig_x',nb_elem_sig_x,'nb_elem_sig_y',nb_elem_sig_y,'nb_elem_sig_z',nb_elem_sig_z);

clear nb_elem_pha_x nb_elem_pha_y nb_elem_pha_z nb_elem_sig_x nb_elem_sig_y nb_elem_sig_z;

%% --- Definition de la structure contenant tous les parametres de calcul pour les matrices de masse et de raideur

struct_param_masse_raideur = struct('n_integration_K',n_integration_K,'n_integration_M',n_integration_M,'test_lump',test_lump);

%% --- Defintion des elements de reference

liste_elem_ref = creation_elem_ref;

%% --- Definition des elements de reference associes aux "phases" et aux "contraintes"

elem_pha_ref = liste_elem_ref{n_elem_pha};
elem_sig_ref = liste_elem_ref{n_elem_sig};

%% --- Lecture du fichier de resultats mesures (positions + deplacements)


% ----------------------- THE ABOVE PART OF THE CODE CAN FIND IT IN param_init.m ----------------------- %


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%LECTURE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('LECTURE FICHIERS MESURES');
t_ini = cputime;

mat_data = load(nom_fichier_deplacement);
mat_pos_mes = mat_data(:,1:3)';
mat_U_mes = mat_data(:,4:2:9)'+1i*mat_data(:,5:2:9)';

clear mat_data;

nb_dim = size(mat_U_mes,1);

%% --- Longueur 1D totale du maillage

% L_tot = max(mat_pos_mes') - min(mat_pos_mes');
% tolerance_position = max(L_tot)/facteur_tolerance_position;

Lx_tot = max(mat_pos_mes(1,:))-min(mat_pos_mes(1,:));
Ly_tot = max(mat_pos_mes(2,:))-min(mat_pos_mes(2,:));
Lz_tot = max(mat_pos_mes(3,:))-min(mat_pos_mes(3,:));

L_tot = [Lx_tot,Ly_tot,Lz_tot];
tolerance_position = max(L_tot)/facteur_tolerance_position;

clear Lx_tot Ly_tot Lz_tot L_tot;

%% --- Reconstruction de la grille (reguliere) de mesure 

% [ L_tot, struct_grille_mes, mat_X_mes_3D, mat_I_mes_3D, mat_U_mes_3D ] = untitled5( mat_pos_mes, mat_U_mes, facteur_tolerance_position );

vec_dx_mes = diff(sort(mat_pos_mes(1,:)));
vec_dy_mes = diff(sort(mat_pos_mes(2,:)));
vec_dz_mes = diff(sort(mat_pos_mes(3,:)));

dx_mes = min(vec_dx_mes(vec_dx_mes > tolerance_position));
dy_mes = min(vec_dy_mes(vec_dy_mes > tolerance_position));
dz_mes = min(vec_dz_mes(vec_dz_mes > tolerance_position));

x_min_mes = min(mat_pos_mes(1,:));
y_min_mes = min(mat_pos_mes(2,:));
z_min_mes = min(mat_pos_mes(3,:));

vec_i_mes = round((mat_pos_mes(1,:)-x_min_mes)/dx_mes)+1;
vec_j_mes = round((mat_pos_mes(2,:)-y_min_mes)/dy_mes)+1;
vec_k_mes = round((mat_pos_mes(3,:)-z_min_mes)/dz_mes)+1;

struct_grille_mes = struct('dx',dx_mes,'dy',dy_mes,'dz',dz_mes,'x_min',x_min_mes,'y_min',y_min_mes,'z_min',z_min_mes,'facteur_tolerance_position',facteur_tolerance_position);

clear vec_dx_mes vec_dy_mes vec_dz_mes dx_mes dy_mes dz_mes x_min_mes y_min_mes z_min_mes;

%% --- Conversion numerotation (i,j,k) => np

ni_mes = max(vec_i_mes);
nj_mes = max(vec_j_mes);
nk_mes = max(vec_k_mes);

vec_n_mes = 1+(vec_i_mes-1)+ni_mes*((vec_j_mes-1)+nj_mes*(vec_k_mes-1));

mat_I_mes_3D = nan(ni_mes,nj_mes,nk_mes,3);
mat_X_mes_3D = nan(ni_mes,nj_mes,nk_mes,3);
mat_U_mes_3D = nan(ni_mes,nj_mes,nk_mes,3);

% Ux

mat_I_tempo_3D = nan(ni_mes,nj_mes,nk_mes);
mat_I_tempo_3D(vec_n_mes) = vec_i_mes;
mat_I_mes_3D(:,:,:,1) = mat_I_tempo_3D;

mat_X_tempo_3D = nan(ni_mes,nj_mes,nk_mes);
mat_X_tempo_3D(vec_n_mes) = mat_pos_mes(1,:);
mat_X_mes_3D(:,:,:,1) = mat_X_tempo_3D;

mat_U_tempo_3D = nan(ni_mes,nj_mes,nk_mes);
mat_U_tempo_3D(vec_n_mes) = mat_U_mes(1,:);
mat_U_mes_3D(:,:,:,1) = mat_U_tempo_3D;

clear mat_I_tempo_3D mat_X_tempo_3D mat_U_tempo_3D;

% Uy

mat_I_tempo_3D = nan(ni_mes,nj_mes,nk_mes);
mat_I_tempo_3D(vec_n_mes) = vec_j_mes;
mat_I_mes_3D(:,:,:,2) = mat_I_tempo_3D;

mat_X_tempo_3D = nan(ni_mes,nj_mes,nk_mes);
mat_X_tempo_3D(vec_n_mes) = mat_pos_mes(2,:);
mat_X_mes_3D(:,:,:,2) = mat_X_tempo_3D;

mat_U_tempo_3D = nan(ni_mes,nj_mes,nk_mes);
mat_U_tempo_3D(vec_n_mes) = mat_U_mes(2,:);
mat_U_mes_3D(:,:,:,2) = mat_U_tempo_3D;

clear mat_I_tempo_3D mat_X_tempo_3D mat_U_tempo_3D;

% Uz

mat_I_tempo_3D = nan(ni_mes,nj_mes,nk_mes);
mat_I_tempo_3D(vec_n_mes) = vec_k_mes;
mat_I_mes_3D(:,:,:,3) = mat_I_tempo_3D;

mat_X_tempo_3D = nan(ni_mes,nj_mes,nk_mes);
mat_X_tempo_3D(vec_n_mes) = mat_pos_mes(3,:);
mat_X_mes_3D(:,:,:,3) = mat_X_tempo_3D;

mat_U_tempo_3D = nan(ni_mes,nj_mes,nk_mes);
mat_U_tempo_3D(vec_n_mes) = mat_U_mes(3,:);
mat_U_mes_3D(:,:,:,3) = mat_U_tempo_3D;

clear mat_I_tempo_3D mat_X_tempo_3D mat_U_tempo_3D;

% [ L_tot, struct_grille_mes, mat_X_mes_3D, mat_I_mes_3D, mat_U_mes_3D ] = untitled5( mat_pos_mes, mat_U_mes, facteur_tolerance_position );

%% --- On supprime un certain nombre de mesures pour simuler une acquisition incomplete ...

% ihg_a_supprimer = 1;
% ibd_a_supprimer = 9;
% jhg_a_supprimer = 1;
% jbd_a_supprimer = 8;
% khg_a_supprimer = 1;
% kbd_a_supprimer = 10;

% for n_dim = 1:nb_dim
%  mat_X_mes_3D(ihg_a_supprimer:ibd_a_supprimer,jhg_a_supprimer:jbd_a_supprimer,khg_a_supprimer:kbd_a_supprimer,n_dim) = nan;
%  mat_U_mes_3D(ihg_a_supprimer:ibd_a_supprimer,jhg_a_supprimer:jbd_a_supprimer,khg_a_supprimer:kbd_a_supprimer,n_dim) = nan;
% end

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

%% --- Affichage de la grille de mesure

% figure;
% hp = plot3(reshape(mat_X_mes_3D(:,:,:,1),1,[]),reshape(mat_X_mes_3D(:,:,:,2),1,[]),reshape(mat_X_mes_3D(:,:,:,3),1,[]),'xk');
% 
% grid;
% xlabel('x (m)');
% ylabel('y (m)');
% zlabel('z (m)');
% title('grille de mesure');

%% --- Creation maillages "element-finis": phases et contraintes

disp('CREATION MAILLAGES "ELEMENTS-FINIS"');

t_ini = cputime;
[liste_elem_pha,mat_pos_maillage_pha,mat_pos_pha,mat_n_pha,liste_elem_sig,mat_pos_maillage_sig,mat_pos_sig,mat_n_sig] = creation_maillages_EF(mat_X_mes_3D,elem_pha_ref,elem_sig_ref,n_integration_K,n_integration_M,struct_parametres_maillage_EF,struct_grille_mes);

%% --- Determination des nombre d'elements de phase dans les directions "x", "y" et "z"

% ni_elem_pha = 0;
% nj_elem_pha = 0;
% nk_elem_pha = 0;

% for n_elem_pha = 1:length(liste_elem_pha)
%  ni_elem_pha = max([ni_elem_pha liste_elem_pha{n_elem_pha}.vec_ijk(1)]);
%  nj_elem_pha = max([nj_elem_pha liste_elem_pha{n_elem_pha}.vec_ijk(2)]);
%  nk_elem_pha = max([nk_elem_pha liste_elem_pha{n_elem_pha}.vec_ijk(3)]);
% end

nb_elem_pha = length(liste_elem_pha);
ni_elem_pha = struct_parametres_maillage_EF.nb_elem_pha_x;
nj_elem_pha = struct_parametres_maillage_EF.nb_elem_pha_y;
nk_elem_pha = struct_parametres_maillage_EF.nb_elem_pha_z;

%% --- Affichage des elements de phases

% mat_coul = jet(nb_elem_pha);
% figure;
% hold on;
% n_elem_pha = 0;
% 
% for i = 1:ni_elem_pha
%     for j = 1:nj_elem_pha
%         for k = 1:nk_elem_pha
%             n_elem_pha = n_elem_pha+1;
%             if ( strcmp(liste_elem_pha{n_elem_pha}.type_elem,'HEX1') == 1 )
%                 hp = plot3(mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds_maillage,1),mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds_maillage,2),mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds_maillage,3),'xk');
%             else
%                 hp = plot3(mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds,1),mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds,2),mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds,3),'xk');
%             end
%             set(hp,'Color',mat_coul(n_elem_pha,:));
%         end
%     end
% end

% grid;
% xlabel('x (m)');
% ylabel('y (m)');
% zlabel('z (m)');
% title('noeuds du maillage de phase');

%% --- Suppression des noeuds inutilises dans le maillage de phase

vec_test_pos_pha = false(1,size(mat_pos_pha,1));

for nn_elem_pha = 1:nb_elem_pha
    vec_test_pos_pha(liste_elem_pha{nn_elem_pha}.vec_n_noeuds) = true;
end

nb_noeuds_pha = length(find(vec_test_pos_pha));
vec_correspondance_n_DDL_pha_n_noeud_pha = find(vec_test_pos_pha);
vec_correspondance_n_noeud_pha_n_DLL_pha = zeros(1,length(vec_test_pos_pha));
vec_correspondance_n_noeud_pha_n_DLL_pha(vec_correspondance_n_DDL_pha_n_noeud_pha) = 1:nb_noeuds_pha;

%% --- Modification du maillage de phase (suppression noeuds inutilises)

mat_pos_pha = mat_pos_pha(vec_correspondance_n_DDL_pha_n_noeud_pha,:);

for n_pha = 1:nb_elem_pha
    liste_elem_pha{n_pha}.vec_n_noeuds = vec_correspondance_n_noeud_pha_n_DLL_pha(liste_elem_pha{n_pha}.vec_n_noeuds);
end

%% --- Determination des nombre d'elements de contrainte dans les directions "x", "y" et "z"

% ni_elem_sig = 0;
% nj_elem_sig = 0;
% nk_elem_sig = 0;

% for n_elem_sig = 1:length(liste_elem_sig)
%  ni_elem_sig = max([ni_elem_sig liste_elem_sig{n_elem_sig}.vec_ijk(1)]);
%  nj_elem_sig = max([nj_elem_sig liste_elem_sig{n_elem_sig}.vec_ijk(2)]);
%  nk_elem_sig = max([nk_elem_sig liste_elem_sig{n_elem_sig}.vec_ijk(3)]);
% end

nb_elem_sig = length(liste_elem_sig);
ni_elem_sig = struct_parametres_maillage_EF.nb_elem_sig_x;
nj_elem_sig = struct_parametres_maillage_EF.nb_elem_sig_y;
nk_elem_sig = struct_parametres_maillage_EF.nb_elem_sig_z;

%% --- Suppression des noeuds inutilises dans le maillage de contrainte

vec_test_pos_sig = false(1,size(mat_pos_sig,1));

for nn_elem_sig = 1:nb_elem_sig
    vec_test_pos_sig(liste_elem_sig{nn_elem_sig}.vec_n_noeuds) = true;
end

nb_noeuds_sig = length(find(vec_test_pos_sig));
vec_correspondance_n_DDL_sig_n_noeud_sig = find(vec_test_pos_sig);
vec_correspondance_n_noeud_sig_n_DLL_sig = zeros(1,length(vec_test_pos_sig));
vec_correspondance_n_noeud_sig_n_DLL_sig(vec_correspondance_n_DDL_sig_n_noeud_sig) = 1:nb_noeuds_sig;

%% --- Modification du maillage de contraintes (suppression noeuds inutilises)

mat_pos_maillage_sig = mat_pos_maillage_sig(vec_correspondance_n_DDL_sig_n_noeud_sig,:);

for n_sig = 1:nb_elem_sig
    liste_elem_sig{n_sig}.vec_n_noeuds = vec_correspondance_n_noeud_sig_n_DLL_sig(liste_elem_sig{n_sig}.vec_n_noeuds);
end

%% --- Affichage des elements de contrainte

% mat_coul = jet(nb_elem_sig);
% figure;
%hold on;
% n_elem_sig = 0;

% for i = 1:ni_elem_sig
%  for j = 1:nj_elem_sig
%   for k = 1:nk_elem_sig
%    n_elem_sig = n_elem_sig+1;
%    if ( strcmp(liste_elem_sig{n_elem_sig}.type_elem,'HEX1') == 1 )
%     hp = plot3(mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds_maillage,1),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds_maillage,2),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds_maillage,3),'xk');
%    else
%     hp = plot3(mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,1),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,2),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,3),'xk');
%    end
%    set(hp,'Color',mat_coul(n_elem_sig,:));
%   end
%  end
% end

% grid;
% xlabel('x (m)');
%ylabel('y (m)');
%zlabel('z (m)');
%title('noeuds du maillage de contrainte');

% mat_coul = jet(nb_elem_sig);
% figure;
% hold on;
% 
% for n_elem_sig = 1:nb_elem_sig
%     if ( strcmp(liste_elem_sig{n_elem_sig}.type_elem,'HEX1') == 1 )
%         hp = plot3(mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds_maillage,1),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds_maillage,2),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds_maillage,3),'xk');
%     else
%         hp = plot3(mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,1),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,2),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,3),'xk');
%     end
%     set(hp,'Color',mat_coul(n_elem_sig,:));
% end
% 
% grid;
% xlabel('x (m)');
% ylabel('y (m)');
% zlabel('z (m)');
% title('noeuds du maillage de contrainte');

t_fin = cputime;
disp(['      ' num2str(t_fin-t_ini) ' s']);
disp(' ');

%% --- Determination des elements de phase qui ne contiennent pas de points de mesure "interieurs" // skip it

mat_test_mes_dans_pha = false(ni_elem_pha,nj_elem_pha,nk_elem_pha);

for nn_elem_pha = 1:nb_elem_pha
    elem_pha = liste_elem_pha{nn_elem_pha};
    if ( strcmp(elem_pha.type_elem,'HEX1') == 1 )
        test_mes_dans_pha = (sum(((mat_pos_mes(1,:) >= (min(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,1))+tolerance_position)) & (mat_pos_mes(1,:) <= (max(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,1))-tolerance_position)) & (mat_pos_mes(2,:) >= (min(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,2))+tolerance_position)) & (mat_pos_mes(2,:) <= (max(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,2))-tolerance_position)) & (mat_pos_mes(3,:) >= (min(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,3))+tolerance_position)) & (mat_pos_mes(3,:) <= (max(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,3))-tolerance_position)))) > 0);
    else
        test_mes_dans_pha = (sum(((mat_pos_mes(1,:) >= (min(mat_pos_pha(elem_pha.vec_n_noeuds,1))+tolerance_position)) & (mat_pos_mes(1,:) <= (max(mat_pos_pha(elem_pha.vec_n_noeuds,1))-tolerance_position)) & (mat_pos_mes(2,:) >= (min(mat_pos_pha(elem_pha.vec_n_noeuds,2))+tolerance_position)) & (mat_pos_mes(2,:) <= (max(mat_pos_pha(elem_pha.vec_n_noeuds,2))-tolerance_position)) & (mat_pos_mes(3,:) >= (min(mat_pos_pha(elem_pha.vec_n_noeuds,3))+tolerance_position)) & (mat_pos_mes(3,:) <= (max(mat_pos_pha(elem_pha.vec_n_noeuds,3))-tolerance_position)))) > 0);
    end
    if ( test_mes_dans_pha )
        mat_test_mes_dans_pha(elem_pha.vec_ijk(1),elem_pha.vec_ijk(2),elem_pha.vec_ijk(3)) = true;
    end
end

vec_test_noeud_pha = false(1,size(mat_pos_pha,1));
vec_test_noeud_maillage_pha = false(1,size(mat_pos_maillage_pha,1));

for nn_elem_pha = 1:length(liste_elem_pha)
    elem_pha = liste_elem_pha{nn_elem_pha};
    if ( mat_test_mes_dans_pha(elem_pha.vec_ijk(1),elem_pha.vec_ijk(2),elem_pha.vec_ijk(3)) )
        vec_test_noeud_pha(elem_pha.vec_n_noeuds) = true;
        vec_test_noeud_maillage_pha(elem_pha.vec_n_noeuds_maillage) = true;
    end
end

vec_n_noeuds_pha_a_conserver = find ( vec_test_noeud_pha );
vec_n_noeuds_pha_a_supprimer = find ( ~vec_test_noeud_pha );
liste_legende = {};

if ( ~isempty(vec_n_noeuds_pha_a_supprimer) )
    liste_legende{length(liste_legende)+1} = 'suppr';
end

if ( ~isempty(vec_n_noeuds_pha_a_conserver) )
    liste_legende{length(liste_legende)+1} = 'cons';
end

% figure;
% hold on;
% plot3(mat_pos_pha(vec_n_noeuds_pha_a_supprimer,1),mat_pos_pha(vec_n_noeuds_pha_a_supprimer,2),mat_pos_pha(vec_n_noeuds_pha_a_supprimer,3),'or');
% plot3(mat_pos_pha(vec_n_noeuds_pha_a_conserver,1),mat_pos_pha(vec_n_noeuds_pha_a_conserver,2),mat_pos_pha(vec_n_noeuds_pha_a_conserver,3),'xk');
% hold off;
% grid;
% 
% xlabel('x (m)');
% ylabel('y (m)');
% zlabel('z (m)');
% legend(liste_legende);
% title('noeuds/elements de phase');

clear liste_legende;

vec_n_noeuds_maillage_pha_a_conserver = find ( vec_test_noeud_maillage_pha );
vec_n_noeuds_maillage_pha_a_supprimer = find ( ~vec_test_noeud_maillage_pha );
liste_legende = {};

if ( ~isempty(vec_n_noeuds_maillage_pha_a_supprimer) )
    liste_legende{length(liste_legende)+1} = 'suppr';
end

if ( ~isempty(vec_n_noeuds_maillage_pha_a_conserver) )
    liste_legende{length(liste_legende)+1} = 'cons';
end

% figure;
% hold on;
% plot3(mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_supprimer,1),mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_supprimer,2),mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_supprimer,3),'or');
% plot3(mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_conserver,1),mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_conserver,2),mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_conserver,3),'xk');
% hold off;
% 
% grid;
% xlabel('x (m)');
% ylabel('y (m)');
% zlabel('z (m)');
% legend(liste_legende);title('noeuds maillage de phase');
% clear liste_legende;

%% --- Determination des matrices donnant la correspondance (i_sig,j_sig,k_sig) => n_elem_sig

mat_correspondance_i_sig_j_sig_k_sig_n_elem_sig = nan(ni_elem_sig,nj_elem_sig,nk_elem_sig);

for n_elem_sig = 1:nb_elem_sig
    elem_sig = liste_elem_sig{n_elem_sig};
    mat_correspondance_i_sig_j_sig_k_sig_n_elem_sig(elem_sig.vec_ijk(1),elem_sig.vec_ijk(2),elem_sig.vec_ijk(3)) = n_elem_sig;
end

%% --- Determination des multiplicites des points de mesure dans le maillage des contraintes

mat_multiplicite_mes_maillage_sig = zeros(ni_mes,nj_mes,nk_mes);

for n_elem_sig = 1:nb_elem_sig
    elem_sig = liste_elem_sig{n_elem_sig};
    vec_i_mes_local = elem_sig.vec_i_mes;
    vec_j_mes_local = elem_sig.vec_j_mes;
    vec_k_mes_local = elem_sig.vec_k_mes;
    vec_n_mes_local = 1+(vec_i_mes_local-1)+ni_mes*((vec_j_mes_local-1)+(vec_k_mes_local-1)*nj_mes);
    mat_multiplicite_mes_maillage_sig(vec_n_mes_local) = mat_multiplicite_mes_maillage_sig(vec_n_mes_local)+1;
end

for n_elem_sig = 1:nb_elem_sig
    elem_sig = liste_elem_sig{n_elem_sig};
    vec_i_mes_local = elem_sig.vec_i_mes;
    vec_j_mes_local = elem_sig.vec_j_mes;
    vec_k_mes_local = elem_sig.vec_k_mes;
    vec_n_mes_local = 1+(vec_i_mes_local-1)+ni_mes*((vec_j_mes_local-1)+(vec_k_mes_local-1)*nj_mes);
    vec_multiplicite_mes = mat_multiplicite_mes_maillage_sig(vec_n_mes_local);
    elem_sig.vec_multiplicite_mes = vec_multiplicite_mes;
    liste_elem_sig{n_elem_sig} = elem_sig;
end

clear mat_multiplicite_mes_maillage_sig vec_multiplicite_mes;

%% --- Filtrage des mesures

disp('FILTRAGE DES MESURES');
t_ini = cputime;

%% --- Filtrage des donnees

if ( strcmp(struct_filtrage_deplacement.type,'sans') == 1 )
else
    mat_x_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,1)),size(mat_X_mes_3D,1),size(mat_X_mes_3D,2),size(mat_X_mes_3D,3));
    mat_y_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,2)),size(mat_X_mes_3D,1),size(mat_X_mes_3D,2),size(mat_X_mes_3D,3));
    mat_z_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,3)),size(mat_X_mes_3D,1),size(mat_X_mes_3D,2),size(mat_X_mes_3D,3));
    [ii_mes_no_nan] = find ( ~isnan(mat_x_mes(:)) & ~isnan(mat_y_mes(:)) & ~isnan(mat_z_mes(:)) );
    mat_pos_mes_filtr = [mat_x_mes(ii_mes_no_nan)';mat_y_mes(ii_mes_no_nan)';mat_z_mes(ii_mes_no_nan)'];
    [mat_U_filtr] = filtrage_deplacements_IRM(mat_X_mes_3D,mat_U_mes_3D,mat_pos_mes_filtr,struct_grille_mes,struct_filtrage_deplacement);
    for n_dim = 1:size(mat_U_filtr,2)
        mat_U_filtr_3D = nan(size(mat_X_mes_3D,1),size(mat_X_mes_3D,2),size(mat_X_mes_3D,3));
        mat_U_filtr_3D(ii_mes_no_nan) = mat_U_filtr(:,n_dim);
        mat_U_mes_3D(:,:,:,n_dim) = mat_U_filtr_3D;
    end
    %% --- %ATT% On met les grilles de position et de deplacement en accord sur les "nan" ... => A AMELIORER %ATT%
    mat_U_mes_3D(isnan(mat_X_mes_3D(:))) = nan;
    mat_X_mes_3D(isnan(mat_U_mes_3D(:))) = nan;
    clear mat_x_mes mat_y_mes mat_z_mes ii_mes_no_nan mat_U_filtr mat_U_filtr_3D;
end

t_fin = cputime;
disp(['      ' num2str(t_fin-t_ini) ' s']);
disp(' ');

%% --- Determination des deplacements a appliquer en conditions aux limites

disp('DETERMINATION DES DEPLACMENTS A APPLIQUER EN CONDITIONS AUX LIMITES');
t_ini = cputime;

if ( strcmp(struct_CL.type,'sans') == 1 )
elseif ( strcmp(struct_CL.type,'mesure') == 1 )
    
    vec_x_sig = mat_pos_maillage_sig(:,1);
    vec_y_sig = mat_pos_maillage_sig(:,2);
    vec_z_sig = mat_pos_maillage_sig(:,3);
    
    vec_x_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,1)),1,[]);
    vec_y_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,2)),1,[]);
    vec_z_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,3)),1,[]);
    
    vec_ux_mes = reshape(squeeze(mat_U_mes_3D(:,:,:,1)),1,[]);
    vec_uy_mes = reshape(squeeze(mat_U_mes_3D(:,:,:,2)),1,[]);
    vec_uz_mes = reshape(squeeze(mat_U_mes_3D(:,:,:,3)),1,[]);
    
    vec_n_a_conserver = find ( ~isnan(vec_x_mes) & ~isnan(vec_y_mes) & ~isnan(vec_z_mes) );
    vec_x_mes = vec_x_mes(vec_n_a_conserver)';
    vec_y_mes = vec_y_mes(vec_n_a_conserver)';
    vec_z_mes = vec_z_mes(vec_n_a_conserver)';
    
    vec_ux_mes = vec_ux_mes(vec_n_a_conserver)';
    vec_uy_mes = vec_uy_mes(vec_n_a_conserver)';
    vec_uz_mes = vec_uz_mes(vec_n_a_conserver)';
    
    F_Ux = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_ux_mes,struct_CL.type_interpolation_U,struct_CL.type_extrapolation_U);
    F_Uy = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_uy_mes,struct_CL.type_interpolation_U,struct_CL.type_extrapolation_U);
    F_Uz = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_uz_mes,struct_CL.type_interpolation_U,struct_CL.type_extrapolation_U);
    
    vec_ux_sig = F_Ux(vec_x_sig,vec_z_sig,vec_y_sig);
    vec_uy_sig = F_Uy(vec_x_sig,vec_z_sig,vec_y_sig);
    vec_uz_sig = F_Uz(vec_x_sig,vec_z_sig,vec_y_sig);
    
    mat_U_sig_3D = [vec_ux_sig.';vec_uy_sig.';vec_uz_sig.'].';
    
    clear vec_x_mes vec_y_mes vec_z_mes vec_ux_mes vec_uy_mes vec_uz_mes vec_n_a_conserver vec_x_sig vec_y_sig vec_z_sig vec_ux_sig vec_uy_sig vec_uz_sig F_Ux F_Uy F_Uz;

elseif ( strcmp(struct_CL.type,'mesure_filtree') == 1 )
    %% --- %A FAIRE% : FILTRER LES CHAMPS MESURES FILTRES ET LES EVALUER SUR LA GRILLE DES CONTRAINTES %A FAIRE%
    disp('A FAIRE : FILTRER LES CHAMPS MESURES FILTRES ET LES EVALUER SUR LA GRILLE DES CONTRAINTES');
    [mat_U_sig_3D] = filtrage_deplacements_IRM(mat_X_mes_3D,mat_U_mes_3D,mat_pos_maillage_sig,struct_grille_mes,struct_CL.parametres_filtrage);
end

t_fin = cputime;
disp(['      ' num2str(t_fin-t_ini) ' s']);
disp(' ');

%% *** INITIALISATION PROPRIETES ***

disp('INITIALISATION PROPRIETES');

struct_param_comportement_a_identifier.mat_param = struct_param_comportement_a_identifier.vec_param_initialisation.'*ones(1,nb_noeuds_pha);
struct_param_comportement_normalisation.mat_param = struct_param_comportement_normalisation.vec_param_initialisation.'*ones(1,nb_noeuds_pha);
disp(' ');

%% --- %C% Operateur de projection mesures %C%

% disp('OPERATEUR DE PROJECTION');
% t_ini = cputime;
% [N,D] = projection_mes(mat_pos_mes,mat_pos_maillage_sig,liste_elem_sig,mat_n_sig,liste_elem_ref,nb_DDL_par_noeud);
% t_fin = cputime;
% disp(['      ' num2str(t_fin-t_ini) ' s']);
% disp(' ');

%% *** DEBUT DU TEST SUR LA CONVERGENCE SUR LES PROPRIETES MATERIELLES ***

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% DEBUT DU TEST SUR LA CONVERGENCE %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% SUR LES PROPRIETES MATERIELLES %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('BOUCLE D''IDENTIFICATION');

test_convergence_LDC = false;
n_iter_LDC = 0;
t_ini_identification = cputime;

liste_proprietes_iterations = cell(1,nb_iter_LDC_max);
liste_proprietes_iterations{n_iter_LDC+1} = struct_param_comportement_normalisation.mat_param;

nb_sub_zones_x = taux_recouvrement_sub_zones_par_MAJ_materielle*(ni_elem_sig-(floor(ni_elem_sig/L_x_sub_zone)+1))+(floor(ni_elem_sig/L_x_sub_zone)+1);
nb_sub_zones_y = taux_recouvrement_sub_zones_par_MAJ_materielle*(nj_elem_sig-(floor(nj_elem_sig/L_y_sub_zone)+1))+(floor(nj_elem_sig/L_y_sub_zone)+1);
nb_sub_zones_z = taux_recouvrement_sub_zones_par_MAJ_materielle*(nk_elem_sig-(floor(nk_elem_sig/L_z_sub_zone)+1))+(floor(nk_elem_sig/L_z_sub_zone)+1);

nb_sub_zones_max = ceil(nb_sub_zones_x*nb_sub_zones_y*nb_sub_zones_z);
nb_faces_elem_ref = size(elem_sig_ref.n_noeuds_faces,1);

while ( (~test_convergence_LDC) && ( n_iter_LDC < nb_iter_LDC_max) ) % Debut du critere sur la convergence (utile pour id)
    
    n_iter_LDC = n_iter_LDC+1; % On compte le nombre d'iterations utiles pour arriver a convergence pour un maillage donne
    disp(['  iteration ' num2str(n_iter_LDC)]);
    
    %% --- Boucles sur les sub-zones
    
    n_sub_zone = 0;
    test_fin_sub_zones = false;
    vec_test_n_elem_sig_centre_sub_zone = false(1,nb_elem_sig);
   
    champ_proprietes = struct('lambda',liste_proprietes_iterations{n_iter_LDC}(1,:),'mu',liste_proprietes_iterations{n_iter_LDC}(2,:),'rho',liste_proprietes_iterations{n_iter_LDC}(3,:));
    champ_proprietes_elastiques = struct('lambda', champ_proprietes.lambda, 'mu', champ_proprietes.mu); 

    %% WHILE DE LA SUBZONA (termina en linea 2164)
    
    vec_nn_sig_local = 1:length(vec_test_n_elem_sig_centre_sub_zone);
    
     while ( (n_sub_zone < nb_sub_zones_max) && ~test_fin_sub_zones ) 
        
        %% --- Determination de l'element de contrainte correspondant au centre de la sub-zone
        n_sub_zone = n_sub_zone+1;
        
        disp(['     traitement de la sub-zone numero ' num2str(n_sub_zone)]);
        disp('        recherche des elements de contraintes de la sub-zone');
        
        t_ini = cputime;
        
        n_sig_sub_zone = vec_nn_sig_local(randi(length(vec_nn_sig_local)));
        
       % vec_nn_sig_local = find ( ~vec_test_n_elem_sig_centre_sub_zone );
        
% the center of the subzone is the stress element in the middle of vector "vec_nn_sig_local"

       % n_sig_sub_zone = vec_nn_sig_local(round(length(vec_nn_sig_local)/2));
       % n_sig_sub_zone = vec_nn_sig_local(floor(length(vec_nn_sig_local)/2)+1);
          
        elem_sig_centre_sub_zone = liste_elem_sig{n_sig_sub_zone};
        
        i_sig_centre_sub_zone = elem_sig_centre_sub_zone.vec_ijk(1);
        j_sig_centre_sub_zone = elem_sig_centre_sub_zone.vec_ijk(2);
        k_sig_centre_sub_zone = elem_sig_centre_sub_zone.vec_ijk(3);
        
        %% --- Determination de tous les elements de contrainte de la sub-zone
        
        vec_n_elem_sig_sub_zone = nan(1,nb_sub_zones_max);
        n_elem_sig_sub_zone = 0;
        nb_points_mesure_sub_zone = 0;
        vec_test_noeuds_sig_sub_zone = false(1,size(mat_pos_maillage_sig,1));
        
        for i = i_sig_centre_sub_zone-floor(L_x_sub_zone/2)+(0:(L_x_sub_zone-1));
            for j = j_sig_centre_sub_zone-floor(L_y_sub_zone/2)+(0:(L_y_sub_zone-1));
                for k = k_sig_centre_sub_zone-floor(L_z_sub_zone/2)+(0:(L_z_sub_zone-1));
                    if ( (i >=1) && (i <= ni_elem_sig) && (j >=1) && (j <= nj_elem_sig) && (k >=1) && (k <= nk_elem_sig) )
                        n_elem_sig_local = mat_correspondance_i_sig_j_sig_k_sig_n_elem_sig(i,j,k);
                        if ( ~isnan(n_elem_sig_local) )
                            n_elem_sig_sub_zone = n_elem_sig_sub_zone+1;
                            vec_n_elem_sig_sub_zone(n_elem_sig_sub_zone) = n_elem_sig_local;
                            nb_points_mesure_sub_zone = nb_points_mesure_sub_zone+length(liste_elem_sig{n_elem_sig_local}.vec_i_mes);
                            vec_test_noeuds_sig_sub_zone(liste_elem_sig{n_elem_sig_local}.vec_n_noeuds) = true;
                        end
                    end
                end
            end
        end
        
        vec_n_elem_sig_sub_zone = vec_n_elem_sig_sub_zone(~isnan(vec_n_elem_sig_sub_zone));
        liste_elem_sig_sub_zone = cell(1,length(vec_n_elem_sig_sub_zone));
        
        for nn_elem_sig = 1:length(vec_n_elem_sig_sub_zone)
            liste_elem_sig_sub_zone{nn_elem_sig} = liste_elem_sig{vec_n_elem_sig_sub_zone(nn_elem_sig)};
        end
        
        t_fin = cputime;
        disp(['        ' num2str(t_fin-t_ini) ' s']);
        
        %% --- Determinations des elements de contrainte de la sub-zone
        
        disp('        determinations des elements de contrainte de la sub-zone');
        t_ini = cputime;
        
        %% --- Determination des noeuds situes dans la sub-zone
        
        vec_test_noeuds_sig_sub_zone = false(1,size(mat_pos_maillage_sig,1));
        
        vec_x_noeuds_mes_sub_zone = nan(1,ni_mes*nj_mes*nk_mes);
        vec_y_noeuds_mes_sub_zone = nan(1,ni_mes*nj_mes*nk_mes);
        vec_z_noeuds_mes_sub_zone = nan(1,ni_mes*nj_mes*nk_mes);
        
        mat_x_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,1)),size(mat_X_mes_3D,1),size(mat_X_mes_3D,2),size(mat_X_mes_3D,3));
        mat_y_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,2)),size(mat_X_mes_3D,1),size(mat_X_mes_3D,2),size(mat_X_mes_3D,3));
        mat_z_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,3)),size(mat_X_mes_3D,1),size(mat_X_mes_3D,2),size(mat_X_mes_3D,3));
        
        for nn_elem_sig = 1:length(liste_elem_sig_sub_zone)
            
            elem_sig = liste_elem_sig_sub_zone{nn_elem_sig};
            
            vec_n_noeuds_sig = elem_sig.vec_n_noeuds;
            vec_test_noeuds_sig_sub_zone(vec_n_noeuds_sig) = true;
            
            vec_i_noeuds_mes_sub_zone = elem_sig.vec_i_mes;
            vec_j_noeuds_mes_sub_zone = elem_sig.vec_j_mes;
            vec_k_noeuds_mes_sub_zone = elem_sig.vec_k_mes;
            
            vec_n_noeuds_mes_sub_zone = vec_i_noeuds_mes_sub_zone+ni_mes*((vec_j_noeuds_mes_sub_zone-1)+nj_mes*(vec_k_noeuds_mes_sub_zone-1));
            
            vec_x_noeuds_mes_sub_zone(vec_n_noeuds_mes_sub_zone) = mat_x_mes(vec_n_noeuds_mes_sub_zone);
            vec_y_noeuds_mes_sub_zone(vec_n_noeuds_mes_sub_zone) = mat_y_mes(vec_n_noeuds_mes_sub_zone);
            vec_z_noeuds_mes_sub_zone(vec_n_noeuds_mes_sub_zone) = mat_z_mes(vec_n_noeuds_mes_sub_zone);
            
        end
        
        vec_n_noeuds_sig_sub_zone = find(vec_test_noeuds_sig_sub_zone);
        vec_n_noeuds_mes_sub_zone = find(~isnan(vec_x_noeuds_mes_sub_zone));
        
        vec_x_noeuds_mes_sub_zone = vec_x_noeuds_mes_sub_zone(vec_n_noeuds_mes_sub_zone);
        vec_y_noeuds_mes_sub_zone = vec_y_noeuds_mes_sub_zone(vec_n_noeuds_mes_sub_zone);
        vec_z_noeuds_mes_sub_zone = vec_z_noeuds_mes_sub_zone(vec_n_noeuds_mes_sub_zone);
        
        clear vec_test_noeuds_sig_sub_zone vec_i_noeuds_mes_sub_zone vec_j_noeuds_mes_sub_zone vec_k_noeuds_mes_sub_zone;
        
        %% --- Determination des correspondances de numerotation "globale" - "locale"
        
        vec_correspondance_n_noeud_sig_global_n_noeud_sig_local = zeros(1,size(mat_pos_maillage_sig,1));
        vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(vec_n_noeuds_sig_sub_zone) = 1:length(vec_n_noeuds_sig_sub_zone);
        
        %% --- Determination des correspondances de numerotation "locale" - "globale"
        
        vec_correspondance_n_noeud_sig_local_n_noeud_sig_global = vec_n_noeuds_sig_sub_zone;
        t_fin = cputime;
        disp(['        ' num2str(t_fin-t_ini) ' s']);
        
        disp('        determinations des elements de phase de la sub-zone');
        t_ini = cputime;
        
        %% --- Determination des noeuds sz phase situes dans la sub-zone
        
        vec_test_noeuds_pha_sub_zone_G_K = false(1,size(mat_pos_pha,1));
        vec_test_noeuds_pha_sub_zone_G_M = false(1,size(mat_pos_pha,1));
        
        for nn_elem_sig = 1:length(liste_elem_sig_sub_zone)
            
            elem_sig = liste_elem_sig_sub_zone{nn_elem_sig};
            
            vec_n_noeuds_pha_G_M = elem_sig.vec_n_pha_G_M;
            vec_n_noeuds_pha_G_K = elem_sig.vec_n_pha_G_K;
            
            vec_test_noeuds_pha_sub_zone_G_K(vec_n_noeuds_pha_G_K) = true;
            vec_test_noeuds_pha_sub_zone_G_M(vec_n_noeuds_pha_G_M) = true;
            
        end
        
        vec_n_noeuds_pha_sub_zone = find(vec_test_noeuds_pha_sub_zone_G_K | vec_test_noeuds_pha_sub_zone_G_M);
        
        clear vec_test_noeuds_pha_sub_zone_G_K vec_test_noeuds_pha_sub_zone_G_M;
        
        %% --- Determination des correspondances de numerotation "GLOBALE" - "LOCALE"
        
        vec_correspondance_n_noeud_pha_global_n_noeud_pha_local = zeros(1,size(mat_pos_pha,1));
        vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha_sub_zone) = 1:length(vec_n_noeuds_pha_sub_zone);
        
        %% --- Determination des correspondances de numerotation "LOCALE" - "GLOBALE"
        vec_correspondance_n_noeud_pha_local_n_noeud_pha_global = vec_n_noeuds_pha_sub_zone;
        t_fin = cputime;
        
        disp(['        ' num2str(t_fin-t_ini) ' s']);
        
        %% --- Affichage des noeuds de contrainte et de deplacement dans la subzone
        
%         figure;
%         hold on;
%         
%         plot3(mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,1),mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,2),mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,3),'xk');
%         plot3(vec_x_noeuds_mes_sub_zone,vec_y_noeuds_mes_sub_zone,vec_z_noeuds_mes_sub_zone,'or');
%         
%         grid;
%         xlabel('x (m)');
%         ylabel('y (m)');
%         zlabel('z (m)');
%         title('maillage siub-zone');
%         legend('sig','mes');
        
        %% --- Determination des noeuds situes sur la frontiere de la sub-zone
        
        %   t_ini = cputime;
        mat_n_noeuds_faces_globale = zeros(nb_faces_elem_ref*length(vec_n_elem_sig_sub_zone),size(elem_sig_ref.n_noeuds_faces,2));
        n_face_globale = 0;
        
        %% --- On determine tous les noeuds de toutes les faces des elements de contrainte de la sub-zone
        
        for nn_elem_sig = 1:length(liste_elem_sig_sub_zone)
            
            elem_sig = liste_elem_sig_sub_zone{nn_elem_sig};
            
            vec_n_noeuds_sig = elem_sig.vec_n_noeuds;
            mat_n_noeuds_faces = vec_n_noeuds_sig(elem_sig_ref.n_noeuds_faces);
            
            for n_face = 1:size(mat_n_noeuds_faces,1)
                
                vec_noeuds_face = mat_n_noeuds_faces(n_face,:);
                vec_noeuds_face = sort(vec_noeuds_face(vec_noeuds_face > 0));
                
                n_face_globale = n_face_globale+1;
                mat_n_noeuds_faces_globale(n_face_globale,1:length(vec_noeuds_face)) = vec_noeuds_face;
                
            end
            
        end
        
        %% --- On determine les faces dont les noeuds n'apparaissent qu'une seule fois => ils constituent la frontiere de la sub-zone
        
        vec_test_face_globale_commune = false(1,size(mat_n_noeuds_faces_globale,1));
        
        for n_face_globale = 1:size(mat_n_noeuds_faces_globale,1)-1
            
            mat_abs_delta_n_noeuds_faces = abs(mat_n_noeuds_faces_globale(n_face_globale+1:end,:)-ones(size(mat_n_noeuds_faces_globale,1)-n_face_globale-1+1,1)*mat_n_noeuds_faces_globale(n_face_globale,:));
            vec_somme_abs_delta_n_noeuds_faces = sum(mat_abs_delta_n_noeuds_faces,2);
            [min_vec_somme_abs_delta_n_noeuds_faces,i_min_vec_somme_abs_delta_n_noeuds_faces] = min(vec_somme_abs_delta_n_noeuds_faces);
            
            if ( min_vec_somme_abs_delta_n_noeuds_faces == 0 )
                
                vec_test_face_globale_commune(n_face_globale) = true;
                vec_test_face_globale_commune(n_face_globale+i_min_vec_somme_abs_delta_n_noeuds_faces) = true;
                
            end
            
        end
        
        vec_noeuds_frontieres = unique(reshape(mat_n_noeuds_faces_globale(~vec_test_face_globale_commune,:),1,[]));
        
        %% --- Affichage des noeuds de contrainte dans la subzone et des noeuds frontiere de la suzone
        
            %figure;
            %hold on;
            %plot3(mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,1),mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,2),mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,3),'xk');
            %plot3(mat_pos_maillage_sig(vec_noeuds_frontieres,1),mat_pos_maillage_sig(vec_noeuds_frontieres,2),mat_pos_maillage_sig(vec_noeuds_frontieres,3),'or');
            
            %grid;
            %xlabel('x (m)');
            %ylabel('y (m)');
            %zlabel('z (m)');
            %title('maillage siub-zone');
            %legend('vol.','front.');
            
        %   t_fin = cputime;
        %   disp(['        ' num2str(t_fin-t_ini) ' s']);
        
        vec_n_noeuds_frontieres_local = vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(vec_noeuds_frontieres);
        
        %% --- Determination des conditions aux limites sur la sub-zone
        
        disp('        determinations des conditions aux limites de la sub-zone');
        t_ini = cputime;
        
        nb_noeuds_sig_sub_zone = length(vec_n_noeuds_sig_sub_zone);
        
        if ( (strcmp(struct_CL.type,'mesure') == 1) || (strcmp(struct_CL.type,'mesure_filtree') == 1) )
            
            liste_DL_bloque = cell(1,2*nb_DDL_par_noeud);
            
            for n_DL_imp = 1:nb_DDL_par_noeud
                
                vec_n_DL_bloque_U = nb_DDL_par_noeud*nb_noeuds_sig_sub_zone+(vec_n_noeuds_frontieres_local-1)*nb_DDL_par_noeud+n_DL_imp;
                vec_val_DL_bloque_U = mat_U_sig_3D(vec_noeuds_frontieres,n_DL_imp);
                liste_DL_bloque{n_DL_imp} = struct('type','U','vec_n_DL_bloque',vec_n_DL_bloque_U,'vec_val_DL_bloque',vec_val_DL_bloque_U','vec_n_noeud',vec_n_noeuds_frontieres_local);
                vec_n_DL_bloque_W = (vec_n_noeuds_frontieres_local-1)*nb_DDL_par_noeud+n_DL_imp;
                liste_DL_bloque{nb_DDL_par_noeud+n_DL_imp} = struct('type','W','vec_n_DL_bloque',vec_n_DL_bloque_W,'vec_val_DL_bloque',zeros(size(vec_n_DL_bloque_W)),'vec_n_noeud',vec_n_noeuds_frontieres_local);
            
            end
            
        elseif ( (strcmp(struct_CL.type,'sans') == 1) )
            
            liste_DL_bloque = cell(1,nb_DDL_par_noeud);
            
            for n_DL_imp = 1:nb_DDL_par_noeud
                
                vec_n_DL_bloque_W = (vec_n_noeuds_frontieres_local-1)*nb_DDL_par_noeud+n_DL_imp;
                liste_DL_bloque{n_DL_imp} = struct('type','W','vec_n_DL_bloque',vec_n_DL_bloque_W,'vec_val_DL_bloque',zeros(size(vec_n_DL_bloque_W)),'vec_n_noeud',vec_n_noeuds_frontieres_local);
            
            end
            
        end
        
        t_fin = cputime;
        disp(['        ' num2str(t_fin-t_ini) ' s']);
        
        %% --- Determination des matrices de masse et de raideur sur la sub-zone
        
        %%%%%%%%%%%%%%%%%%%%%%%% Raideur equivalente %%%%%%%%%%%%%%%%%%%%%%
        
        disp(['        matrices equivalentes - ' num2str(nb_noeuds_sig_sub_zone*nb_DDL_par_noeud) ' DDL']);
        t_ini = cputime;
        
        [K,T,M,d_K_d_p] = raideur_ERCM(liste_elem_pha,liste_elem_sig_sub_zone,liste_elem_ref,struct_param_masse_raideur,vec_correspondance_n_noeud_sig_global_n_noeud_sig_local,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local,mat_pos_maillage_sig,nb_DDL_par_noeud,struct_param_comportement_a_identifier,struct_param_comportement_normalisation);
        t_fin = cputime;
        disp(['        ' num2str(t_fin-t_ini) ' s']);
        
        %% --- Determination de l'operateur de projection sur la sub-zone
        
        disp('        operateur de projection');
        t_ini = cputime;
        %  [N,D] = projection_mes(mat_pos_mes,mat_pos_maillage_sig,liste_elem_sig_sub_zone,mat_n_sig,liste_elem_ref,nb_DDL_par_noeud);
        
        ni_N = nb_points_mesure_sub_zone*nb_DDL_par_noeud;
        nj_N = length(vec_n_noeuds_sig_sub_zone)*nb_DDL_par_noeud;
        
        Nf = squeeze(liste_elem_ref{liste_elem_sig_sub_zone{1}.n_elem_ref}.f_Nf(0,0,0));
        
        vec_i_N = nan(1,length(Nf)*ni_N);
        vec_j_N = nan(1,length(Nf)*ni_N);
        vec_s_N = nan(1,length(Nf)*ni_N);
        
        i_prec = 0;
        n_mes_sub_zone = 0;
        
        %   vec_test_points_mes_dans_sub_zone = false(1,size(mat_pos_mes,2));
        vec_n_points_mes_dans_sub_zone = nan(1,size(mat_pos_mes,2));
        
        for nn_elem_sig = 1:length(liste_elem_sig_sub_zone)
            
            elem_sig = liste_elem_sig_sub_zone{nn_elem_sig};
            
            for nn_mes_local = 1:length(elem_sig.vec_i_mes)
                
                i_mes = elem_sig.vec_i_mes(nn_mes_local);
                j_mes = elem_sig.vec_j_mes(nn_mes_local);
                k_mes = elem_sig.vec_k_mes(nn_mes_local);
                
                n_mes = 1+(i_mes-1)+ni_mes*((j_mes-1)+nj_mes*(k_mes-1));
                
                %     if ( ~vec_test_points_mes_dans_sub_zone(n_mes) )
                %      vec_test_points_mes_dans_sub_zone(n_mes) = true;
                
                n_mes_sub_zone = n_mes_sub_zone+1;
                vec_n_points_mes_dans_sub_zone(n_mes_sub_zone) = n_mes;
                
                x_ref = elem_sig.vec_ksi_mes(nn_mes_local);
                y_ref = elem_sig.vec_eta_mes(nn_mes_local);
                z_ref = elem_sig.vec_zeta_mes(nn_mes_local);
                
                Nf = squeeze(liste_elem_ref{elem_sig.n_elem_ref}.f_Nf(x_ref,y_ref,z_ref));
                
                for n_DDL = 1:nb_DDL_par_noeud
                    
                    vec_i_N(i_prec+(1:length(Nf))) = nb_DDL_par_noeud*(n_mes_sub_zone-1)+n_DDL;
                    vec_j_N(i_prec+(1:length(Nf))) = nb_DDL_par_noeud*(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)-1)+n_DDL;
                    vec_s_N(i_prec+(1:length(Nf))) = Nf;
                    
                    i_prec = i_prec+length(Nf);
                    
                end
                %     end
            end
        end
        
        vec_n_points_mes_dans_sub_zone = vec_n_points_mes_dans_sub_zone(~isnan(vec_n_points_mes_dans_sub_zone));
        
        vec_i_N = vec_i_N(~isnan(vec_i_N));
        vec_j_N = vec_j_N(~isnan(vec_j_N));
        vec_s_N = vec_s_N(~isnan(vec_s_N));
        
        %   ni_N = max(vec_i_N);
        
        N = sparse(vec_i_N,vec_j_N,vec_s_N,ni_N,nj_N);
        D = N'*N;
        t_fin = cputime;
        disp(['      ' num2str(t_fin-t_ini) ' s']);
        
        %% Weeeee STOPPED HERE
        
        %% --- Calcul du seconds membre R
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%% Calcul du seconds membre R %%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        disp('        second membre');
        t_ini = cputime;
        nb_mes = length(vec_n_points_mes_dans_sub_zone);
        vec_U_mes = zeros(nb_DDL_par_noeud*nb_mes,1);
        
        for n_DDL = 1:nb_DDL_par_noeud
            
            vec_U_mes(n_DDL:nb_DDL_par_noeud:end) = mat_U_mes(n_DDL,vec_n_points_mes_dans_sub_zone);
            
        end
        
        vec_F = zeros(size(K,1),1);
        vec_R = N'*vec_U_mes;
        
        %% --- %C% Affichage des deplacements projetes sur la grille de contraine %C%
        
        %   vec_R_U = (D\N')*vec_U_mes;
        % % champs calcules
        %   vec_R_Ux = vec_R_U(1:nb_DDL_par_noeud:size(K,1));
        %   vec_R_Uy = vec_R_U(2:nb_DDL_par_noeud:size(K,1));
        %   vec_R_Uz = vec_R_U(3:nb_DDL_par_noeud:size(K,1));
        % %  mat_pos_maillage_sig_sub_zone = mat_pos_maillage_sig(vec_correspondance_n_noeud_sig_local_n_noeud_sig_global,:);
        %   mat_pos_maillage_sig_sub_zone = mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,:);
        %   vec_x_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,1);
        %   vec_y_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,2);
        %   vec_z_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,3);
        %   F_Ux_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_R_Ux),'linear','none');
        %   F_Ux_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_R_Ux),'linear','none');
        %   F_Uy_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_R_Uy),'linear','none');
        %   F_Uy_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_R_Uy),'linear','none');
        %   F_Uz_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_R_Uz),'linear','none');
        %   F_Uz_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_R_Uz),'linear','none');
        %   vec_grille_x_sig_sub_zone = unique(sort(vec_x_sig_sub_zone));
        %   vec_grille_y_sig_sub_zone = unique(sort(vec_y_sig_sub_zone));
        %   vec_grille_z_sig_sub_zone = unique(sort(vec_z_sig_sub_zone));
        %   [mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone] = meshgrid(vec_grille_x_sig_sub_zone,vec_grille_y_sig_sub_zone,vec_grille_z_sig_sub_zone);
        %   mat_Ux_real_sig_sub_zone_affichage = F_Ux_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Ux_imag_sig_sub_zone_affichage = F_Ux_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uy_real_sig_sub_zone_affichage = F_Uy_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uy_imag_sig_sub_zone_affichage = F_Uy_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uz_real_sig_sub_zone_affichage = F_Uz_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uz_imag_sig_sub_zone_affichage = F_Uz_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   coupe_x_sig = [min(vec_grille_x_sig_sub_zone) mean(vec_grille_x_sig_sub_zone) max(vec_grille_x_sig_sub_zone)];
        %   coupe_y_sig = max(vec_grille_y_sig_sub_zone);
        %   coupe_z_sig = [min(vec_grille_z_sig_sub_zone) mean(vec_grille_z_sig_sub_zone)];
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Ux projete), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Ux projete), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uy projete), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uy projete), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uz projete), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uz projete), (m)');
        %   clear vec_R_U vec_R_Ux vec_R_Uy vec_R_Uz mat_pos_maillage_sig_sub_zone vec_x_sig_sub_zone vec_y_sig_sub_zone vec_z_sig_sub_zone F_Ux_real_sig F_Ux_imag_sig F_Uy_real_sig F_Uy_imag_sig F_Uz_real_sig F_Uz_imag_sig vec_grille_x_sig_sub_zone vec_grille_y_sig_sub_zone vec_grille_z_sig_sub_zone mat_x_sig_sub_zone mat_y_sig_sub_zone mat_z_sig_sub_zone mat_Ux_real_sig_sub_zone_affichage mat_Ux_imag_sig_sub_zone_affichage mat_Uy_real_sig_sub_zone_affichage mat_Uy_imag_sig_sub_zone_affichage mat_Uz_real_sig_sub_zone_affichage mat_Uz_imag_sig_sub_zone_affichage coupe_x_sig coupe_y_sig coupe_z_sig;
        t_fin = cputime;
        disp(['      ' num2str(t_fin-t_ini) ' s']);
        
        %% --- Determination de la raideur globale (elastodynamics)
        
        disp('        raideur globale');
        t_ini = cputime;
        K_tilde = K-omega_mec^2*M; % direct FEM problems.
        
        [vec_i_K_tilde,vec_j_K_tilde,vec_s_K_tilde] = find(K_tilde);
        [vec_i_T,vec_j_T,vec_s_T] = find(T);
        [vec_i_D,vec_j_D,vec_s_D] = find(D);
        
        % MERC
        
        kappa_critique = max(abs(vec_s_K_tilde))/max(abs(vec_s_D));
        %  kappa = kappa_critique/1000;
        %  kappa = kappa_critique;
        %  kappa = kappa_critique*1000;
        
        if ( strcmp(type_identification,'FEMU_DFC') == 1 )
        elseif ( strcmp(type_identification,'FEMU_AFC') == 1 )
            
            kappa = -1;
            
            vec_i_global = [vec_i_K_tilde'                   , (vec_j_K_tilde+size(K_tilde,1))' , (vec_i_D+size(K_tilde,1))'];
            vec_j_global = [(vec_j_K_tilde+size(K_tilde,2))' , vec_i_K_tilde'                   , (vec_j_D+size(K_tilde,2))'];
            vec_s_global = [vec_s_K_tilde.'                 , conj(vec_s_K_tilde).'            , (-kappa*vec_s_D.')];
            
            K_global = sparse(vec_i_global,vec_j_global,vec_s_global,2*size(K_tilde,1),2*size(K_tilde,2));
            
            clear vec_i_K_tilde vec_j_K_tilde vec_s_K_tilde;
            clear vec_i_T vec_j_T vec_s_T;
            clear vec_i_D vec_j_D vec_s_D;
            clear vec_i_global vec_j_global vec_s_global;
            
            %   clear K M T K_tilde D;
            
            vec_F_global = zeros(size(K_global,1),1);
            vec_F_global(1:size(K_tilde,1)) = vec_F;
            vec_F_global(size(K_tilde,1)+1:end) = -kappa*vec_R;
            
            %  figure;spy(K_global);title('structure K global');
            
        elseif ( strcmp(type_identification,'MERC') == 1 )
            
            disp(['           valeur a donner a "kappa" pour avoir des poids equivalents sur "K_tilde" et "D" : ' num2str(kappa_critique)]);
            disp(['           valeur de kappa : ' num2str(kappa)]);
            
            vec_i_global = [vec_i_T'  , vec_i_K_tilde'                   , (vec_j_K_tilde+size(K_tilde,1))' , (vec_i_D+size(K_tilde,1))'];
            vec_j_global = [vec_j_T'  , (vec_j_K_tilde+size(K_tilde,2))' , vec_i_K_tilde'                   , (vec_j_D+size(K_tilde,2))'];
            vec_s_global = [vec_s_T.' ,  vec_s_K_tilde.'                 , conj(vec_s_K_tilde).'            , (-kappa*vec_s_D.')];
            
            K_global = sparse(vec_i_global,vec_j_global,vec_s_global,2*size(K_tilde,1),2*size(K_tilde,2));
            
            clear vec_i_K_tilde vec_j_K_tilde vec_s_K_tilde;
            clear vec_i_T vec_j_T vec_s_T;
            clear vec_i_D vec_j_D vec_s_D;
            clear vec_i_global vec_j_global vec_s_global;
            
            %   clear K M T K_tilde D;
            
            vec_F_global = zeros(size(K_global,1),1);
            vec_F_global(1:size(K_tilde,1)) = vec_F;
            vec_F_global(size(K_tilde,1)+1:end) = -kappa*vec_R;
            
            %  figure;spy(K_global);title('structure K global');
            
            % substitution of the lines/colums supporting BC  
            %% run it through here
            [Ks_global,Fs_global,U_impose_global,vec_n_DDL_conserves_global,vec_n_DDL_supprimes_global] = maj_matrices(K_global,vec_F_global,liste_DL_bloque);
            
        end
        
        %% --- %C% No boundary conditions %C%
        
        %   vec_i_global = [vec_i_K_tilde'              , vec_i_D'                     , (vec_i_K_tilde+size(K_tilde,1))'];
        %   vec_j_global = [vec_j_K_tilde'              , (vec_j_D+size(K_tilde,2))'   , (vec_j_K_tilde++size(K_tilde,1))'];
        %   vec_s_global = [conj(vec_s_K_tilde).'       , vec_s_D .'                   , (vec_s_K_tilde).'];
        %   K_global = sparse(vec_i_global,vec_j_global,vec_s_global,2*size(K_tilde,1),2*size(K_tilde,2));
        %   clear vec_i_K_tilde vec_j_K_tilde vec_s_K_tilde;
        %   clear vec_i_T vec_j_T vec_s_T;
        %   clear vec_i_D vec_j_D vec_s_D;
        %   clear vec_i_global vec_j_global vec_s_global;
        % %  clear K M T K_tilde D;
        %   vec_F_global = zeros(size(K_global,1),1);
        %   vec_F_global(1:size(K_tilde,1)) = vec_R;
        %   vec_F_global(size(K_tilde,1)+1:end) = vec_F;
        % % figure;spy(K_global);title('structure K global');
        
        %% --- MAJ du systeme sur K_global pour tenir compte des CL
        
%        [Ks_global,Fs_global,U_impose_global,vec_n_DDL_conserves_global,vec_n_DDL_supprimes_global] = maj_matrices(K_global,vec_F_global,liste_DL_bloque);
        t_fin = cputime;
        disp(['      ' num2str(t_fin-t_ini) ' s']);
        
        %% --- %C% TEST POUR VOIR SI LES IMPLEMENTATIONS DES MATRICES SONT CORRECTES =>  CALCUL DES (U) en CL imposees en deplacement %C%
        
        % % MAJ du systeme sur K_tilde pour tenir compte des CL
        
        %   vec_x_sig = mat_pos_maillage_sig(:,1);
        %   vec_y_sig = mat_pos_maillage_sig(:,2);
        %   vec_z_sig = mat_pos_maillage_sig(:,3);
        
        %   vec_x_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,1)),1,[]);
        %   vec_y_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,2)),1,[]);
        %   vec_z_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,3)),1,[]);
        
        %   vec_ux_mes = reshape(squeeze(mat_U_mes_3D(:,:,:,1)),1,[]);
        %   vec_uy_mes = reshape(squeeze(mat_U_mes_3D(:,:,:,2)),1,[]);
        %   vec_uz_mes = reshape(squeeze(mat_U_mes_3D(:,:,:,3)),1,[]);
        
        %   vec_n_a_conserver = find ( ~isnan(vec_x_mes) & ~isnan() & ~isnan(vec_z_mes) );
        
        %   vec_x_mes = vec_x_mes(vec_n_a_conserver)';
        %   vec_y_mes = vec_y_mes(vec_n_a_conserver)';
        %   vec_z_mes = vec_z_mes(vec_n_a_conserver)';
        
        %   vec_ux_mes = vec_ux_mes(vec_n_a_conserver)';
        %   vec_uy_mes = vec_uy_mes(vec_n_a_conserver)';
        %   vec_uz_mes = vec_uz_mes(vec_n_a_conserver)';
        
        % %   F_Ux = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_ux_mes,'linear','none');
        % %   F_Uy = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_uy_mes,'linear','none');
        % %   F_Uz = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_uz_mes,'linear','none');
       

        %   F_Ux = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_ux_mes,'linear','linear');
        %   F_Uy = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_uy_mes,'linear','linear');
        %   F_Uz = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_uz_mes,'linear','linear');
        
        %   vec_ux_sig = F_Ux(vec_x_sig,vec_y_sig,vec_z_sig);
        %   vec_uy_sig = F_Uy(vec_x_sig,vec_y_sig,vec_z_sig);
        %   vec_uz_sig = F_Uz(vec_x_sig,vec_y_sig,vec_z_sig);
        
        %   mat_U_sig_3D = [vec_ux_sig.';vec_uy_sig.';vec_uz_sig.'].';
        
        %   clear vec_x_mes vec_y_mes vec_z_mes vec_ux_mes vec_uy_mes vec_uz_mes vec_n_a_conserver vec_x_sig vec_y_sig vec_z_sig vec_ux_sig vec_uy_sig vec_uz_sig F_Ux F_Uy F_Uz;
        
        %   liste_DL_bloque_U_impose = cell(1,nb_DDL_par_noeud);
        
        %   for n_DL_imp = 1:nb_DDL_par_noeud
        %    vec_n_DL_bloque_U_impose = (vec_n_noeuds_frontieres_local-1)*nb_DDL_par_noeud+n_DL_imp;
        %    vec_val_DL_bloque_U_impose = mat_U_sig_3D(vec_noeuds_frontieres,n_DL_imp)';
        %    liste_DL_bloque_U_impose{n_DL_imp} = struct('type','U','vec_n_DL_bloque',vec_n_DL_bloque_U_impose,'vec_val_DL_bloque',vec_val_DL_bloque_U_impose,'vec_n_noeud',vec_n_noeuds_frontieres_local);
        %   end
        
        %   vec_F_U_impose = zeros(size(K_tilde,1),1);
        
        %   [Ks_U_impose,Fs_U_impose,U_impose_local,vec_n_DDL_conserves_U_impose,vec_n_DDL_supprimes_U_impose] = maj_matrices(K_tilde,vec_F_U_impose,liste_DL_bloque_U_impose);
       
        %   Us_local = Ks_U_impose\Fs_U_impose;
        %   U_local = CL_assemblage(Us_local,U_impose_local,vec_n_DDL_conserves_U_impose,vec_n_DDL_supprimes_U_impose);
        %   Ux_local = U_local(1:nb_DDL_par_noeud:end);
        %   Uy_local = U_local(2:nb_DDL_par_noeud:end);
        %   Uz_local = U_local(3:nb_DDL_par_noeud:end);
       
        %   figure;hold on;plot(real(Ux_local),'r');plot(real(Uy_local),'g');plot(real(Uz_local),'b');title('real(U) local');legend('Ux','Uy','Uz');
        %   figure;hold on;plot(imag(Ux_local),'r');plot(imag(Uy_local),'g');plot(imag(Uz_local),'b');title('imag(U) local');legend('Ux','Uy','Uz');
        
        %% ** AFFICHAGE DES CHAMPS CALCULES POUR LA SUB-ZONE COURANTE ***
        
        %% --- Deplacements mesures
        
        %   vec_Ux_mes_sub_zone = mat_U_mes(1,vec_n_noeuds_mes_sub_zone);
        %   vec_Uy_mes_sub_zone = mat_U_mes(2,vec_n_noeuds_mes_sub_zone);
        %   vec_Uz_mes_sub_zone = mat_U_mes(3,vec_n_noeuds_mes_sub_zone);
        
        %   vec_i_mes_sub_zone = vec_i_mes(vec_n_noeuds_mes_sub_zone);
        %   vec_j_mes_sub_zone = vec_j_mes(vec_n_noeuds_mes_sub_zone);
        %   vec_k_mes_sub_zone = vec_k_mes(vec_n_noeuds_mes_sub_zone);
        
        %   vec_np_mes_sub_zone_global = vec_i_mes_sub_zone+ni_mes*((vec_j_mes_sub_zone-1)+nj_mes*(vec_k_mes_sub_zone-1));
        
        %   vec_i_mes_sub_zone = vec_i_mes_sub_zone-min(vec_i_mes_sub_zone)+1;
        %   vec_j_mes_sub_zone = vec_j_mes_sub_zone-min(vec_j_mes_sub_zone)+1;
        %   vec_k_mes_sub_zone = vec_k_mes_sub_zone-min(vec_k_mes_sub_zone)+1;
        
        %   ni_mes_sub_zone = max(vec_i_mes_sub_zone);
        %   nj_mes_sub_zone = max(vec_j_mes_sub_zone);
        %   nk_mes_sub_zone = max(vec_k_mes_sub_zone);
        
        %   vec_np_mes_sub_zone_local = vec_i_mes_sub_zone+ni_mes_sub_zone*((vec_j_mes_sub_zone-1)+nj_mes_sub_zone*(vec_k_mes_sub_zone-1));
        
        %   mat_Ux_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %   mat_Ux_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %   mat_Uy_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %   mat_Uy_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %   mat_Uz_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %   mat_Uz_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        
        %   mat_Ux_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Ux_mes_sub_zone);
        %   mat_Ux_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Ux_mes_sub_zone);
        %   mat_Uy_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Uy_mes_sub_zone);
        %   mat_Uy_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Uy_mes_sub_zone);
        %   mat_Uz_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Uz_mes_sub_zone);
        %   mat_Uz_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Uz_mes_sub_zone);
        
        %   vec_x_grille_mes_sub_zone = struct_grille_mes.x_min+struct_grille_mes.dx*((min(vec_i_mes(vec_np_mes_sub_zone_global)):max(vec_i_mes(vec_np_mes_sub_zone_global)))-1);
        %   vec_y_grille_mes_sub_zone = struct_grille_mes.y_min+struct_grille_mes.dy*((min(vec_j_mes(vec_np_mes_sub_zone_global)):max(vec_j_mes(vec_np_mes_sub_zone_global)))-1);
        %   vec_z_grille_mes_sub_zone = struct_grille_mes.z_min+struct_grille_mes.dz*((min(vec_k_mes(vec_np_mes_sub_zone_global)):max(vec_k_mes(vec_np_mes_sub_zone_global)))-1);
        
        %   [mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone] = meshgrid(vec_y_grille_mes_sub_zone,vec_x_grille_mes_sub_zone,vec_z_grille_mes_sub_zone);
        
        %   coupe_x_mes = [min(vec_x_grille_mes_sub_zone) mean(vec_x_grille_mes_sub_zone) max(vec_x_grille_mes_sub_zone)];
        %   coupe_y_mes = max(vec_y_grille_mes_sub_zone);
        %   coupe_z_mes = [min(vec_z_grille_mes_sub_zone) mean(vec_z_grille_mes_sub_zone)];
        
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Ux_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Ux mes), (m)');
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Ux_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Ux mes), (m)');
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uy_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Uy mes), (m)');
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uy_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Uy mes), (m)');
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uz_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Uz mes), (m)');
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uz_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Uz mes), (m)');
        
        %% --- Champs calcules
        
        %   vec_Ux_sub_zone = Ux_local;
        %   vec_Uy_sub_zone = Uy_local;
        %   vec_Uz_sub_zone = Uz_local;
        
        % %  mat_pos_maillage_sig_sub_zone = mat_pos_maillage_sig(vec_correspondance_n_noeud_sig_local_n_noeud_sig_global,:);
        %   mat_pos_maillage_sig_sub_zone = mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,:);
        
        %   vec_x_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,1);
        %   vec_y_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,2);
        %   vec_z_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,3);
        
        %   F_Ux_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Ux_sub_zone),'linear','none');
        %   F_Ux_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Ux_sub_zone),'linear','none');
        %   F_Uy_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Uy_sub_zone),'linear','none');
        %   F_Uy_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Uy_sub_zone),'linear','none');
        %   F_Uz_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Uz_sub_zone),'linear','none');
        %   F_Uz_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Uz_sub_zone),'linear','none');
        
        %   vec_grille_x_sig_sub_zone = unique(sort(vec_x_sig_sub_zone));
        %   vec_grille_y_sig_sub_zone = unique(sort(vec_y_sig_sub_zone));
        %   vec_grille_z_sig_sub_zone = unique(sort(vec_z_sig_sub_zone));
        
        %   [mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone] = meshgrid(vec_grille_x_sig_sub_zone,vec_grille_y_sig_sub_zone,vec_grille_z_sig_sub_zone);
        
        %   mat_Ux_real_sig_sub_zone_affichage = F_Ux_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Ux_imag_sig_sub_zone_affichage = F_Ux_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uy_real_sig_sub_zone_affichage = F_Uy_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uy_imag_sig_sub_zone_affichage = F_Uy_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uz_real_sig_sub_zone_affichage = F_Uz_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uz_imag_sig_sub_zone_affichage = F_Uz_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        
        %   coupe_x_sig = [min(vec_grille_x_sig_sub_zone) mean(vec_grille_x_sig_sub_zone) max(vec_grille_x_sig_sub_zone)];
        %   coupe_y_sig = max(vec_grille_y_sig_sub_zone);
        %   coupe_z_sig = [min(vec_grille_z_sig_sub_zone) mean(vec_grille_z_sig_sub_zone)];
        
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Ux), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Ux), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uy), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uy), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uz), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uz), (m)');
        
        %% --- Resolution et reecriture de U (assemblage du vecteur resultat)

        disp('        resolution');
        t_ini = cputime;
        
        nb_DDL_K = size(K,1);
        
        Us_global = Ks_global\Fs_global;
        U_global = CL_assemblage(Us_global,U_impose_global,vec_n_DDL_conserves_global,vec_n_DDL_supprimes_global);
        
        Wx = U_global(1:nb_DDL_par_noeud:nb_DDL_K);
        Wy = U_global(2:nb_DDL_par_noeud:nb_DDL_K);
        Wz = U_global(3:nb_DDL_par_noeud:nb_DDL_K);
        
        Ux = U_global(nb_DDL_K+1:nb_DDL_par_noeud:end);
        Uy = U_global(nb_DDL_K+2:nb_DDL_par_noeud:end);
        Uz = U_global(nb_DDL_K+3:nb_DDL_par_noeud:end);
        
       % figure;hold on;plot(real(Ux),'r');plot(real(Uy),'g');plot(real(Uz),'b');title('real(U)');legend('Ux','Uy','Uz');title('Re(U calc))');
       % figure;hold on;plot(imag(Ux),'r');plot(imag(Uy),'g');plot(imag(Uz),'b');title('imag(U)');legend('Ux','Uy','Uz');title('Im(U calc))');
       % figure;hold on;plot(real(Wx),'r');plot(real(Wy),'g');plot(real(Wz),'b');title('real(W)');legend('Wx','Wy','Wz');title('Re(W calc))');
       % figure;hold on;plot(imag(Wx),'r');plot(imag(Wy),'g');plot(imag(Wz),'b');title('imag(W)');legend('Wx','Wy','Wz');title('Im(W calc))');
        
        %   figure;hold on;plot(real(Ux(vec_n_noeuds_frontieres_local)),'-r');plot(real(Uy(vec_n_noeuds_frontieres_local)),'-g');plot(real(Uz(vec_n_noeuds_frontieres_local)),'-b');grid;legend('Ux','Uy','Uz');title('Re(U calc frontiere))');
        %   figure;hold on;plot(imag(Ux(vec_n_noeuds_frontieres_local)),'-r');plot(imag(Uy(vec_n_noeuds_frontieres_local)),'-g');plot(imag(Uz(vec_n_noeuds_frontieres_local)),'-b');grid;legend('Ux','Uy','Uz');title('Im(U calc frontiere))');
        %   figure;hold on;plot(real(Wx(vec_n_noeuds_frontieres_local)),'-r');plot(real(Wy(vec_n_noeuds_frontieres_local)),'-g');plot(real(Wz(vec_n_noeuds_frontieres_local)),'-b');grid;legend('Wx','Wy','Wz');title('Re(W calc frontiere))');
        %   figure;hold on;plot(imag(Wx(vec_n_noeuds_frontieres_local)),'-r');plot(imag(Wy(vec_n_noeuds_frontieres_local)),'-g');plot(imag(Wz(vec_n_noeuds_frontieres_local)),'-b');grid;legend('Wx','Wy','Wz');title('Im(W calc frontiere))');
        
        t_fin = cputime;
        disp(['        ' num2str(t_fin-t_ini) ' s']);
        
        %% ** AFFICHAGE DES CHAMPS CALCULES POUR LA SUB-ZONE COURANTE **
        
        %% --- %C% Deplacements mesures %C%
        
        %   vec_Ux_mes_sub_zone = mat_U_mes(1,vec_n_noeuds_mes_sub_zone);
        %   vec_Uy_mes_sub_zone = mat_U_mes(2,vec_n_noeuds_mes_sub_zone);
        %   vec_Uz_mes_sub_zone = mat_U_mes(3,vec_n_noeuds_mes_sub_zone);
        
        %   vec_i_mes_sub_zone = vec_i_mes(vec_n_noeuds_mes_sub_zone);
        %   vec_j_mes_sub_zone = vec_j_mes(vec_n_noeuds_mes_sub_zone);
        %   vec_k_mes_sub_zone = vec_k_mes(vec_n_noeuds_mes_sub_zone);
        
        %   vec_np_mes_sub_zone_global = vec_i_mes_sub_zone+ni_mes*((vec_j_mes_sub_zone-1)+nj_mes*(vec_k_mes_sub_zone-1));
        %   vec_i_mes_sub_zone = vec_i_mes_sub_zone-min(vec_i_mes_sub_zone)+1;
        %   vec_j_mes_sub_zone = vec_j_mes_sub_zone-min(vec_j_mes_sub_zone)+1;
        %   vec_k_mes_sub_zone = vec_k_mes_sub_zone-min(vec_k_mes_sub_zone)+1;
        
        %   ni_mes_sub_zone = max(vec_i_mes_sub_zone);
        %   nj_mes_sub_zone = max(vec_j_mes_sub_zone);
        %   nk_mes_sub_zone = max(vec_k_mes_sub_zone);
        
        %   vec_np_mes_sub_zone_local = vec_i_mes_sub_zone+ni_mes_sub_zone*((vec_j_mes_sub_zone-1)+nj_mes_sub_zone*(vec_k_mes_sub_zone-1));
        
        %   mat_Ux_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %   mat_Ux_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %   mat_Uy_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %   mat_Uy_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %   mat_Uz_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %   mat_Uz_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        
        %   mat_Ux_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Ux_mes_sub_zone);
        %   mat_Ux_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Ux_mes_sub_zone);
        %   mat_Uy_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Uy_mes_sub_zone);
        %   mat_Uy_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Uy_mes_sub_zone);
        %   mat_Uz_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Uz_mes_sub_zone);
        %   mat_Uz_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Uz_mes_sub_zone);
        
        %   vec_x_grille_mes_sub_zone = struct_grille_mes.x_min+struct_grille_mes.dx*((min(vec_i_mes(vec_np_mes_sub_zone_global)):max(vec_i_mes(vec_np_mes_sub_zone_global)))-1);
        %   vec_y_grille_mes_sub_zone = struct_grille_mes.y_min+struct_grille_mes.dy*((min(vec_j_mes(vec_np_mes_sub_zone_global)):max(vec_j_mes(vec_np_mes_sub_zone_global)))-1);
        %   vec_z_grille_mes_sub_zone = struct_grille_mes.z_min+struct_grille_mes.dz*((min(vec_k_mes(vec_np_mes_sub_zone_global)):max(vec_k_mes(vec_np_mes_sub_zone_global)))-1);
        
        %   [mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone] = meshgrid(vec_y_grille_mes_sub_zone,vec_x_grille_mes_sub_zone,vec_z_grille_mes_sub_zone);
        
        %   coupe_x_mes = [min(vec_x_grille_mes_sub_zone) mean(vec_x_grille_mes_sub_zone) max(vec_x_grille_mes_sub_zone)];
        %   coupe_y_mes = max(vec_y_grille_mes_sub_zone);
        %   coupe_z_mes = [min(vec_z_grille_mes_sub_zone) mean(vec_z_grille_mes_sub_zone)];
        
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Ux_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Ux mes), (m)');
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Ux_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Ux mes), (m)');
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uy_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Uy mes), (m)');
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uy_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Uy mes), (m)');
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uz_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Uz mes), (m)');
        %   figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uz_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Uz mes), (m)');
        
        %% --- %C% Champs calcules %C%
        
        %   vec_Ux_sub_zone = Ux;
        %   vec_Uy_sub_zone = Uy;
        %   vec_Uz_sub_zone = Uz;
        %   vec_Wx_sub_zone = Wx;
        %   vec_Wy_sub_zone = Wy;
        %   vec_Wz_sub_zone = Wz;
        
        %   mat_pos_maillage_sig_sub_zone = mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,:);
        
        %   vec_x_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,1);
        %   vec_y_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,2);
        %   vec_z_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,3);
        
        %   F_Ux_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Ux_sub_zone),'linear','none');
        %   F_Ux_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Ux_sub_zone),'linear','none');
        %   F_Uy_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Uy_sub_zone),'linear','none');
        %   F_Uy_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Uy_sub_zone),'linear','none');
        %   F_Uz_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Uz_sub_zone),'linear','none');
        %   F_Uz_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Uz_sub_zone),'linear','none');
        %   F_Wx_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Wx_sub_zone),'linear','none');
        %   F_Wx_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Wx_sub_zone),'linear','none');
        %   F_Wy_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Wy_sub_zone),'linear','none');
        %   F_Wy_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Wy_sub_zone),'linear','none');
        %   F_Wz_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Wz_sub_zone),'linear','none');
        %   F_Wz_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Wz_sub_zone),'linear','none');
        
        %   vec_grille_x_sig_sub_zone = unique(sort(vec_x_sig_sub_zone));
        %   vec_grille_y_sig_sub_zone = unique(sort(vec_y_sig_sub_zone));
        %   vec_grille_z_sig_sub_zone = unique(sort(vec_z_sig_sub_zone));
        
        %   [mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone] = meshgrid(vec_grille_x_sig_sub_zone,vec_grille_y_sig_sub_zone,vec_grille_z_sig_sub_zone);
        
        %   mat_Ux_real_sig_sub_zone_affichage = F_Ux_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Ux_imag_sig_sub_zone_affichage = F_Ux_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uy_real_sig_sub_zone_affichage = F_Uy_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uy_imag_sig_sub_zone_affichage = F_Uy_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uz_real_sig_sub_zone_affichage = F_Uz_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Uz_imag_sig_sub_zone_affichage = F_Uz_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Wx_real_sig_sub_zone_affichage = F_Wx_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Wx_imag_sig_sub_zone_affichage = F_Wx_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Wy_real_sig_sub_zone_affichage = F_Wy_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Wy_imag_sig_sub_zone_affichage = F_Wy_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Wz_real_sig_sub_zone_affichage = F_Wz_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %   mat_Wz_imag_sig_sub_zone_affichage = F_Wz_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        
        %   coupe_x_sig = [min(vec_grille_x_sig_sub_zone) mean(vec_grille_x_sig_sub_zone) max(vec_grille_x_sig_sub_zone)];
        %   coupe_y_sig = max(vec_grille_y_sig_sub_zone);
        %   coupe_z_sig = [min(vec_grille_z_sig_sub_zone) mean(vec_grille_z_sig_sub_zone)];
        
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Ux), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Ux), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uy), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uy), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uz), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uz), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wx_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Wx), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wx_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Wx), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wy_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Wy), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wy_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Wy), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wz_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Wz), (m)');
        %   figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wz_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Wz), (m)');
        
        %% --- Verification de la numerotation a 3 indices
        
        % % mat_V = nan(3,5,7);
        % % n = 0;
        
        % % vec_i = zeros(1,size(mat_V,1)*size(mat_V,2)*size(mat_V,3));
        % % vec_j = zeros(1,size(mat_V,1)*size(mat_V,2)*size(mat_V,3));
        % % vec_k = zeros(1,size(mat_V,1)*size(mat_V,2)*size(mat_V,3));
        % % vec_V = zeros(1,size(mat_V,1)*size(mat_V,2)*size(mat_V,3));
        
        % % for i = 1:size(mat_V,1)
        % %  for j = 1:size(mat_V,2)
        % %   for k = 1:size(mat_V,3)
        % %    n = n+1;
        % %    mat_V(i,j,k) = n^2;
        % %    vec_i(n) = i;
        % %    vec_j(n) = j;
        % %    vec_k(n) = k;
        % %    vec_V(n) = mat_V(i,j,k);
        % %   end
        % %  end
        % % end
        
        % % vec_n = vec_i+size(mat_V,1)*((vec_j-1)+size(mat_V,2)*(vec_k-1));
        % % mat_V_tilde = nan(size(mat_V));
        % % mat_V_tilde(vec_n) = vec_V;
        
        % % k = 3;figure;subplot(1,2,1);imagesc(squeeze(mat_V(:,:,k)));colorbar;subplot(1,2,2);imagesc(squeeze(mat_V_tilde(:,:,k)));colorbar;
        % % for k = 1:size(mat_V,3),figure;imagesc(squeeze(mat_V(:,:,k)-mat_V_tilde(:,:,k)));colorbar;end;
        
        %% ** CALCUL DE LA MISE A JOUR DES PROPRIETES MECANIQUES: EN (B,G) AVEC "B = LAMBDA + 2/3*MU" ET " G = MU" **
        
        disp('        mise a jour des propri�tes mecaniques');
        t_ini = cputime;
        
        if ( strcmp(type_identification,'FEMU_DFC') == 1 )
            
        elseif ( strcmp(type_identification,'FEMU_AFC') == 1 )
            
            %% --- Calcul du gradient
            
            mat_gradient_A = zeros(size(d_K_d_p,1),size(d_K_d_p,3));
            W_local = U_global((1:nb_DDL_K));
            U_local = U_global(nb_DDL_K+(1:nb_DDL_K));
            
            for l = 1:size(d_K_d_p,3)
                mat_gradient_A(:,l) = sum(squeeze(d_K_d_p(:,:,l)).*(ones(size(d_K_d_p,1),1)*(U_local.*W_local).'),2);
            end
            
            %% --- Minimisation de la fonction cout en (u-u_m) par rapport aux proprietes materielles
            
            %   => algorithme du gradient conjugue
            mat_d = -mat_gradient_A;
            
            %%%%%%%%%%%%%%%%%
            %%% A CONTINUER ...
            %%%%%%%%%%%%%%%%%
            
        elseif( ( strcmp(type_identification,'MERC') == 1 ) )
            
            %d_K_d_p
	    % loop on all the gauss points
            %vec_n_noeuds_pha_sub_zone
            % calculer sigma ...
            
            %% --- Calcul de la contrainte : "sigma = C:epsilon[u] + P:epsilon[w]" et des integrales (HERE)

            % loop on the U/W elements of the subzone
            
%            for nn_elem_sig = 1:length(liste_elem_sig_sub_zone)
%            
%                elem_sig = liste_elem_sig_sub_zone{nn_elem_sig};
%            
%                vec_n_noeuds_sig = elem_sig.vec_n_noeuds;
%
%           %% --- I RAN THIS 
%     
%                mat_pos_Gauss_point = elem_sig.mat_pos_pha_G_K;
%                nb_Gauss_point = size(mat_pos_Gauss_point,2);
%
%	    % -- Loop over all Gaussian points
% 
%                for n_Gauss = 1:nb_Gauss_point
%                     a(:,n_Gauss,:) = liste_elem_ref{elem_sig.n_elem_ref}.f_Nf( elem_sig.mat_pos_pha_G_K(1,n_Gauss), elem_sig.mat_pos_pha_G_K(2,n_Gauss), elem_sig.mat_pos_pha_G_K(3,n_Gauss) );
%                end
%           
%           %% --- I RAN THIS
%     
%            end

%tic;
%aux = cellfun(@(liste_elem_sig) liste_elem_sig.vec_n_noeuds, liste_elem_sig, 'UniformOutput', false);
%  
%test = reshape(cell2mat(aux),length(liste_elem_ref{4}.pos_noeuds),length(liste_elem_sig));
%
%[a_vec_U_local, a_vec_conjugue_U_local, a_vec_W_local] = deal(nan(3*length(liste_elem_ref{4}.pos_noeuds),length(liste_elem_sig)));
%
%a = struct('a_x',repmat(Ux,size(Ux,2), size(test,2)),'a_y',repmat(Uy,size(Uy,2), size(test,2)),'a_z',repmat(Uz,size(Uz,2), size(test,2)));
%
%a_vec_U_local(1:3:end) = a.a_x(test);
%a_vec_U_local(2:3:end) = a.a_y(test);
%a_vec_U_local(3:3:end) = a.a_z(test);
%
%a_vec_conjugue_U_local(1:3:end) = a.a_x(test);
%a_vec_conjugue_U_local(2:3:end) = a.a_y(test);
%a_vec_conjugue_U_local(3:3:end) = a.a_z(test);
%
%a_vec_W_local(1:3:end) = a.a_x(test);
%a_vec_W_local(2:3:end) = a.a_y(test);
%a_vec_W_local(3:3:end) = a.a_z(test);
%toc;
%
% tic;
%foig.vec_n_noeuds) n_sig = 1:length(liste_elem_sig
% elem_sig = liste_elem_sig{n_sig};
% elem_ref_sig = liste_elem_ref{elem_sig.n_elem_ref};
% vec_U_local = nan(3*length(elem_sig.vec_n_noeuds),1);
% vec_U_local(1:3:end) = Ux(elem_sig.vec_n_noeuds).';
% vec_U_local(2:3:end) = Uy(elem_sig.vec_n_noeuds).';
% vec_U_local(3:3:end) = Uz(elem_sig.vec_n_noeuds).';
% vec_conjugue_U_local = nan(3*length(elem_sig.vec_n_noeuds),1);
% vec_conjugue_U_local(1:3:end) = conj(Ux(elem_sig.vec_n_noeuds).');
% vec_conjugue_U_local(2:3:end) = conj(Uy(elem_sig.vec_n_noeuds).');
% vec_conjugue_U_local(3:3:end) = conj(Uz(elem_sig.vec_n_noeuds).');
% vec_W_local = nan(3*length(elem_sig.vec_n_noeuds),1);
% vec_W_local(1:3:end) = Wx(elem_sig.vec_n_noeuds).';
% vec_W_local(2:3:end) = Wy(elem_sig.vec_n_noeuds).';
% _U_local(1:3:end) = Ux(elem_sig.vec_n_noeuds).';(3:3:end) = Wz(elem_sig.vec_n_noeuds).';
%end
% toc;
%
%a1;

vec_x_G = zeros(1,length(liste_elem_sig_sub_zone)*27);
vec_y_G = zeros(1,length(liste_elem_sig_sub_zone)*27);
vec_z_G = zeros(1,length(liste_elem_sig_sub_zone)*27);
vec_tr_epsilon_U_G = nan(1,length(liste_elem_sig_sub_zone)*27);
vec_tr_sigma_G = zeros(1,length(liste_elem_sig_sub_zone)*27);
vec_deviateur_epsilon_U_deviateur_epsilon_conjugue_U_G = nan(1,length(liste_elem_sig_sub_zone)*27);
vec_deviateur_sigma_deviateur_epsilon_conjugue_U_G = nan(1,length(liste_elem_sig_sub_zone)*27);
mat_coef_mat = zeros(length(liste_elem_sig_sub_zone)*27,2*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local));
vec_second_membre_mat = zeros(length(liste_elem_sig_sub_zone)*27,1);
vec_n_elem_pha_systeme = zeros(1,length(liste_elem_sig_sub_zone)*27);

%A = zeros(length(liste_elem_sig_sub_zone)*27,...
%    max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local));
%[b, st_str, st_pha] = deal(zeros(1,length(liste_elem_sig_sub_zone)*27));

% if n_iter_LDC == 1
%     vector_solucion = zeros(2,1);
%     vector_solucion(:,n_iter_LDC) = [vec_param_initialisation(1,1); vec_param_initialisation(1,2)];
% end

n_G_local = 0;
for n_sig = 1:length(liste_elem_sig_sub_zone)
    
    elem_sig = liste_elem_sig_sub_zone{n_sig};
    elem_ref_sig = liste_elem_ref{elem_sig.n_elem_ref};
    vec_U_local = nan(3*length(elem_sig.vec_n_noeuds),1);
    vec_U_local(1:3:end) = Ux(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_U_local(2:3:end) = Uy(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_U_local(3:3:end) = Uz(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_conjugue_U_local = conj(vec_U_local);
    vec_W_local = nan(3*length(elem_sig.vec_n_noeuds),1);
    vec_W_local(1:3:end) = Wx(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_W_local(2:3:end) = Wy(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_W_local(3:3:end) = Wz(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_poids_Gauss = liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.poids_Gauss;
    
    
    %
    %
    %    mat_d_f_Nf_sig = (elem_ref_sig.d_f_Nf(elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(:,1),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(:,2),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(:,3)));
    %
    %    n_pha = vec_n_pha_G_K(nn_pha_local);
    %    elem_pha = liste_elem_pha{n_pha};
    %    vec_n_noeuds_pha = elem_pha.vec_n_noeuds;
    %    elem_ref_pha = liste_elem_ref{elem_pha.n_elem_ref};
    %    vec_f_Nf_pha = zeros(size(elem_sig.mat_pos_pha_G_K,2),1,);
    %    vec_n_pha_G_K = unique(elem_sig.vec_n_pha_G_K);
    %    for nn_pha_local = 1:length(vec_n_pha_G_K)
    %        n_pha = vec_n_pha_G_K(nn_pha_local);
    %        elem_pha = liste_elem_pha{n_pha};
    %        vec_n_noeuds_pha = elem_pha.vec_n_noeuds;
    %        elem_ref_pha = liste_elem_ref{elem_pha.n_elem_ref};
    %        nn_noeud_sig_pha_courante = find ( elem_sig.vec_n_pha_G_K == vec_n_pha_G_K(nn_pha_local) );
    %        vec_f_Nf_pha(nn_noeud_sig_pha_courante,1,:) = elem_ref_pha.f_Nf(elem_sig.mat_pos_pha_G_K(1,nn_noeud_sig_pha_courante),elem_sig.mat_pos_pha_G_K(2,nn_noeud_sig_pha_courante),elem_sig.mat_pos_pha_G_K(3,nn_noeud_sig_pha_courante));
    %    end
    %
    
    for n_G = 1:size(elem_sig.mat_pos_pha_G_K,2)
        
        n_pha = elem_sig.vec_n_pha_G_K(n_G);
        vec_pos_G_ref = elem_sig.mat_pos_pha_G_K(:,n_G);
        elem_pha = liste_elem_pha{n_pha};
        vec_n_noeuds_pha = elem_pha.vec_n_noeuds;
        elem_ref_pha = liste_elem_ref{elem_pha.n_elem_ref};
        vec_f_Nf_pha = squeeze(elem_ref_pha.f_Nf(vec_pos_G_ref(1),vec_pos_G_ref(2),vec_pos_G_ref(3)))';
        % % calcul de la deformation au point de Gauss de l'element de contrainte
        mat_d_f_Nf_sig = squeeze(elem_ref_sig.d_f_Nf(elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,1),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,2),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,3)))';
        Jaco = mat_d_f_Nf_sig*mat_pos_maillage_sig(elem_sig.vec_n_noeuds,:); % matrice 3x3. Produit du gradient fonction de forme associ� et coordonn�es des noeuds
        J = det(Jaco); % d�terminant
        % % matrice du gradient des fonctions de formes dans la configuration reelle
        % % permettant de calculer le tenseur de deformation en notation de "Voigt modifiee" : (exx,eyy,ezz,2*exy,2*exz,2*eyz)
        [Be] = calcul_Be(mat_d_f_Nf_sig,Jaco,nb_DDL_par_noeud);
        vec_epsilon_U_Voigt = Be*vec_U_local;
        vec_epsilon_conjugue_U_Voigt = Be*vec_conjugue_U_local;
        vec_epsilon_W_Voigt = Be*vec_W_local;
        % % calcul des fonctions de forme sur les phases au point de Gauss de l'element de contrainte
        lambda_G = champ_proprietes.lambda(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
        mu_G = champ_proprietes.mu(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
        data_LDC = struct('lambda',lambda_G,'mu',mu_G);
        A_visco = LDC(data_LDC);
        lambda_elastique_G = champ_proprietes_elastiques.lambda(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
        mu_elastique_G = champ_proprietes_elastiques.mu(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
        data_LDC_elastique = struct('lambda',lambda_elastique_G,'mu',mu_elastique_G);
        A_elastique = LDC(data_LDC_elastique);
        % % calcul de sigma : sigma = C:epsilon[u] + P:epsilon[w]
        vec_sigma_Voigt = A_visco*vec_epsilon_U_Voigt+A_elastique*vec_epsilon_W_Voigt;
        % % calcul des quantites necessaires a la mise a jour des proprietes des phases
        deviateur_epsilon_U = calcul_deviateur(vec_epsilon_U_Voigt./[1 1 1 2 2 2]');
        deviateur_epsilon_conjugue_U = calcul_deviateur(vec_epsilon_conjugue_U_Voigt./[1 1 1 2 2 2]');
        deviateur_sigma = calcul_deviateur(vec_sigma_Voigt);
        tr_epsilon_U = vec_epsilon_U_Voigt(1)+vec_epsilon_U_Voigt(2)+vec_epsilon_U_Voigt(3);
        tr_sigma = vec_sigma_Voigt(1)+vec_sigma_Voigt(2)+vec_sigma_Voigt(3);
        deviateur_epsilon_U_deviateur_epsilon_conjugue_U = sum(sum(deviateur_epsilon_U.*deviateur_epsilon_conjugue_U));
        deviateur_sigma_deviateur_epsilon_conjugue_U = sum(sum(deviateur_sigma.*deviateur_epsilon_conjugue_U));
        
        % remplissage des vecteurs pour tester
        
        n_G_local = n_G_local+1;
        vec_x_G(n_G_local) = (min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,1))+(max(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,1))-min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,1)))*(liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,1)+1)/2);
        vec_y_G(n_G_local) = (min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,2))+(max(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,2))-min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,2)))*(liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,2)+1)/2);
        vec_z_G(n_G_local) = (min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,3))+(max(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,3))-min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,3)))*(liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,3)+1)/2);
        vec_tr_epsilon_U_G(n_G_local) = tr_epsilon_U;
        vec_tr_sigma_G(n_G_local) = tr_sigma;
        vec_deviateur_epsilon_U_deviateur_epsilon_conjugue_U_G(n_G_local) = deviateur_epsilon_U_deviateur_epsilon_conjugue_U;
        vec_deviateur_sigma_deviateur_epsilon_conjugue_U_G(n_G_local) = deviateur_sigma_deviateur_epsilon_conjugue_U; % remplissage du systeme lineaire :
        % remplissage du systeme lineaire :
        
        % % methode (i) ecriture des equations en (lambda,mu)
        
        % % equation sur lambda
        
        mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*3*tr_epsilon_U*vec_f_Nf_pha; % lambda
        mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*2*tr_epsilon_U*vec_f_Nf_pha; % mu
        
        vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = vec_poids_Gauss(n_G)*J*tr_sigma;
        
        vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_pha;
        
        % % equation sur mu
        
        mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2,max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U*vec_f_Nf_pha; % mu
        
        vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = vec_poids_Gauss(n_G)*J*deviateur_sigma_deviateur_epsilon_conjugue_U;
        
        vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_pha;
        
        %  % methode (ii) ecriture des equations en (B,G)
        %  % equation sur B
        %     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,vec_n_noeuds_pha) = vec_poids_Gauss(n_G)*J*3*tr_epsilon_U*vec_f_Nf_pha; % B
        %     vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = vec_poids_Gauss(n_G)*J*tr_sigma;
        %     vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_pha;
        %  % equation sur G
        %     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2,size(mat_pos_pha,1)+vec_n_noeuds_pha) = vec_poids_Gauss(n_G)*J*2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U*vec_f_Nf_pha; % G
        %     vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = vec_poids_Gauss(n_G)*J*deviateur_sigma_deviateur_epsilon_conjugue_U;
        %     vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_pha;
        
        %  % third method: solve the linear equation A * lambda = b
        
%         A(elem_sig_ref.liste_parametres_integration{n_integration_K}...
%             .nb_Gauss*(n_sig-1)+n_G,...
%             vec_correspondance_n_noeud_pha_global_n_noeud_pha_local...
%             (vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*...
%             3*tr_epsilon_U*vec_f_Nf_pha;
%         
%         b(elem_sig_ref.liste_parametres_integration{n_integration_K}...
%             .nb_Gauss*(n_sig-1)+n_G) = vec_poids_Gauss(n_G)*J*...
%             tr_sigma - vec_poids_Gauss(n_G)*J*...
%             ( (deviateur_sigma_deviateur_epsilon_conjugue_U*J)...
%             * (tr_epsilon_U*J*vec_f_Nf_pha) )...
%             ./ deviateur_epsilon_U_deviateur_epsilon_conjugue_U;
        
        st_str(elem_sig_ref.liste_parametres_integration{n_integration_K}...
            .nb_Gauss*(n_sig-1)+n_G) = n_sig;
        
        st_pha(elem_sig_ref.liste_parametres_integration{n_integration_K}...
            .nb_Gauss*(n_sig-1)+n_G) = n_pha;
        
        % fourth method
        
        A_mu = mat_coef_mat(:,size(mat_coef_mat,2)/2+1:...
            size(mat_coef_mat,2));
        
        b_mu = vec_second_membre_mat;
        
    end
end

% third approach

% mat_coef_mat_square = (mat_coef_mat.'*mat_coef_mat);
% vec_second_membre_mat_square = (mat_coef_mat.'*vec_second_membre_mat);
% vec_max_square = max(abs(mat_coef_mat_square),[],2);
% %vec_max_square = ones(size(mat_coef_mat_square,1),1);
% mat_coef_mat_square = mat_coef_mat_square./(vec_max_square*ones(1,size(mat_coef_mat_square,2)));
% vec_second_membre_mat_square = vec_second_membre_mat_square./vec_max_square;
% vec_sol_square = mat_coef_mat_square\vec_second_membre_mat_square;
% 
% vec_sol_square = sign_change(vec_sol_square,...
%      struct_param_comportement_a_identifier.vec_param_initialisation,0.6);


% if nnz(sum(real(vec_sol_square)<0))~=0
%     vec_sol_square(real(vec_sol_square)<0) = vec_sol_square(real(vec_sol_square)<0) + 2*real(vec_sol_square(real(vec_sol_square)<0));
% end
% 
% if nnz(sum(imag(vec_sol_square)<0))~=0
%     vec_sol_square(imag(vec_sol_square)<0) = conj(vec_sol_square(imag(vec_sol_square)<0));
% end

% third approach (phase elements)

M_p = zeros(2*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local));
v_p = zeros(2*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local),1);

aux = unique(vec_n_elem_pha_systeme);
n_eq = zeros(length(aux),nb_iter_LDC_max);

for i=1:size(mat_coef_mat,2)/2
    
    A_aux = mat_coef_mat(vec_n_elem_pha_systeme(:) == aux(i),:);
    %A(A_aux==0) = NaN;
    
    n_eq(i,n_iter_LDC) = length(A_aux);
    
	v_aux = vec_second_membre_mat(vec_n_elem_pha_systeme(:) == aux(i));
    
    M_p(2*(i-1)+1,:) = mean(A_aux(1:2:end,:));
    M_p(2*i,:) = mean(A_aux(2:2:end,:));
    
    v_p(2*(i-1)+1) = mean(v_aux(1:2:end));
    v_p(2*i) = mean(v_aux(2:2:end));
    
    M_p(isnan(M_p)) = 0;
    
end

M_p = M_p(:,size(M_p,2)/2+1:size(M_p));

M_ps = (M_p.'*M_p);
v_ps = (M_p.'*v_p);
vmax = max(abs(M_ps),[],2);
%vec_max_square = ones(size(mat_coef_mat_square,1),1);
M_ps = M_ps./(vmax*ones(1,size(M_ps,2)));
v_ps = v_ps./vmax;
vss = M_ps\v_ps;

vss = sign_change(vss,...
     struct_param_comportement_a_identifier.vec_param_initialisation,0.6);

% third approach related to the third method (A * lambda = b)

% M_p = zeros(max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local));
% v_p = zeros(max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local),1);
% 
% aux = unique(st_pha);
% 
% for i=1:size(A,2)
%     A_aux = A(st_pha(:) == aux(i),:);
%     v_aux = b(st_pha(:) == aux(i));
%     M_p(i,:) = mean(A_aux);
%     v_p(i) = mean(v_aux);
% end
% 
% 
% M_ps = (M_p.'*M_p);
% v_ps = (M_p.'*v_p);
% vmax = max(abs(M_ps),[],2);
% %vec_max_square = ones(size(mat_coef_mat_square,1),1);
% M_ps = M_ps./(vmax*ones(1,size(M_ps,2)));
% v_ps = v_ps./vmax;
% vss = M_ps\v_ps;
% 
%  vss = sign_change(vss,...
%      struct_param_comportement_a_identifier.vec_param_initialisation,0.6);

% fourth approach

% A_mu(A_mu==0) = NaN;
% 
% M_p = zeros(max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local));
% v_p = zeros(max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local),1);
% 
% aux = unique(st_pha);
% 
% for i=1:size(A,2)
%     A_aux = A_mu(st_pha(:) == aux(i),:);
%     v_aux = b_mu(st_pha(:) == aux(i));
%     M_p(i,:) = mean(A_aux,'omitnan');
%     v_p(i) = mean(v_aux,'omitnan');
% end
% 
% M_p(isnan(M_p)) = 0;
% 
% 
% M_ps = (M_p.'*M_p);
% v_ps = (M_p.'*v_p);
% vmax = max(abs(M_ps),[],2);
% %vec_max_square = ones(size(mat_coef_mat_square,1),1);
% M_ps = M_ps./(vmax*ones(1,size(M_ps,2)));
% v_ps = v_ps./vmax;
% vss = M_ps\v_ps;
% 
%  vss = sign_change(vss,...
%      struct_param_comportement_a_identifier.vec_param_initialisation,0.6);


% if nnz(sum(real(vss)<0))~=0
%     vss(real(vss)<0) =...
%         vss(real(vss)<0) +...
%         2*real(vss(real(vss)<0));
% end
% 
% if nnz(sum(imag(vss)<0))~=0
%     vss(imag(vss)<0) =...
%         conj(vss(imag(vss)<0));
% end

% if nnz(sum(real(vss)<0))~=0
%     vss(real(vss)<0) = vss(real(vss)<0) + 2*real(vss(real(vss)<0));
% end
% 
% if nnz(sum(imag(vss)<0))~=0
%     vss(imag(vss)<0) = conj(vss(imag(vss)<0));
% end



% --------

% mat2 = (mat_red.'*mat_red);
% vsm2 = (mat_red.'*vsm_red);
% mat2_max = max(abs(mat2),[],2);
% %vec_max_square = ones(size(mat_coef_mat_square,1),1);
% mat2 = mat2./(mat2_max*ones(1,size(mat2,2)));
% vsm2 = vsm2./mat2_max;
% vec2 = mat2\vsm2;
% 
% 
% if nnz(sum(real(vec2)<0))~=0
%     vec2(real(vec2)<0) = vec2(real(vec2)<0) - 2*real(vec2(real(vec2)<0));
% end
% 
% if nnz(sum(imag(vec2)<0))~=0
%     vec2(imag(vec2)<0) = conj(vec2(imag(vec2)<0));
% end

% vector_solucion(:,n_+1) = [ mean(vec_sol_square(1:length(vec_sol_square)/2,:)); mean(vec_sol_square((length(vec_sol_square)/2):end,:)) ];
% 
% if ( norm(vector_solucion(:,n_iter_LDC+1)) < tolerance_LDC*norm(vector_solucion(:,n_iter_LDC)) )
%     test_convergence_LDC = true;
% end
           
            % Define A_visco, A_elastique, vec_epsilon_U_Voigt
            
            %% --- Calcul des quantites necessaires a la mise a jour des proprietes des phases
            
%             deviateur_epsilon_U = calcul_deviateur(vec_epsilon_U_Voigt./[1 1 1 2 2 2]');
%             deviateur_epsilon_conjugue_U = calcul_deviateur(vec_epsilon_conjugue_U_Voigt./[1 1 1 2 2 2]');
%             deviateur_sigma = calcul_deviateur(vec_sigma_Voigt);
%             
%             tr_epsilon_U = vec_epsilon_U_Voigt(1)+vec_epsilon_U_Voigt(2)+vec_epsilon_U_Voigt(3);
%             tr_sigma = vec_sigma_Voigt(1)+vec_sigma_Voigt(2)+vec_sigma_Voigt(3);
%             
%             deviateur_epsilon_U_deviateur_epsilon_conjugue_U = sum(sum(deviateur_epsilon_U.*deviateur_epsilon_conjugue_U));
%             deviateur_sigma_deviateur_epsilon_conjugue_U = sum(sum(deviateur_sigma.*deviateur_epsilon_conjugue_U));
            
        end
        
        t_fin = cputime;
        disp(['        ' num2str(t_fin-t_ini) ' s']);
        
        vec_test_n_elem_sig_centre_sub_zone(n_sig_sub_zone) = true;
        
        if ( prod(vec_test_n_elem_sig_centre_sub_zone) == 1 )
            test_fin_sub_zones = true;
        end
        
    end % FINAL DEL WHILE DE LA SUBZONA

    %% --- Mise a jour des proprietes
    
    liste_proprietes_iterations{n_iter_LDC+1} = nan(3,60);
    
    for i=1:size(vss,1)
        liste_proprietes_iterations{n_iter_LDC+1}(1,aux(i)) = 0;
        liste_proprietes_iterations{n_iter_LDC+1}(2,aux(i)) = vss(i);
        liste_proprietes_iterations{n_iter_LDC+1}(3,aux(i)) = 0;
    end
    
%     gcf = figure;
%     plot(real(liste_proprietes_iterations{n_iter_LDC+1}(2,:)),'b*');
%     hold on;
%     plot(imag(liste_proprietes_iterations{n_iter_LDC+1}(2,:)),'r*');
%     grid;
%     clc;
    
    % disp(vss);
    % pause;
    
    
%     liste_proprietes_iterations{n_iter_LDC+1} = champ_proprietes;
%     champ_proprietes_prec = champ_proprietes;
%     
%     % champ_proprietes_elastiques.lambda = abs(champ_proprietes.lambda);
%     % champ_proprietes_elastiques.mu = abs(champ_proprietes.mu);
%     
%     
     %% --- Test de la convergence
     
     if n_iter_LDC == 1
        prev_val = ones(size(aux))*liste_proprietes_iterations...
            {n_iter_LDC}(2,1);
        act_val = liste_proprietes_iterations{n_iter_LDC+1}...
         (2,~isnan(real(liste_proprietes_iterations{n_iter_LDC+1}(2,:))));...
     else
        prev_val = liste_proprietes_iterations{n_iter_LDC}...
            (2,~isnan(real(liste_proprietes_iterations{n_iter_LDC+1}(2,:))));...
        act_val = liste_proprietes_iterations{n_iter_LDC+1}...
         (2,~isnan(real(liste_proprietes_iterations{n_iter_LDC+1}(2,:))));...
     end
     
     dev = act_val - prev_val;
     
     if ( norm(dev) < tolerance_LDC*norm(prev_val) )
         test_convergence_LDC = true;
     end
     
%     vec_difference_proprietes = [champ_proprietes.lambda champ_proprietes.mu]-[champ_proprietes_prec.lambda champ_proprietes_prec.mu];
%     
%     if ( norm(vec_difference_proprietes) < tolerance_LDC*norm([champ_proprietes_prec.lambda champ_proprietes_prec.mu]) )
%         test_convergence_LDC = true;
%     end
    
%     disp(['     norme relative de la correction = ' num2str(norm(vec_difference_proprietes)/norm([champ_proprietes_prec.lambda champ_proprietes_prec.mu]))]);
%     disp(' ');
     
     clc;
    

end

plotting;

%% To be identified....

% 
% 
% % calcul de la mise a jour des proprietes mecaniques : en (B,G) avec "B = lambda+2/3*mu" et "G = mu"
% % formules valables en chaque point du domaine => a integrer
% % B_kp1 = tr(sigma)/(3*tr(epsilon[u]) => equation en module de compressibilite"
% % G_kp1 = (sigma_dev:epsilon_dev[u])/(2*epsilon_dev[u]:epsilon_dev[conj(u)])  => equation en module de cisaillement"
% % puis :
% % mu_kp1 = G_kp1
% % lambda_kp1 = B_kp1-2/3*G_kp1
%  disp('     mise a jour des propri�tes mecaniques');
%  t_ini = cputime;
%  mat_coef_mat = zeros(2*length(liste_elem_sig)*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss,size(mat_pos_pha,1)*2);
%  vec_second_membre_mat = zeros(2*length(liste_elem_sig)*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss,1);
%  vec_n_elem_pha_systeme = nan(2*length(liste_elem_sig)*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss,1);
%  vec_x_G = nan(1,length(liste_elem_sig)*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss);
%  vec_y_G = nan(1,length(liste_elem_sig)*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss);
%  vec_z_G = nan(1,length(liste_elem_sig)*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss);
%  vec_tr_epsilon_U_G = nan(1,length(liste_elem_sig)*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss);
%  vec_tr_sigma_G = nan(1,length(liste_elem_sig)*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss);
%  vec_deviateur_epsilon_U_deviateur_epsilon_conjugue_U_G = nan(1,length(liste_elem_sig)*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss);
%  vec_deviateur_sigma_deviateur_epsilon_conjugue_U_G = nan(1,length(liste_elem_sig)*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss);
%  n_G_local = 0;
%  for n_sig = 1:length(liste_elem_sig)
%   elem_sig = liste_elem_sig{n_sig};
%   elem_ref_sig = liste_elem_ref{elem_sig.n_elem_ref};
%   vec_U_local = nan(3*length(elem_sig.vec_n_noeuds),1);
%   vec_U_local(1:3:end) = Ux(elem_sig.vec_n_noeuds).';
%   vec_U_local(2:3:end) = Uy(elem_sig.vec_n_noeuds).';
%   vec_U_local(3:3:end) = Uz(elem_sig.vec_n_noeuds).';
%   vec_conjugue_U_local = nan(3*length(elem_sig.vec_n_noeuds),1);
%   vec_conjugue_U_local(1:3:end) = conj(Ux(elem_sig.vec_n_noeuds).');
%   vec_conjugue_U_local(2:3:end) = conj(Uy(elem_sig.vec_n_noeuds).');
%   vec_conjugue_U_local(3:3:end) = conj(Uz(elem_sig.vec_n_noeuds).');
%   vec_W_local = nan(3*length(elem_sig.vec_n_noeuds),1);
%   vec_W_local(1:3:end) = Wx(elem_sig.vec_n_noeuds).';
%   vec_W_local(2:3:end) = Wy(elem_sig.vec_n_noeuds).';
%   vec_W_local(3:3:end) = Wz(elem_sig.vec_n_noeuds).';
%   vec_poids_Gauss = liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.poids_Gauss;
%   for n_G = 1:size(elem_sig.mat_pos_pha_G_K,2)
%    n_pha = elem_sig.vec_n_pha_G_K(n_G);
%    vec_pos_G_ref = elem_sig.mat_pos_pha_G_K(:,n_G);
%    elem_pha = liste_elem_pha{n_pha};
%    vec_n_noeuds_pha = elem_pha.vec_n_noeuds;
%    elem_ref_pha = liste_elem_ref{elem_pha.n_elem_ref};
%    vec_f_Nf_pha = squeeze(elem_ref_pha.f_Nf(vec_pos_G_ref(1),vec_pos_G_ref(2),vec_pos_G_ref(3)))';
% % calcul de la deformation au point de Gauss de l'element de contrainte
%    mat_d_f_Nf_sig = squeeze(elem_ref_sig.d_f_Nf(elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,1),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,2),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,3)))';
%    Jaco = mat_d_f_Nf_sig*mat_pos_maillage_sig(elem_sig.vec_n_noeuds,:); % matrice 3x3. Produit du gradient fonction de forme associ� et coordonn�es des noeuds
%    J = det(Jaco); % d�terminant
% % matrice du gradient des fonctions de formes dans la configuration reelle
% % permettant de calculer le tenseur de deformation en notation de "Voigt modifiee" : (exx,eyy,ezz,2*exy,2*exz,2*eyz)
%    [Be] = calcul_Be(mat_d_f_Nf_sig,Jaco,nb_DDL_par_noeud);
%    vec_epsilon_U_Voigt = Be*vec_U_local;
%    vec_epsilon_conjugue_U_Voigt = Be*vec_conjugue_U_local;
%    vec_epsilon_W_Voigt = Be*vec_W_local;
% % calcul des fonctions de forme sur les phases au point de Gauss de l'element de contrainte
%    lambda_G = champ_proprietes.lambda(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
%    mu_G = champ_proprietes.mu(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
%    data_LDC = struct('lambda',lambda_G,'mu',mu_G);
%    A_visco = LDC(data_LDC);
%    lambda_elastique_G = champ_proprietes_elastiques.lambda(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
%    mu_elastique_G = champ_proprietes_elastiques.mu(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
%    data_LDC_elastique = struct('lambda',lambda_elastique_G,'mu',mu_elastique_G);
%    A_elastique = LDC(data_LDC_elastique);
% % calcul de sigma : sigma = C:epsilon[u] + P:epsilon[w]
%    vec_sigma_Voigt = A_visco*vec_epsilon_U_Voigt+A_elastique*vec_epsilon_W_Voigt;
% % calcul des quantites necessaires a la mise a jour des proprietes des phases
%    deviateur_epsilon_U = calcul_deviateur(vec_epsilon_U_Voigt./[1 1 1 2 2 2]');
%    deviateur_epsilon_conjugue_U = calcul_deviateur(vec_epsilon_conjugue_U_Voigt./[1 1 1 2 2 2]');
%    deviateur_sigma = calcul_deviateur(vec_sigma_Voigt);
%    tr_epsilon_U = vec_epsilon_U_Voigt(1)+vec_epsilon_U_Voigt(2)+vec_epsilon_U_Voigt(3);
%    tr_sigma = vec_sigma_Voigt(1)+vec_sigma_Voigt(2)+vec_sigma_Voigt(3);
%    deviateur_epsilon_U_deviateur_epsilon_conjugue_U = sum(sum(deviateur_epsilon_U.*deviateur_epsilon_conjugue_U));
%    deviateur_sigma_deviateur_epsilon_conjugue_U = sum(sum(deviateur_sigma.*deviateur_epsilon_conjugue_U));
% % remplissage des vecteurs pour tester
%    n_G_local = n_G_local+1;
%    vec_x_G(n_G_local) = (min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,1))+(max(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,1))-min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,1)))*(liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,1)+1)/2);
%    vec_y_G(n_G_local) = (min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,2))+(max(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,2))-min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,2)))*(liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,2)+1)/2);
%    vec_z_G(n_G_local) = (min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,3))+(max(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,3))-min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,3)))*(liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,3)+1)/2);
%    vec_tr_epsilon_U_G(n_G_local) = tr_epsilon_U;
%    vec_tr_sigma_G(n_G_local) = tr_sigma;
%    vec_deviateur_epsilon_U_deviateur_epsilon_conjugue_U_G(n_G_local) = deviateur_epsilon_U_deviateur_epsilon_conjugue_U;
%    vec_deviateur_sigma_deviateur_epsilon_conjugue_U_G(n_G_local) = deviateur_sigma_deviateur_epsilon_conjugue_U;
% % remplissage du systeme lineaire : 
% 
% % % methode (i) ecriture des equations en (lambda,mu)
% % % equation sur B
% %    mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,vec_n_noeuds_pha) = vec_poids_Gauss(n_G)*J*3*tr_epsilon_U*vec_f_Nf_pha; % lambda
% %    mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,size(mat_pos_pha,1)+vec_n_noeuds_pha) = vec_poids_Gauss(n_G)*J*2*tr_epsilon_U*vec_f_Nf_pha; % mu
% %    vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = vec_poids_Gauss(n_G)*J*tr_sigma;
% %    vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_pha;
% % % equation sur G
% %    mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2,size(mat_pos_pha,1)+vec_n_noeuds_pha) = vec_poids_Gauss(n_G)*J*2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U*vec_f_Nf_pha; % mu
% %    vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = vec_poids_Gauss(n_G)*J*deviateur_sigma_deviateur_epsilon_conjugue_U;
% %    vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_pha;
% 
% % methode (ii) ecriture des equations en (B,G)
% % equation sur B
%    mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,vec_n_noeuds_pha) = vec_poids_Gauss(n_G)*J*3*tr_epsilon_U*vec_f_Nf_pha; % B
%    vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = vec_poids_Gauss(n_G)*J*tr_sigma;
%    vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_pha;
% % equation sur G
%    mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2,size(mat_pos_pha,1)+vec_n_noeuds_pha) = vec_poids_Gauss(n_G)*J*2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U*vec_f_Nf_pha; % G
%    vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = vec_poids_Gauss(n_G)*J*deviateur_sigma_deviateur_epsilon_conjugue_U;
%    vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_pha;
%   end
%  end
% 
% % Traces_grandeurs_MAJ_pha;
% 
% % resolution du systeme => mise a jour des proprietes
%  vec_lambda_mu_prec = [champ_proprietes.lambda champ_proprietes.mu].';
%  if ( strcmp(elem_pha_ref.type_elem,'HEX1') == 1 ) % elements de phase constants
% % 0) resolution des equations en moyenne par phase
%   if ( test_methode_MAJ_pha == 0 )
%    mat_systeme_pha = zeros(2*length(liste_elem_pha),size(mat_coef_mat,2));
%    vec_systeme_pha = zeros(2*length(liste_elem_pha),1);
%    for n_pha = 1:length(liste_elem_pha)
% %    vec_n_ligne_tempo_1 = find ( vec_n_elem_pha_systeme(1:2:end) == n_pha);
%     vec_n_ligne_tempo_1 = 2*(find(vec_n_elem_pha_systeme(1:2:end) == n_pha)-1)+1;
%     if ( ~isempty(vec_n_ligne_tempo_1) )
%      mat_systeme_pha(2*(n_pha-1)+1,:) = sum(mat_coef_mat(vec_n_ligne_tempo_1,:),1);
%      vec_systeme_pha(2*(n_pha-1)+1,:) = sum(vec_second_membre_mat(vec_n_ligne_tempo_1,1),1);
%     end
% %    vec_n_ligne_tempo_2 = find ( vec_n_elem_pha_systeme(2:2:end) == n_pha);
%     vec_n_ligne_tempo_2 = 2*(find(vec_n_elem_pha_systeme(2:2:end) == n_pha)-1)+2;
%     if ( ~isempty(vec_n_ligne_tempo_2) )
%      mat_systeme_pha(2*(n_pha-1)+2,:) = sum(mat_coef_mat(vec_n_ligne_tempo_2,:),1);
%      vec_systeme_pha(2*(n_pha-1)+2,:) = sum(vec_second_membre_mat(vec_n_ligne_tempo_2,1),1);
%     end
%    end
%    mat_systeme_pha_module_compressibilite = mat_systeme_pha(1:2:end,1:size(mat_systeme_pha,2)/2);
%    vec_systeme_pha_module_compressibilite = vec_systeme_pha(1:2:end,1);
%    mat_systeme_pha_module_cisaillement = mat_systeme_pha(2:2:end,size(mat_systeme_pha,2)/2+1:end);
%    vec_systeme_pha_module_cisaillement = vec_systeme_pha(2:2:end,1);
%    vec_norme_systeme_pha_module_compressibilite = max(abs(mat_systeme_pha_module_compressibilite),[],2);
%    mat_systeme_pha_module_compressibilite = mat_systeme_pha_module_compressibilite./(vec_norme_systeme_pha_module_compressibilite*ones(1,length(vec_norme_systeme_pha_module_compressibilite)));
%    vec_systeme_pha_module_compressibilite = vec_systeme_pha_module_compressibilite./vec_norme_systeme_pha_module_compressibilite;
%    vec_norme_systeme_pha_module_cisaillement = max(abs(mat_systeme_pha_module_cisaillement),[],2);
%    mat_systeme_pha_module_cisaillement = mat_systeme_pha_module_cisaillement./(vec_norme_systeme_pha_module_cisaillement*ones(1,length(vec_norme_systeme_pha_module_cisaillement)));
%    vec_systeme_pha_module_cisaillement = vec_systeme_pha_module_cisaillement./vec_norme_systeme_pha_module_cisaillement;
%    vec_B_tempo = mat_systeme_pha_module_compressibilite\vec_systeme_pha_module_compressibilite;
%    vec_G_tempo = mat_systeme_pha_module_cisaillement\vec_systeme_pha_module_cisaillement;
%    vec_lambda_mu_tempo = nan(size(mat_systeme_pha,1),1);
%    if ( test_identification_K )
%     vec_lambda_mu_tempo(1:size(mat_systeme_pha,1)/2) = vec_B_tempo-2/3*vec_G_tempo;
%    else
%     vec_lambda_mu_tempo(1:size(mat_systeme_pha,1)/2) = vec_lambda_mu_prec(1:size(mat_systeme_pha,1)/2);
%    end
%    vec_lambda_mu_tempo(size(mat_systeme_pha,1)/2+1:end) = vec_G_tempo;
%    vec_lambda_mu = vec_lambda_mu_tempo;
% %  1) resolution de toutes les equations au sens des moindres carres
%   elseif ( test_methode_MAJ_pha == 1 )
% %   vec_n_DDL_a_supprimer = [vec_n_noeuds_pha_a_supprimer (vec_n_noeuds_pha_a_supprimer+size(mat_pos_pha,1))];
% %   vec_n_DDL_a_conserver = [vec_n_noeuds_pha_a_conserver (vec_n_noeuds_pha_a_conserver+size(mat_pos_pha,1))];
%    vec_n_DDL_a_supprimer = [];
%    vec_n_DDL_a_conserver = 1:(2*size(mat_pos_pha,1));
%    mat_systeme_pha = mat_coef_mat;
%    vec_systeme_pha = vec_second_membre_mat;
%    if ( ~isempty(vec_n_DDL_a_supprimer) )
%     vec_systeme_pha = vec_systeme_pha-mat_systeme_pha(:,vec_n_DDL_a_supprimer)*vec_lambda_mu_prec(vec_n_DDL_a_supprimer);
%    end
%    mat_systeme_pha = mat_systeme_pha(:,vec_n_DDL_a_conserver);
% % diviser la matrice et le vecteur en deux pour determiner separement B et G
%    mat_coef_mat_compressibilite = mat_systeme_pha(1:2:end,1:size(mat_coef_mat,2)/2);
%    vec_second_membre_mat_compressibilite = vec_systeme_pha(1:2:end,1);
%    mat_coef_mat_cisaillement = mat_systeme_pha(2:2:end,size(mat_coef_mat,2)/2+1:end);
%    vec_second_membre_mat_cisaillement = vec_systeme_pha(2:2:end,1);
%    mat_systeme_pha_module_compressibilite = mat_coef_mat_compressibilite'*mat_coef_mat_compressibilite;
%    vec_systeme_pha_module_compressibilite = mat_coef_mat_compressibilite'*vec_second_membre_mat_compressibilite;
%    mat_systeme_pha_module_cisaillement = mat_coef_mat_cisaillement'*mat_coef_mat_cisaillement;
%    vec_systeme_pha_module_cisaillement = mat_coef_mat_cisaillement'*vec_second_membre_mat_cisaillement;
%    vec_norme_systeme_pha_module_compressibilite = max(abs(mat_systeme_pha_module_compressibilite),[],2);
%    mat_systeme_pha_module_compressibilite = mat_systeme_pha_module_compressibilite./(vec_norme_systeme_pha_module_compressibilite*ones(1,length(vec_norme_systeme_pha_module_compressibilite)));
%    vec_systeme_pha_module_compressibilite = vec_systeme_pha_module_compressibilite./vec_norme_systeme_pha_module_compressibilite;
%    vec_norme_systeme_pha_module_cisaillement = max(abs(mat_systeme_pha_module_cisaillement),[],2);
%    mat_systeme_pha_module_cisaillement = mat_systeme_pha_module_cisaillement./(vec_norme_systeme_pha_module_cisaillement*ones(1,length(vec_norme_systeme_pha_module_cisaillement)));
%    vec_systeme_pha_module_cisaillement = vec_systeme_pha_module_cisaillement./vec_norme_systeme_pha_module_cisaillement;
%    vec_B_tempo = mat_systeme_pha_module_compressibilite\vec_systeme_pha_module_compressibilite;
%    vec_G_tempo = mat_systeme_pha_module_cisaillement\vec_systeme_pha_module_cisaillement;
%    vec_lambda_mu_tempo = nan(size(mat_systeme_pha,1),1);
%    if ( test_identification_K )
%     vec_lambda_mu_tempo(1:size(mat_systeme_pha,1)/2) = vec_B_tempo-2/3*vec_G_tempo;
%    else
%     vec_lambda_mu_tempo(1:size(mat_systeme_pha,1)/2) = vec_lambda_mu_prec(1:size(mat_systeme_pha,1)/2);
%    end
%    vec_lambda_mu_tempo(size(mat_systeme_pha,1)/2+1:end) = vec_G_tempo;
%    vec_lambda_mu = vec_lambda_mu_prec;
%    vec_lambda_mu(vec_n_DDL_a_conserver) = vec_lambda_mu_tempo;
% % 2) resolution des equations en moyenne par phase en imposant les valeurs "sur la frontiere"
%   elseif ( test_methode_MAJ_pha == 2 )
% % 3) resolution des equations en moyenne par phase en imposant des gradients de valeurs nuls "sur la frontiere"
%   elseif ( test_methode_MAJ_pha == 3 )
% % 4) resolution des equations en moyenne par subdivision de phase
%   elseif ( test_methode_MAJ_pha == 4 )
%   end
%  else % elements de phase non constants
% % 1) resolution de toutes les equations au sens des moindres carres
%   if ( test_methode_MAJ_pha == 1 )
% % 2) resolution des equations en moyenne par phase en imposant les valeurs "sur la frontiere"
%   elseif ( test_methode_MAJ_pha == 2 )
% % 3) resolution des equations en moyenne par phase en imposant des gradients de valeurs nuls "sur la frontiere"
%   elseif ( test_methode_MAJ_pha == 3 )
% % 4) resolution des equations en moyenne par subdivision de phase
%   elseif ( test_methode_MAJ_pha == 4 )
%   end
%  end
% 
% % figure;hold on;plot(real(vec_lambda_mu_prec(1:length(liste_elem_pha))),'-r');plot(real(vec_lambda_mu(1:length(liste_elem_pha))),'-b');title('real(lambda) (Pa)');legend('prec','MAJ');
% % figure;hold on;plot(real(vec_lambda_mu_prec(length(liste_elem_pha)+1:end)),'-r');plot(real(vec_lambda_mu(length(liste_elem_pha)+1:end)),'-b');title('real(mu) (Pa)');legend('prec','MAJ');
% % figure;hold on;plot(imag(vec_lambda_mu_prec(1:length(liste_elem_pha))),'-r');plot(imag(vec_lambda_mu(1:length(liste_elem_pha))),'-b');title('imag(lambda) (Pa)');legend('prec','MAJ');
% % figure;hold on;plot(imag(vec_lambda_mu_prec(length(liste_elem_pha)+1:end)),'-r');plot(imag(vec_lambda_mu(length(liste_elem_pha)+1:end)),'-b');title('imag(mu) (Pa)');legend('prec','MAJ');
% 
% % mise a jour des proprietes
%  champ_proprietes.lambda = vec_lambda_mu(1:size(mat_pos_pha,1));
%  champ_proprietes.mu = vec_lambda_mu(size(mat_pos_pha,1)+1:end);
% 
%  t_fin = cputime;
%  disp(['        ' num2str(t_fin-t_ini) ' s']);
% 
% % test de la convergence
%  vec_difference_proprietes = [champ_proprietes.lambda champ_proprietes.mu]-[champ_proprietes_prec.lambda champ_proprietes_prec.mu];
%  if ( norm(vec_difference_proprietes) < tolerance_LDC*norm([champ_proprietes_prec.lambda champ_proprietes_prec.mu]) )
%   test_convergence_LDC = true;
%  end
%  disp(['     norme relative de la correction = ' num2str(norm(vec_difference_proprietes)/norm([champ_proprietes_prec.lambda champ_proprietes_prec.mu]))]);
%  disp(' ');
% 
% % mise a jour des proprietes
%  liste_proprietes_iterations{n_iter_LDC+1} = champ_proprietes;
%  champ_proprietes_prec = champ_proprietes;
% % champ_proprietes_elastiques.lambda = abs(champ_proprietes.lambda);
% % champ_proprietes_elastiques.mu = abs(champ_proprietes.mu);
% end

% vec_mu_reel_convergence = zeros(1,length(liste_proprietes_iterations));
% vec_mu_imag_convergence = zeros(1,length(liste_proprietes_iterations));
% for n_iter_LDC = 1:length(liste_proprietes_iterations)
%  champ_proprietes = liste_proprietes_iterations{n_iter_LDC};
%  vec_mu_reel_convergence(n_iter_LDC) = real(champ_proprietes.mu);
%  vec_mu_imag_convergence(n_iter_LDC) = imag(champ_proprietes.mu);
% end
% figure;hold on;plot(vec_mu_reel_convergence,'-r');plot(vec_mu_imag_convergence,'-b');grid;xlabel('n iter');ylabel('mu (Pa)');legend('reel','imag');

% %%%%%%%%%%%%%% TEST DE LA RESOLUTION SUR LA MATRICE DU PROBLEME EN U 
% 
% %%%%%%%% MAJ de la matrice de raideur (suppression lignes et colonnes des CL connues)
% % [Ks,vec_n_DDL_conserves,vec_n_DDL_supprimes,vec_normalisation_Ks] = maj_K(K,liste_DL_bloque);
% % [Ks,vec_n_DDL_conserves,vec_n_DDL_supprimes,vec_normalisation_Ks] = maj_matrices(K,T,M,liste_DL_bloque);
% %K_tot = T;
% %K_tot = T-omega_mec^2*M;
% K_tot = K;
% %K_tot = K-omega_mec^2*M;
% F_tot = vec_F;
% [Ks,Fs,U_impose,vec_n_DDL_conserves,vec_n_DDL_supprimes] = maj_matrices(K_tot,F_tot,liste_DL_bloque_U);
% % figure;hold on;plot3(mat_pos(1,floor((vec_n_DDL_conserves-1)/3)+1),mat_pos(2,floor((vec_n_DDL_conserves-1)/3)+1),mat_pos(3,floor((vec_n_DDL_conserves-1)/3)+1),'xr');plot3(mat_pos(1,floor((vec_n_DDL_supprimes-1)/3)+1),mat_pos(2,floor((vec_n_DDL_supprimes-1)/3)+1),mat_pos(3,floor((vec_n_DDL_supprimes-1)/3)+1),'ob');grid;
% 
% 
% %%%%%%%%%% R�solution
% disp('RESOLUTION');
% t_ini = cputime;
% Us = Ks\Fs;
% t_fin = cputime;
% disp(['      RESOLUTION DIRECTE : ' num2str(t_fin-t_ini) ' s']);
% %Us = resolution(Ks,Fs,nb_iter_restart_gmres,tolerance_gmres,nb_iter_gmres,U_impose);
% % vec_err = Ks*Us-Fs;
% % figure;hold on;plot(real(Ks*Us),'-r');plot(real(Fs),'-b');legend('real(Ks*Us)','real(Fs)');grid;
% % figure;hold on;plot(real(Ks*Us-Fs),'-r');plot(imag(Ks*Us-Fs),'-b');grid;title('Ks*Us-Fs');legend('real','imag');
% 
% %%%%%% R�ecriture de U (assemblage du vecteur resultat)
% U = CL_assemblage(Us,U_impose,vec_n_DDL_conserves,vec_n_DDL_supprimes);
% Ux = U(1:nb_DDL_par_noeud:end);
% Uy = U(2:nb_DDL_par_noeud:end);
% Uz = U(3:nb_DDL_par_noeud:end);
% 
% figure;hold on;plot(real(Ux),'r');plot(real(Uy),'g');plot(real(Uz),'b');title('real(U)');legend('Ux','Uy','Uz');
% figure;hold on;plot(imag(Ux),'r');plot(imag(Uy),'g');plot(imag(Uz),'b');title('imag(U)');legend('Ux','Uy','Uz');
% 
% %%%% trace champs dsp 3D
% Traces_dsp_ERCM;
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FIN ERCM
% 
% % 
% % 
% % 
% % 
% % if ( test_identification )
% % t_ini = cputime;
% % 
% % %%%%%% Calcul de l'�nergie de d�formation
% % if ( test_nouvelle_phase ) % calcul inutile si on a le meme maillage de phase
% % %  disp('CALCUL ENERGIES DE DEFORMATIONS');
% %  [val_energie_0,val_V_totale] = calcul_energie_totale(type_deplacement,liste_data_LDC_phase,U,liste_elem_sig,liste_elem_pha,liste_elem_ref,mat_pos,nb_DDL_par_noeud);
% %  liste_E_e = cell(1,nb_pha(it_pyr));
% % %  E_2 = zeros(1,nb_pha(it_pyr));
% %  for n_elem_pha = 1:length(liste_data_LDC_phase)
% %   type_phase = liste_data_LDC_phase{n_elem_pha}.type_phase;
% % %   [E_e,E_2(n_elem_pha)] = id_visco_coeffs_deformations (n_elem_pha,U_mes,liste_elem_eps,liste_elem_pha,liste_elem_ref,mat_pos,nb_DDL_par_noeud,type_deplacement,type_phase);
% %   [E_e] = id_visco_coeffs_deformations (n_elem_pha,U_mes,liste_elem_mes,liste_elem_pha,liste_elem_ref,mat_pos,nb_DDL_par_noeud,type_deplacement,type_phase);
% %   liste_E_e{n_elem_pha} = E_e;
% %  end
% %  test_nouvelle_phase = false;
% % end
% % 
% % %%%%%%% Minimisation materielle
% % % disp('CALCUL ENERGIES DE CONTRAINTES');
% % vec_NRJ_def = zeros(1,nb_pha(it_pyr));
% % vec_V_phase = zeros(1,nb_pha(it_pyr));
% % liste_E_s = cell(1,nb_pha(it_pyr));
% % % S_2 = zeros(1,nb_pha(it_pyr));
% % pha_fail = [];
% % % calculs des energies
% % for n_elem_pha = 1:length(liste_data_LDC_phase)
% %  data_LDC = liste_data_LDC_phase{n_elem_pha};
% % %  [E_s,S_2(n_elem_pha),NRJ_def,V_phase] = id_visco_coeffs_contraintes (data_LDC,n_elem_pha,U,liste_elem_sig,liste_elem_pha,liste_elem_ref,mat_pos,nb_DDL_par_noeud,type_deplacement);
% %  [E_s,NRJ_def,V_phase] = id_visco_coeffs_contraintes (data_LDC,n_elem_pha,U,liste_elem_sig,liste_elem_pha,liste_elem_ref,mat_pos,nb_DDL_par_noeud,type_deplacement);
% %  liste_E_s{n_elem_pha} = E_s;
% %  rapport_E_s{it_pyr}{n_iter_LDC}{n_elem_pha} = max(E_s)/min(E_s);
% %  vec_NRJ_def(n_elem_pha) = NRJ_def;
% %  vec_V_phase(n_elem_pha) = V_phase;
% % end
% % NRJ_def_totale = sum(vec_NRJ_def);
% % NRJ_def_moyenne = NRJ_def_totale/val_V_totale; % !! NRJ "volumique"
% % 
% % % IDENTIFICATION
% % % disp(['IDENTIFICATION DES PROPRIETES MATERIELLES #' num2str(n_iter_LDC) ]);
% % for n_elem_pha = 1:length(liste_data_LDC_phase)
% %   data_LDC = liste_data_LDC_phase{n_elem_pha};
% %   if ( vec_NRJ_def(n_elem_pha)/vec_V_phase(n_elem_pha) > seuil_NRJ*NRJ_def_moyenne ) % energie suffisante
% %    E_s = liste_E_s{n_elem_pha}; % coeffs d'energies sur contraintes
% %    E_e = liste_E_e{n_elem_pha}; % coeffs d'energies sur deformations
% %    x0 = attrib_anteID(liste_var); % ./ vec_normalisation (valeur initiale)
% %    % annulation des gradients (x | F_x(x) = 0 )
% %    vec_norme_x = max(abs(x0),eps_norme_x*ones(1,length(x0))); % vecteur de normalisation 
% %    
% %    grad_norme = @(x) id_visco_gradient(x,liste_var,E_e,E_s,vec_norme_x); % gradient normal
% %    
% %    options = optimoptions(@fsolve,'Display','off','TolX',TolX,'TolFun',TolFun,'MaxIterations',MaxIterations); %%%% completer param
% %    [x,fval,exitflag,output] = fsolve(grad_norme,x0./vec_norme_x,options); % def vec_norm_fsolve
% %    
% %    % attributions materielles
% %    [lambda_r_ident,mu_r_ident,mu_i_ident,liste_var] = attrib_postID(liste_var,x,vec_norme_x);
% %    
% %    % ancienne methode (resolution directe formulee en elastique reel)
% % %    mu_r_ident_direct = sqrt(S_2(n_elem_pha)/E_2(n_elem_pha))/2; % resultats bizzares : AD
% %    
% %    % relaxations des modules ; A REVOIR
% %    if ( lambda_r_ident < 0 )
% %     lambda_r_ident = data_LDC.lambda_r+alpha_relaxation_lambda*(lambda_r_ident-data_LDC.lambda_r);
% %    end
% %    if ( mu_r_ident < 0 )
% %     mu_r_ident = data_LDC.mu_r+alpha_relaxation_mu_r*(mu_r_ident-data_LDC.mu_r);
% %    end
% %    if ( mu_i_ident < 0 )
% %     mu_i_ident = data_LDC.mu_i+alpha_relaxation_mu_i*(mu_i_ident-data_LDC.mu_i);
% %    end
% %   else % energie insuffisante
% %    mu_r_ident = data_LDC.mu_r;
% %    mu_i_ident = data_LDC.mu_i;
% %   end
% %   nu_r_ident = lambda_r_ident/(2*(lambda_r_ident+mu_r_ident));
% %   nu_i_ident = lambda_r_ident/(2*(lambda_r_ident+mu_i_ident));
% % %   data_LDC.mu_r = mu_r_ident_direct; % !!! on remplace par le calcul direct ; AD
% %   data_LDC.mu_r = mu_r_ident;
% %   data_LDC.mu_i = mu_i_ident;
% %   data_LDC.lambda_r = lambda_r_ident;
% %   data_LDC.nu_r = nu_r_ident;
% %   data_LDC.nu_i = nu_i_ident;
% %   data_LDC.nb_noeuds_calcul = nb_sig(it_pyr);
% %   liste_data_LDC_phase{n_elem_pha} = data_LDC; % on a la liste des donnees MAJ
% %  end
% % 
% % % test sur la convergence des phases
% % test_convergence_LDC = true;
% % vec_dA_relatif = zeros(1,length(liste_data_LDC_phase));
% % pctg_conv = zeros(1,length(liste_data_LDC_phase)); % pourcentage de phases converges
% % for n_elem_pha = 1:length(liste_data_LDC_phase)
% %  data_LDC_prec = liste_data_LDC_phase_prec{n_elem_pha};
% %  data_LDC = liste_data_LDC_phase{n_elem_pha};
% %  A_prec = LDC(type_deplacement,data_LDC_prec);
% %  A = LDC(type_deplacement,data_LDC);
% %  % crit�re sur la positivit� de A pour validite de ERC
% %  if ( (sum(find(eig(A) <= 0))) >= 1 )
% %   n_i_A = n_i_A + 1;
% %   indice_A_nonpos{n_i_A} = struct('it_pyr',it_pyr,'n_iter_LDC',n_iter_LDC,'n_elem_pha',n_elem_pha);
% %  end
% %  % crit�re sur la variation de la norme de A
% %  d_A = norm(A-A_prec)/norm(A);
% %  vec_dA_relatif(n_elem_pha) = d_A;
% %  vec_conv(n_elem_pha) = tolerance_LDC(it_pyr)/d_A;
% %  test_convergence_LDC = test_convergence_LDC*(d_A <= tolerance_LDC(it_pyr)); 
% % end
% % pctg_conv = 100*length(find(vec_conv>=1))/length(vec_conv);
% % 
% % % sauvegarde des donnees pour PT
% % for n_elem_pha = 1:length(liste_data_LDC_phase)
% %  NRJ_phase{n_elem_pha} = struct('liste_E_e',liste_E_e{n_elem_pha},'liste_E_s',liste_E_s{n_elem_pha},'vec_NRJ_def',vec_NRJ_def(n_elem_pha),'V_phase',vec_V_phase(n_elem_pha)); % liste des NRJ par phases
% %  vec_LDC_eps(liste_elem_pha{n_elem_pha}.vec_n_elem_eps) = n_elem_pha; % attribution du numero de LDC � chaque elements
% % end
% % liste_data_LDC_phase_save{it_pyr+1}{n_iter_LDC} = liste_data_LDC_phase; % sauvegarde des valeurs identifiees !!! num differente !!!
% % liste_NRJ_save{it_pyr}{n_iter_LDC} = struct('NRJ_phase',NRJ_phase,'val_energie_0',val_energie_0,'val_V_totale',val_V_totale,'NRJ_def_totale',NRJ_def_totale,'NRJ_def_moyenne',NRJ_def_moyenne); % liste globale des NRJs
% % liste_convergence_save{it_pyr}{n_iter_LDC} = struct('vec_dA_relatif',vec_dA_relatif,'pctg_conv',pctg_conv,'val_tol',tolerance_LDC(it_pyr)); % liste des valeurs de convergences
% % 
% % t_fin = cputime;
% % disp(['IDENTIFICATION : ' num2str(t_fin-t_ini) ' s - ' num2str(pctg_conv) '% PHASES CONVERGES']);
% % disp(' ');
% % 
% % else % identification OFF
% %  disp(['IDENTIFICATION DESACTIVEE']);
% %  test_convergence_LDC = true; % on a fait 1 boucle, on sort gr�ce au end suivant
% % end % ON/OFF de l'identification mat
% % 
% % % end % Boucle de crit�res
% % 
% % if ( n_iter_LDC == nb_iter_LDC_max-1) % si on n'a pas atteint la convergence
% %  liste_data_LDC_phase_save{it_pyr+1}{nb_iter_LDC_max} = nonconv_ID(nb_pha,it_pyr,nb_iter_LDC_max,nb_iter_LDC_save,liste_data_LDC_phase_save); % sauvegarde d'une moyenne des n dernieres valeurs ID
% % end
% % 
% % n_pyr_pha_old = n_pyr_pha; % sauvegarde de la config en phases pour detecter un nouveau maillage
% % U_save{it_pyr} = save_dep(mat_pos,nb_noeuds_maillage_sig,Ux,Uy,Uz); % sauvegarde des dep obtenus avant chgt de maillage
% % liste_LDC_eps_save(it_pyr+1,:) = vec_LDC_eps; % sauvegarde des regles des LDC sur le maillage mesure !!! num differente !!!
% % liste_elem_pha_save{it_pyr} = liste_elem_pha;
% % liste_elem_eps_save{it_pyr} = liste_elem_mes;
% % mat_pos_save{it_pyr} = mat_pos;
% % 
% % % end % end pour le raffinement des elts sig
% % % end % end pour le raffinement des elts pha OU pyr pr�programm�
% % 
% % % nom de sauvegarde
% % nom_save = [ num2str(nb_pha(end)) 'pha' num2str(nb_sig(end)) 'sig_' 'ipyr' num2str(indice_pyr) '_' num2str(var_fixe) 'Fixe_f' num2str(err_Lame) '_' num2str(type_cpt) '_relax' num2str(alpha_relaxation_lambda) '-' num2str(alpha_relaxation_mu_r) ];
% % 
% % %%% REPRESENTATIONS GRAPHIQUES DES IDENTIFICATIONS
% % if ( test_identification )
% %  disp([ '      ' num2str(n_i_A) ' matrices identif�es non-d�finies positives']);
% %  sauvegarde_PT;
% %  if ( nb_pha(it_pyr) == 1 ) % homogene
% %   plot_hom; % trace la convergences des coeffs id
% %  else % heterogene
% % %   plot_het;
% %   PT_traces_ID;
% %  end
% % elseif ( strcmp(test_save_fic ,'Dep') == 1 )
% %  save([ num2str(test_save_fic) '_' num2str(nom_save) '.mat'],'U_save','nb_pha','nb_sig');
% % end
% % 
%toc
