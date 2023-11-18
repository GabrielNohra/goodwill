clearvars;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% INITIALISATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('INITIALISATION');
disp(' ');

% nom du fichier de mesures de deplacement
%nom_fichier_deplacement = 'd:\Documents\Bureautique\Recherche\Theses\TheseSamuel\MFiles_ERCM\donnees_dep.don';
% nom_fichier_deplacement = 'd:\Documents\Bureautique\Recherche\Theses\TheseSamuel\MFiles_ERCM\donnees_dep_cisaillement.don';
%nom_fichier_deplacement = 'd:\Documents\Bureautique\Recherche\Theses\TheseSamuel\MFiles_ERCM\donnees_dep_cisaillement_80Hz.don';
%nom_fichier_deplacement = 'd:\Documents\Bureautique\Recherche\Theses\TheseSamuel\MFiles_ERCM\donnees_dep_cisaillement_80Hz_bis.don';
%nom_fichier_deplacement = 'd:\Documents\Bureautique\Recherche\Theses\TheseSamuel\MFiles_ERCM\donnees_dep_cisaillement_80Hz_het.don';
%nom_fichier_deplacement = '/users/bionanonmri/nohra/Documents/MATLAB/data/donnees_dep.don';
liste_LdC = creation_LdC_anisotrope_repere_global();

path_dir = {'/users/bionanonmri/nohra/Documents/MATLAB/data/donnees_dep_cisaillement.don', ...
            '/users/bionanonmri/nohra/Documents/MATLAB/goodwill',...
            '/users/bionanonmri/nohra/Documents/MATLAB/results/161123/kappa'};

            % '/users/bionanonmri/nohra/Documents/MATLAB/results/091123/initialValues/elastic',...
            % '/users/bionanonmri/nohra/Documents/MATLAB/results/091123/initialValues/viscoelastic',...

% valeur de l'amplitude du bruit a rajouter (utile pour les donnees synthetiques uniquement)
amplitude_bruit_Gaussien_U = 0; % pourcentage de norme_U_max
% amplitude_bruit_Gaussien_U = 0.05; % pourcentage de norme_U_max

% frequence de l'essai
f_mec = 35; % Defintion de la frequence de sollicitation [Hz]
%f_mec = 80; % Defintion de la frequence de sollicitation [Hz]

% DEFINITION DES PARAMETRES DU COMPORTEMENT A IDENTIFIER
% type de comportement ? identifier => n_LdC_identification = 1 .. 12
%n_LdC_identification = 1;
n_LdC_identification = 7;
struct_param_comportement_a_identifier = liste_LdC{n_LdC_identification};
vec_numeros_parametres_comportement = [1 2];
vec_numeros_parametres_masse = [3];
% definition des parametres a identifier et a conserver "tels quels"
% vec_numeros_parametres_a_identifier = [1 2];
vec_numeros_parametres_a_identifier = [2];
vec_test_local = true(1,length(struct_param_comportement_a_identifier.liste_parametres_comportement)+length(struct_param_comportement_a_identifier.liste_parametres_masse));
vec_test_local(vec_numeros_parametres_a_identifier) = false;
vec_numeros_parametres_fixes = find ( vec_test_local );
clear vec_test_local;
% parametres d'initialisation
% lambda_r0 = 40000.; % [Pa]
% lambda_i0 = 0.; % [Pa]
% mu_r0 = 1800.; % [Pa]
% mu_i0 = 180.; % [Pa]
% rho_0 = 1020.; % [kg/m^3]
lambda_r0 = 41382.; % [Pa]
lambda_i0 = 0.; % [Pa]
mu_r0 = 1.5*1743.; % [Pa]
mu_i0 = 0.75*174.3; % [Pa]
rho_0 = 1020.; % [kg/m^3]
vec_param_initialisation = [(lambda_r0+1i*lambda_i0) (mu_r0+1i*mu_i0) rho_0];
% bornes d'identification
seuil_identification_min = 0.1;
seuil_identification_max = 10.;
vec_borne_min_param_identification = seuil_identification_min*vec_param_initialisation(vec_numeros_parametres_a_identifier);
vec_borne_max_param_identification = seuil_identification_max*vec_param_initialisation(vec_numeros_parametres_a_identifier);
% modification des parties imaginaires des bornes max qui sont a "0"
vec_borne_max_param_identification(imag(vec_borne_max_param_identification) == 0) = real(vec_borne_max_param_identification(imag(vec_borne_max_param_identification) == 0))+1i*0.1*real(vec_borne_max_param_identification(imag(vec_borne_max_param_identification) == 0));
% seuil de la relaxation sur les bornes "min" et "max"
seuil_relaxation_bornes_min_max = 0.5;
% creation de la structure contenant les parametres du comportement a identifier
struct_param_comportement_a_identifier.vec_numeros_parametres_comportement = vec_numeros_parametres_comportement;
struct_param_comportement_a_identifier.vec_numeros_parametres_masse = vec_numeros_parametres_masse;
struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier = vec_numeros_parametres_a_identifier;
struct_param_comportement_a_identifier.vec_numeros_parametres_fixes = vec_numeros_parametres_fixes;
struct_param_comportement_a_identifier.vec_param_initialisation = vec_param_initialisation;
struct_param_comportement_a_identifier.vec_borne_min_param_identification = vec_borne_min_param_identification;
struct_param_comportement_a_identifier.vec_borne_max_param_identification = vec_borne_max_param_identification;
struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max = seuil_relaxation_bornes_min_max;
clear lambda_r0 lambda_i0 mu_r0 mu_i0 rho_0 vec_numeros_parametres_comportement vec_numeros_parametres_masse vec_numeros_parametres_a_identifier vec_numeros_parametres_fixes seuil_identification_min seuil_identification_max vec_borne_min_param_identification vec_borne_max_param_identification seuil_relaxation_bornes_min_max;

% DEFINITION DES PARAMETRES DU COMPORTEMENT POUR NORMALISER
% definition du type de comportement pour normaliser => n_LdC_normalisation = 1 .. 6
% choisir un comportement ayant les memes symetries que celui qui est identifie
n_LdC_normalisation = 1;
struct_param_comportement_normalisation = liste_LdC{n_LdC_normalisation};
vec_numeros_parametres_comportement = struct_param_comportement_a_identifier.vec_numeros_parametres_comportement;
% parametres d'initialisation
beta_r = 1;
beta_i = 1;
vec_param_initialisation = beta_r*real(struct_param_comportement_a_identifier.vec_param_initialisation(struct_param_comportement_a_identifier.vec_numeros_parametres_comportement)) + beta_i*imag(struct_param_comportement_a_identifier.vec_param_initialisation(struct_param_comportement_a_identifier.vec_numeros_parametres_comportement));
% creation de la structure contenant les parametres pour normaliser
struct_param_comportement_normalisation.vec_numeros_parametres_comportement = vec_numeros_parametres_comportement;
struct_param_comportement_normalisation.vec_param_initialisation = vec_param_initialisation;
clear lambda_r0 mu_r0 vec_numeros_parametres_comportement;

% parametres de filtrage des deplacements
struct_filtrage_deplacement = struct('type','sans'); % Filtrage eventuel des deplacements
%struct_filtrage_deplacement = struct('type','Gaussien','Sigma_x',1,'R_x',3,'Sigma_y',1,'R_y',3,'Sigma_z',1,'R_z',3);

% PARAMETRES DES SUBZONES
% dimensions de la subzone (en nombre d'elements de contrainte)
%L_x_sub_zone = (2*10+1); % en nombre d'elements de contrainte dans la direction "x"
%L_y_sub_zone = (2*10+1); % en nombre d'elements de contrainte dans la direction "y"
%L_z_sub_zone = (2*10+1); % en nombre d'elements de contrainte dans la direction "z"
L_x_sub_zone = (2*5+1); % en nombre d'elements de contrainte dans la direction "x"
L_y_sub_zone = (2*5+1); % en nombre d'elements de contrainte dans la direction "y"
L_z_sub_zone = (2*5+1); % en nombre d'elements de contrainte dans la direction "z"
%L_x_sub_zone = (2*3+1); % en nombre d'elements de contrainte dans la direction "x"
%L_y_sub_zone = (2*3+1); % en nombre d'elements de contrainte dans la direction "y"
%L_z_sub_zone = (2*3+1); % en nombre d'elements de contrainte dans la direction "z"
%L_x_sub_zone = (2*2+1); % en nombre d'elements de contrainte dans la direction "x"
%L_y_sub_zone = (2*2+1); % en nombre d'elements de contrainte dans la direction "y"
%L_z_sub_zone = (2*2+1); % en nombre d'elements de contrainte dans la direction "z"
% taux de recouvrement des subzones
taux_recouvrement_sub_zones_par_MAJ_materielle = 0.5; % 0 : sub-zones disjointes, 1 : sub-zones chevauchantes decalees d'un seul element de contrainte

% TYPE DE METHODE D'IDENTIFICATION:
type_identification = 'MERC';
% type_identification = 'FEMU_AFC'; % no_BC
% type_identification = 'FEMU_DFC'; % BC


% Facteur de relaxation [0: totale, 1: nul] a imposer si _ident devient < 0 %C%
% alpha_relaxation_lambda = 1.; % pas de relaxation : on garde la valeur identifiee
% alpha_relaxation_mu_r = 1.; % relax forte mais plus efficace
% alpha_relaxation_mu_i = 1.; % relax forte mais plus efficace

% MAILLAGE DES CONTRAINTES
% numero de l'element de reference: constant (1) ou lineaire (2) ou quad 20 noeuds (3) ou quad 27 noeuds (4) ; retravailler les (1)
n_elem_sig = 4; % 1, 2, 3 ou 4
% % discretisation du maillage
% %nb_elem_sig_x = 10;
% %nb_elem_sig_y = 11;
% %nb_elem_sig_z = 12;
% nb_elem_sig_x = 20;
% nb_elem_sig_y = 21;
% nb_elem_sig_z = 22;
% Type d'integration pour les elements de la matrice de raideur "K" et la matrice de masse "M"
%n_integration_K = 7;
%n_integration_M = 7;
n_integration_K = 6;
n_integration_M = 6;
% Booleen pour savoir si on "lump" la matrice de masse
test_lump = false;
%test_lump = true;

% MAILLAGE DES PHASES
% Numero element de reference: constant (1) ou lineaire (2) ou quad 20 noeuds (3) ou quad 27 noeuds (4) ; retravailler les (1)
n_elem_pha = 1; % 1, 2, 3 ou 4
% discretisation du maillage
nb_elem_pha_x = 1;
nb_elem_pha_y = 1;
nb_elem_pha_z = 1;
%nb_elem_pha_x = 3;
%nb_elem_pha_y = 4;
%nb_elem_pha_z = 5;

% facteur pour calculer la tolerance sur la position : tolerance = dimension_maxi/facteur
facteur_tolerance_position = 10000.;

% parametres de convergence sur l'identification materielle
tolerance_LDC = 1e-4;
%nb_iter_LDC_max = 5;
% nb_iter_LDC_max = 10;
nb_iter_LDC_max = 200;
% nb_iter_LDC_max = 200;

% % seuil permettant de definir les elements de phase "pas assez deformes" pour pouvoir y identifier des proprietes
% seuil_NRJ = 0.1; % 0 : pas de filtre

% parametre de ponderation mesure/comportement de l'ERCM
% kappa = 0.0001;
%kappa = 0.01;
%kappa = 1;
%kappa = 10;
%kappa = 20;
%kappa = 200;
%kappa = 10000;
%kappa = 100000000000;
%kappa = 10000000000000;
%kappa = 0.01*1e11; % convergence extremement lente 
%kappa = 100*1e11; % bonne convergence si pas de bruit sur donnees

kappa = [0.001 0.01 0.1 1 10 100] * 1e13;
% kappa = [10000 100000 1000000] * 1e13;

% nombre de DDL par noeud
nb_DDL_par_noeud = 3;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEFINITIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% definition de la pulsation
omega_mec = 2*pi*f_mec;

% definition du type de conditions aux limites a imposer
if ( strcmp(type_identification,'MERC') == 1 )
 struct_CL = struct('type','sans');
elseif ( strcmp(type_identification,'FEMU_AFC') == 1 )
 struct_CL = struct('type','sans');
elseif ( strcmp(type_identification,'FEMU_DFC') == 1 )
%  struct_CL = struct('type','mesure','type_interpolation_U','linear','type_extrapolation_U','linear');
 struct_CL = struct('type','mesure_filtree','parametres_filtrage',struct('type','Gaussien','Sigma_x',1,'R_x',3,'Sigma_y',1,'R_y',3,'Sigma_z',1,'R_z',3));
end

% definition de la structure contenant tous les parametres de definition du maillage EF
%struct_parametres_maillage_EF = struct('nb_elem_pha_x',nb_elem_pha_x,'nb_elem_pha_y',nb_elem_pha_y,'nb_elem_pha_z',nb_elem_pha_z,'nb_elem_sig_x',nb_elem_sig_x,'nb_elem_sig_y',nb_elem_sig_y,'nb_elem_sig_z',nb_elem_sig_z);
struct_parametres_maillage_EF = struct('nb_elem_pha_x',nb_elem_pha_x,'nb_elem_pha_y',nb_elem_pha_y,'nb_elem_pha_z',nb_elem_pha_z);
clear nb_elem_pha_x nb_elem_pha_y nb_elem_pha_z nb_elem_sig_x nb_elem_sig_y nb_elem_sig_z;

% definition de la structure contenant tous les parametres de calcul pour les matrices de masse et de raideur
struct_param_masse_raideur = struct('n_integration_K',n_integration_K,'n_integration_M',n_integration_M,'test_lump',test_lump);

% defintion des elements de reference
liste_elem_ref = creation_elem_ref;

% definition des elements de reference associes aux "phases" et aux "contraintes"
elem_pha_ref = liste_elem_ref{n_elem_pha};
elem_sig_ref = liste_elem_ref{n_elem_sig};


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LECTURE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lecture du fichier de resultats mesures (positions + deplacements)
disp('LECTURE FICHIERS MESURES');
% t_ini = cputime;

mat_data = load(path_dir{1});

mat_pos_mes = mat_data(:,1:3)';
mat_U_mes = mat_data(:,4:2:9)'+1i*mat_data(:,5:2:9)';
% prise en compte eventuelle du bruit rajoute
if ( amplitude_bruit_Gaussien_U > 0 )
 norme_U_max = sqrt(max(sum(abs(mat_U_mes).^2,1)));
 mat_U_mes = (real(mat_U_mes)+norme_U_max*amplitude_bruit_Gaussien_U*randn(size(mat_U_mes)))+1i*(imag(mat_U_mes)+norme_U_max*amplitude_bruit_Gaussien_U*randn(size(mat_U_mes))); 
 clear norme_U_max;
end
clear mat_data;

nb_dim = size(mat_U_mes,1);

% longueur 1D totale du maillage

% L_tot = max(mat_pos_mes') - min(mat_pos_mes');
% tolerance_position = max(L_tot)/facteur_tolerance_position;

Lx_tot = max(mat_pos_mes(1,:))-min(mat_pos_mes(1,:));
Ly_tot = max(mat_pos_mes(2,:))-min(mat_pos_mes(2,:));
Lz_tot = max(mat_pos_mes(3,:))-min(mat_pos_mes(3,:));

L_tot = [Lx_tot,Ly_tot,Lz_tot];
tolerance_position = max(L_tot)/facteur_tolerance_position;

clear Lx_tot Ly_tot Lz_tot L_tot;

% reconstruction de la grille (reguliere) de mesure 
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

% conversion numerotation (i,j,k) => np
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

% on supprime un certain nombre de mesures pour simuler une acquisition incomplete ...
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

% t_fin = cputime;
% disp(['      ' num2str(t_fin-t_ini) ' s']);
disp(' ');

% affichage de la grille de mesure

figure;
hp = plot3(reshape(mat_X_mes_3D(:,:,:,1),1,[]),reshape(mat_X_mes_3D(:,:,:,2),1,[]),reshape(mat_X_mes_3D(:,:,:,3),1,[]),'xk');
grid;
xlabel('x (m)');
ylabel('y (m)');
zlabel('z (m)');
title('grille de mesure');

% creation maillages "element-finis": phases et contraintes
disp('CREATION MAILLAGES "ELEMENTS-FINIS"');

% t_ini = cputime;
[liste_elem_pha,mat_pos_maillage_pha,mat_pos_pha,mat_n_pha,liste_elem_sig,mat_pos_maillage_sig,mat_pos_sig,mat_n_sig] = creation_maillages_EF(mat_X_mes_3D,elem_pha_ref,elem_sig_ref,n_integration_K,n_integration_M,struct_parametres_maillage_EF,struct_grille_mes);
[nb_elem_sig_x,nb_elem_sig_y,nb_elem_sig_z] = size(mat_n_sig);
struct_parametres_maillage_EF.nb_elem_sig_x = nb_elem_sig_x;
struct_parametres_maillage_EF.nb_elem_sig_y = nb_elem_sig_y;
struct_parametres_maillage_EF.nb_elem_sig_z = nb_elem_sig_z;

% determination des nombre d'elements de phase dans les directions "x", "y" et "z"
nb_elem_pha = length(liste_elem_pha);
ni_elem_pha = struct_parametres_maillage_EF.nb_elem_pha_x;
nj_elem_pha = struct_parametres_maillage_EF.nb_elem_pha_y;
nk_elem_pha = struct_parametres_maillage_EF.nb_elem_pha_z;

% affichage des elements de phases
mat_coul = jet(nb_elem_pha);
figure;
hold on;
n_elem_pha = 0;
for i = 1:ni_elem_pha
    for j = 1:nj_elem_pha
        for k = 1:nk_elem_pha
            n_elem_pha = n_elem_pha+1;
            if ( strcmp(liste_elem_pha{n_elem_pha}.type_elem,'HEX1') == 1 )
                hp = plot3(mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds_maillage,1),mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds_maillage,2),mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds_maillage,3),'xk');
            else
                hp = plot3(mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds,1),mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds,2),mat_pos_maillage_pha(liste_elem_pha{n_elem_pha}.vec_n_noeuds,3),'xk');
            end
            set(hp,'Color',mat_coul(n_elem_pha,:));
        end
    end
end
grid;
xlabel('x (m)');
ylabel('y (m)');
zlabel('z (m)');
title('noeuds du maillage de phase');

% suppression des noeuds inutilises dans le maillage de phase
vec_test_pos_pha = false(1,size(mat_pos_pha,1));
for nn_elem_pha = 1:nb_elem_pha
    vec_test_pos_pha(liste_elem_pha{nn_elem_pha}.vec_n_noeuds) = true;
end
nb_noeuds_pha = length(find(vec_test_pos_pha));
vec_correspondance_n_DDL_pha_n_noeud_pha = find(vec_test_pos_pha);
vec_correspondance_n_noeud_pha_n_DLL_pha = zeros(1,length(vec_test_pos_pha));
vec_correspondance_n_noeud_pha_n_DLL_pha(vec_correspondance_n_DDL_pha_n_noeud_pha) = 1:nb_noeuds_pha;

% modification du maillage de phase (suppression noeuds inutilises)
mat_pos_pha = mat_pos_pha(vec_correspondance_n_DDL_pha_n_noeud_pha,:);
for n_pha = 1:nb_elem_pha
    liste_elem_pha{n_pha}.vec_n_noeuds = vec_correspondance_n_noeud_pha_n_DLL_pha(liste_elem_pha{n_pha}.vec_n_noeuds);
end

% determination des numeros des noeuds de phase par element de phase
mat_n_noeuds_pha_par_pha = nan(elem_pha_ref.nb_noeuds,length(liste_elem_pha));
for n_pha = 1:nb_elem_pha
 mat_n_noeuds_pha_par_pha(:,n_pha) = liste_elem_pha{n_pha}.vec_n_noeuds';
end

% determination des elements de contrainte contenus dans chaque phase
for n_pha = 1:nb_elem_pha
 liste_elem_pha{n_pha}.vec_n_sig_K = [];
 liste_elem_pha{n_pha}.vec_n_sig_M = [];
end
for n_sig = 1:length(liste_elem_sig)
 vec_n_pha_elem_sig_K = unique(liste_elem_sig{n_sig}.vec_n_pha_G_K);
 for nn_pha = 1:length(vec_n_pha_elem_sig_K)
  n_pha = vec_n_pha_elem_sig_K(nn_pha);
  liste_elem_pha{n_pha}.vec_n_sig_K = [liste_elem_pha{n_pha}.vec_n_sig_K n_sig];
 end
 vec_n_pha_elem_sig_M = unique(liste_elem_sig{n_sig}.vec_n_pha_G_M);
 for nn_pha = 1:length(vec_n_pha_elem_sig_M)
  n_pha = vec_n_pha_elem_sig_M(nn_pha);
  liste_elem_pha{n_pha}.vec_n_sig_M = [liste_elem_pha{n_pha}.vec_n_sig_M n_sig];
 end
end

% determination des nombre d'elements de contrainte dans les directions "x", "y" et "z"
nb_elem_sig = length(liste_elem_sig);
ni_elem_sig = struct_parametres_maillage_EF.nb_elem_sig_x;
nj_elem_sig = struct_parametres_maillage_EF.nb_elem_sig_y;
nk_elem_sig = struct_parametres_maillage_EF.nb_elem_sig_z;

% suppression des noeuds inutilises dans le maillage de contrainte
vec_test_pos_sig = false(1,size(mat_pos_sig,1));
for nn_elem_sig = 1:nb_elem_sig
    vec_test_pos_sig(liste_elem_sig{nn_elem_sig}.vec_n_noeuds) = true;
end
nb_noeuds_sig = length(find(vec_test_pos_sig));
vec_correspondance_n_DDL_sig_n_noeud_sig = find(vec_test_pos_sig);
vec_correspondance_n_noeud_sig_n_DLL_sig = zeros(1,length(vec_test_pos_sig));
vec_correspondance_n_noeud_sig_n_DLL_sig(vec_correspondance_n_DDL_sig_n_noeud_sig) = 1:nb_noeuds_sig;

% modification du maillage de contraintes (suppression noeuds inutilises)
mat_pos_maillage_sig = mat_pos_maillage_sig(vec_correspondance_n_DDL_sig_n_noeud_sig,:);
for n_sig = 1:nb_elem_sig
    liste_elem_sig{n_sig}.vec_n_noeuds = vec_correspondance_n_noeud_sig_n_DLL_sig(liste_elem_sig{n_sig}.vec_n_noeuds);
end

% % affichage des elements de contrainte
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

mat_coul = jet(nb_elem_sig);
figure;
hold on;
for n_elem_sig = 1:nb_elem_sig
    if ( strcmp(liste_elem_sig{n_elem_sig}.type_elem,'HEX1') == 1 )
        hp = plot3(mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds_maillage,1),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds_maillage,2),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds_maillage,3),'xk');
    else
        hp = plot3(mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,1),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,2),mat_pos_maillage_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,3),'xk');
    end
    set(hp,'Color',mat_coul(n_elem_sig,:));
end
grid;
xlabel('x (m)');
ylabel('y (m)');
zlabel('z (m)');
title('noeuds du maillage de contrainte');
% t_fin = cputime;
% disp(['      ' num2str(t_fin-t_ini) ' s']);
disp(' ');

% determination des elements de phase qui ne contiennent pas de points de mesure "interieurs" ou bien pas d'element de contrainte
mat_test_mes_dans_pha = false(ni_elem_pha,nj_elem_pha,nk_elem_pha);
for nn_elem_pha = 1:nb_elem_pha
    elem_pha = liste_elem_pha{nn_elem_pha};
    if ( strcmp(elem_pha.type_elem,'HEX1') == 1 )
        test_mes_dans_pha = (sum(((mat_pos_mes(1,:) >= (min(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,1))+tolerance_position)) & (mat_pos_mes(1,:) <= (max(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,1))-tolerance_position)) & (mat_pos_mes(2,:) >= (min(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,2))+tolerance_position)) & (mat_pos_mes(2,:) <= (max(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,2))-tolerance_position)) & (mat_pos_mes(3,:) >= (min(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,3))+tolerance_position)) & (mat_pos_mes(3,:) <= (max(mat_pos_maillage_pha(elem_pha.vec_n_noeuds_maillage,3))-tolerance_position)))) > 0);
    else
        test_mes_dans_pha = (sum(((mat_pos_mes(1,:) >= (min(mat_pos_pha(elem_pha.vec_n_noeuds,1))+tolerance_position)) & (mat_pos_mes(1,:) <= (max(mat_pos_pha(elem_pha.vec_n_noeuds,1))-tolerance_position)) & (mat_pos_mes(2,:) >= (min(mat_pos_pha(elem_pha.vec_n_noeuds,2))+tolerance_position)) & (mat_pos_mes(2,:) <= (max(mat_pos_pha(elem_pha.vec_n_noeuds,2))-tolerance_position)) & (mat_pos_mes(3,:) >= (min(mat_pos_pha(elem_pha.vec_n_noeuds,3))+tolerance_position)) & (mat_pos_mes(3,:) <= (max(mat_pos_pha(elem_pha.vec_n_noeuds,3))-tolerance_position)))) > 0);
    end
%    if ( test_mes_dans_pha )
    if ( test_mes_dans_pha && ~isempty(elem_pha.vec_n_sig_K) && ~isempty(elem_pha.vec_n_sig_M) )
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

figure;
hold on;
plot3(mat_pos_pha(vec_n_noeuds_pha_a_supprimer,1),mat_pos_pha(vec_n_noeuds_pha_a_supprimer,2),mat_pos_pha(vec_n_noeuds_pha_a_supprimer,3),'or');
plot3(mat_pos_pha(vec_n_noeuds_pha_a_conserver,1),mat_pos_pha(vec_n_noeuds_pha_a_conserver,2),mat_pos_pha(vec_n_noeuds_pha_a_conserver,3),'xk');
hold off;
grid;
xlabel('x (m)');
ylabel('y (m)');
zlabel('z (m)');
legend(liste_legende);
title('noeuds/elements de phase');

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
figure;
hold on;
plot3(mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_supprimer,1),mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_supprimer,2),mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_supprimer,3),'or');
plot3(mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_conserver,1),mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_conserver,2),mat_pos_maillage_pha(vec_n_noeuds_maillage_pha_a_conserver,3),'xk');
hold off;
grid;
xlabel('x (m)');
ylabel('y (m)');
zlabel('z (m)');
legend(liste_legende);title('noeuds maillage de phase');
clear liste_legende;

% determination des matrices donnant la correspondance (i_sig,j_sig,k_sig) => n_elem_sig
mat_correspondance_i_sig_j_sig_k_sig_n_elem_sig = nan(ni_elem_sig,nj_elem_sig,nk_elem_sig);
for n_elem_sig = 1:nb_elem_sig
    elem_sig = liste_elem_sig{n_elem_sig};
    mat_correspondance_i_sig_j_sig_k_sig_n_elem_sig(elem_sig.vec_ijk(1),elem_sig.vec_ijk(2),elem_sig.vec_ijk(3)) = n_elem_sig;
end

% determination des multiplicites des points de mesure dans le maillage des contraintes
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

% filtrage eventuel des mesures
disp('FILTRAGE DES MESURES');
% t_ini = cputime;
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
% --- %ATT% On met les grilles de position et de deplacement en accord sur les "nan" ... => A AMELIORER %ATT%
    mat_U_mes_3D(isnan(mat_X_mes_3D(:))) = nan;
    mat_X_mes_3D(isnan(mat_U_mes_3D(:))) = nan;
    clear mat_x_mes mat_y_mes mat_z_mes ii_mes_no_nan mat_U_filtr mat_U_filtr_3D;
end
% t_fin = cputime;
% disp(['      ' num2str(t_fin-t_ini) ' s']);
disp(' ');

% determination des deplacements a appliquer en conditions aux limites
disp('DETERMINATION DES DEPLACMENTS A APPLIQUER EN CONDITIONS AUX LIMITES');
% t_ini = cputime;
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
% --- %A FAIRE% : FILTRER LES CHAMPS MESURES FILTRES ET LES EVALUER SUR LA GRILLE DES CONTRAINTES %A FAIRE%
    disp('A FAIRE : FILTRER LES CHAMPS MESURES FILTRES ET LES EVALUER SUR LA GRILLE DES CONTRAINTES');
    [mat_U_sig_3D] = filtrage_deplacements_IRM(mat_X_mes_3D,mat_U_mes_3D,mat_pos_maillage_sig,struct_grille_mes,struct_CL.parametres_filtrage);
end
% t_fin = cputime;
% disp(['      ' num2str(t_fin-t_ini) ' s']);
disp(' ');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% DEBUT DU TEST SUR LA CONVERGENCE %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% SUR LES PROPRIETES MATERIELLES %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('BOUCLE D''IDENTIFICATION');

nb_sub_zones_x = taux_recouvrement_sub_zones_par_MAJ_materielle*(ni_elem_sig-(floor(ni_elem_sig/L_x_sub_zone)+1))+(floor(ni_elem_sig/L_x_sub_zone)+1);
nb_sub_zones_y = taux_recouvrement_sub_zones_par_MAJ_materielle*(nj_elem_sig-(floor(nj_elem_sig/L_y_sub_zone)+1))+(floor(nj_elem_sig/L_y_sub_zone)+1);
nb_sub_zones_z = taux_recouvrement_sub_zones_par_MAJ_materielle*(nk_elem_sig-(floor(nk_elem_sig/L_z_sub_zone)+1))+(floor(nk_elem_sig/L_z_sub_zone)+1);

nb_sub_zones_max = ceil(nb_sub_zones_x*nb_sub_zones_y*nb_sub_zones_z);
nb_faces_elem_ref = size(elem_sig_ref.n_noeuds_faces,1);

nb_parametres_comportement_a_identifier = length(struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier);

valKappa = nan(length(kappa),nb_iter_LDC_max+1);
sTime = zeros(1,length(kappa));

[t_ini_identification, t_fin_identification] = deal(zeros(1,length(kappa)));

for idx = 1:length(kappa)

    % INITIALISATION DES PROPRIETES
    disp('INITIALISATION DES PROPRIETES');
    struct_param_comportement_a_identifier.mat_param = struct_param_comportement_a_identifier.vec_param_initialisation.'*ones(1,nb_noeuds_pha);
    struct_param_comportement_normalisation.mat_param = struct_param_comportement_normalisation.vec_param_initialisation.'*ones(1,nb_noeuds_pha);
    disp(' ');

    test_convergence_LDC = false;
    n_iter_LDC = 1;
    t_ini_identification(1,idx) = cputime;
    
    liste_proprietes_iterations = cell(1,nb_iter_LDC_max+1);
    liste_proprietes_iterations{n_iter_LDC} = struct_param_comportement_a_identifier.mat_param(struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier,:);

    while ( (~test_convergence_LDC) && ( n_iter_LDC <= nb_iter_LDC_max) ) % Debut du critere sur la convergence (utile pour id)
        
        tic;

        disp(['    iteration ' num2str(n_iter_LDC)]);
        
        % boucles sur les sub-zones
        n_sub_zone = 0;
        test_fin_sub_zones = false;
        vec_test_n_elem_sig_centre_sub_zone = false(1,nb_elem_sig);
        
        mat_proprietes_identifies_sub_zones = zeros(nb_parametres_comportement_a_identifier,nb_sub_zones_max,size(struct_param_comportement_a_identifier.mat_param,2));
        mat_test_proprietes_identifies_sub_zones = false(nb_sub_zones_max,size(struct_param_comportement_a_identifier.mat_param,2));
        
        liste_sub_zones = cell(1,nb_sub_zones_max);
        
        %     while ( (n_sub_zone < nb_sub_zones_max) && ~test_fin_sub_zones )
        %
        % % determination de l'element de contrainte correspondant au centre de la sub-zone
        %         n_sub_zone = n_sub_zone+1;
        %
        %         disp(['        traitement de la sub-zone numero ' num2str(n_sub_zone)]);
        % %         disp('            recherche des elements de contraintes de la sub-zone');
        %
        %         t_ini = cputime;
        %
        %         vec_nn_sig_local = find ( ~vec_test_n_elem_sig_centre_sub_zone );
        %         n_sig_sub_zone = vec_nn_sig_local(randi(length(vec_nn_sig_local)));
        %         elem_sig_centre_sub_zone = liste_elem_sig{n_sig_sub_zone};
        n_sub_zone = n_sub_zone+1;
        vec_nn_sig_local = 1:length(vec_test_n_elem_sig_centre_sub_zone);
        n_sig_sub_zone = vec_nn_sig_local(floor(length(vec_nn_sig_local)/2)+1);
        elem_sig_centre_sub_zone = liste_elem_sig{n_sig_sub_zone};
        
        i_sig_centre_sub_zone = elem_sig_centre_sub_zone.vec_ijk(1);
        j_sig_centre_sub_zone = elem_sig_centre_sub_zone.vec_ijk(2);
        k_sig_centre_sub_zone = elem_sig_centre_sub_zone.vec_ijk(3);
        
        % determination de tous les elements de contrainte de la sub-zone
        vec_n_elem_sig_sub_zone = nan(1,nb_sub_zones_max);
        n_elem_sig_sub_zone = 0;
        nb_points_mesure_sub_zone = 0;
        vec_test_noeuds_sig_sub_zone = false(1,size(mat_pos_maillage_sig,1));
        for i = i_sig_centre_sub_zone-floor(L_x_sub_zone/2)+(0:(L_x_sub_zone-1))
            for j = j_sig_centre_sub_zone-floor(L_y_sub_zone/2)+(0:(L_y_sub_zone-1))
                for k = k_sig_centre_sub_zone-floor(L_z_sub_zone/2)+(0:(L_z_sub_zone-1))
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
        % t_fin = cputime;
        %         disp(['                ' num2str(t_fin-t_ini) ' s']);
        
        % determinations des elements de contrainte de la sub-zone
        %         disp('            determinations des elements de contrainte de la sub-zone');
        % t_ini = cputime;
        % determination des noeuds situes dans la sub-zone
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
        
        % determination des correspondances de numerotation "globale" - "locale"
        vec_correspondance_n_noeud_sig_global_n_noeud_sig_local = zeros(1,size(mat_pos_maillage_sig,1));
        vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(vec_n_noeuds_sig_sub_zone) = 1:length(vec_n_noeuds_sig_sub_zone);
        
        % determination des correspondances de numerotation "locale" - "globale"
        vec_correspondance_n_noeud_sig_local_n_noeud_sig_global = vec_n_noeuds_sig_sub_zone;
        % t_fin = cputime;
        %         disp(['                ' num2str(t_fin-t_ini) ' s']);
        
        %         disp('            determinations des elements de phase de la sub-zone');
        % t_ini = cputime;
        % determination des noeuds de phase situes dans la sub-zone
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
        
        % determination des correspondances de numerotation "GLOBALE" - "LOCALE"
        vec_correspondance_n_noeud_pha_global_n_noeud_pha_local = zeros(1,size(mat_pos_pha,1));
        vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha_sub_zone) = 1:length(vec_n_noeuds_pha_sub_zone);
        
        % sauvegarde des informations de la sub-zone
        liste_sub_zones{n_sub_zone} = struct('vec_x_noeuds_mes_sub_zone',vec_x_noeuds_mes_sub_zone,'vec_y_noeuds_mes_sub_zone',vec_y_noeuds_mes_sub_zone,'vec_z_noeuds_mes_sub_zone',vec_z_noeuds_mes_sub_zone,'vec_correspondance_n_noeud_sig_global_n_noeud_sig_local',vec_correspondance_n_noeud_sig_global_n_noeud_sig_local,'vec_correspondance_n_noeud_sig_local_n_noeud_sig_global',vec_correspondance_n_noeud_sig_local_n_noeud_sig_global,'vec_n_noeuds_pha_sub_zone',vec_n_noeuds_pha_sub_zone,'vec_correspondance_n_noeud_pha_global_n_noeud_pha_local',vec_correspondance_n_noeud_pha_global_n_noeud_pha_local);
        liste_sub_zones{n_sub_zone}.liste_elem_sig_sub_zone = liste_elem_sig_sub_zone;
        
        % determination des correspondances de numerotation "LOCALE" - "GLOBALE"
        vec_correspondance_n_noeud_pha_local_n_noeud_pha_global = vec_n_noeuds_pha_sub_zone;
        % t_fin = cputime;
        %         disp(['                ' num2str(t_fin-t_ini) ' s']);
        
        % % affichage des noeuds de contrainte et de deplacement dans la subzone
        %         figure;
        %         hold on;
        %         plot3(mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,1),mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,2),mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,3),'xk');
        %         plot3(vec_x_noeuds_mes_sub_zone,vec_y_noeuds_mes_sub_zone,vec_z_noeuds_mes_sub_zone,'or');
        %         grid;
        %         xlabel('x (m)');
        %         ylabel('y (m)');
        %         zlabel('z (m)');
        %         title(['maillage sub-zone numero ' int2str(n_sub_zone)]);
        %         legend('sig','mes');
        
        % determination des noeuds situes sur la frontiere de la sub-zone
        mat_n_noeuds_faces_globale = zeros(nb_faces_elem_ref*length(vec_n_elem_sig_sub_zone),size(elem_sig_ref.n_noeuds_faces,2));
        n_face_globale = 0;
        
        % on determine tous les noeuds de toutes les faces des elements de contrainte de la sub-zone
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
        
        % on determine les faces dont les noeuds n'apparaissent qu'une seule fois => ils constituent la frontiere de la sub-zone
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
        
        % affichage des noeuds de contrainte dans la subzone et des noeuds frontiere de la suzone
        %         figure;
        %         hold on;
        %         plot3(mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,1),mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,2),mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,3),'xk');
        %         plot3(mat_pos_maillage_sig(vec_noeuds_frontieres,1),mat_pos_maillage_sig(vec_noeuds_frontieres,2),mat_pos_maillage_sig(vec_noeuds_frontieres,3),'or');
        %         grid;
        %         xlabel('x (m)');
        %         ylabel('y (m)');
        %         zlabel('z (m)');
        %         title('maillage siub-zone');
        %         legend('vol.','front.');
        
        vec_n_noeuds_frontieres_local = vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(vec_noeuds_frontieres);
        
        % determination des conditions aux limites sur la sub-zone
        %         disp('            determinations des conditions aux limites de la sub-zone');
        % t_ini = cputime;
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
        % t_fin = cputime;
        %         disp(['                ' num2str(t_fin-t_ini) ' s']);
        
        % determination des matrices de masse et de raideur sur la sub-zone
        %         disp(['            matrices equivalentes - ' num2str(nb_noeuds_sig_sub_zone*nb_DDL_par_noeud) ' DDL']);
        t_ini = cputime;
        [K,T,M,D,d_K_d_p] = raideur_ERCM(liste_elem_pha,liste_elem_sig_sub_zone,liste_elem_ref,struct_param_masse_raideur,vec_correspondance_n_noeud_sig_global_n_noeud_sig_local,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local,mat_pos_maillage_sig,nb_DDL_par_noeud,struct_param_comportement_a_identifier,struct_param_comportement_normalisation);
        % t_fin = cputime;
        %         disp(['                ' num2str(t_fin-t_ini) ' s']);
        
        % determination de l'operateur de projection sur la sub-zone
        %         disp('            operateur de projection');
        % t_ini = cputime;
        %         [N_mes,D_mes] = projection_mes(mat_pos_mes,mat_pos_maillage_sig,liste_elem_sig_sub_zone,mat_n_sig,liste_elem_ref,nb_DDL_par_noeud);
        ni_N_mes = nb_points_mesure_sub_zone*nb_DDL_par_noeud;
        nj_N_mes = length(vec_n_noeuds_sig_sub_zone)*nb_DDL_par_noeud;
        Nf = squeeze(liste_elem_ref{liste_elem_sig_sub_zone{1}.n_elem_ref}.f_Nf(0,0,0));
        vec_i_N_mes = nan(1,length(Nf)*ni_N_mes);
        vec_j_N_mes = nan(1,length(Nf)*ni_N_mes);
        vec_s_N_mes = nan(1,length(Nf)*ni_N_mes);
        i_prec = 0;
        n_mes_sub_zone = 0;
        vec_test_points_mes_dans_sub_zone = false(1,size(mat_pos_mes,2)); % pour ne pas tenir compte plusieurs fois d'une meme mesure
        vec_n_points_mes_local_dans_sub_zone = nan(1,size(mat_pos_mes,2)); % pour ne pas tenir compte plusieurs fois d'une meme mesure
        vec_n_points_mes_dans_sub_zone = nan(1,size(mat_pos_mes,2));
        for nn_elem_sig = 1:length(liste_elem_sig_sub_zone)
            elem_sig = liste_elem_sig_sub_zone{nn_elem_sig};
            for nn_mes_local = 1:length(elem_sig.vec_i_mes)
                i_mes = elem_sig.vec_i_mes(nn_mes_local);
                j_mes = elem_sig.vec_j_mes(nn_mes_local);
                k_mes = elem_sig.vec_k_mes(nn_mes_local);
                n_mes = 1+(i_mes-1)+ni_mes*((j_mes-1)+nj_mes*(k_mes-1));
                %                if ( ~isnan(mat_x_mes(n_mes)) )
                if ( ~vec_test_points_mes_dans_sub_zone(n_mes) ) % pour ne pas tenir compte plusieurs fois d'une meme mesure
                    vec_test_points_mes_dans_sub_zone(n_mes) = true;
                    n_mes_sub_zone = n_mes_sub_zone+1;
                    vec_n_points_mes_dans_sub_zone(n_mes_sub_zone) = n_mes;
                    vec_n_points_mes_local_dans_sub_zone(n_mes) = n_mes_sub_zone;
                    x_ref = elem_sig.vec_ksi_mes(nn_mes_local);
                    y_ref = elem_sig.vec_eta_mes(nn_mes_local);
                    z_ref = elem_sig.vec_zeta_mes(nn_mes_local);
                    Nf = squeeze(liste_elem_ref{elem_sig.n_elem_ref}.f_Nf(x_ref,y_ref,z_ref));
                    for n_DDL = 1:nb_DDL_par_noeud
                        vec_i_N_mes(i_prec+(1:length(Nf))) = nb_DDL_par_noeud*(n_mes_sub_zone-1)+n_DDL;
                        vec_j_N_mes(i_prec+(1:length(Nf))) = nb_DDL_par_noeud*(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)-1)+n_DDL;
                        vec_s_N_mes(i_prec+(1:length(Nf))) = Nf;
                        i_prec = i_prec+length(Nf);
                    end
                end
                %                end
            end
        end
        vec_n_points_mes_dans_sub_zone = vec_n_points_mes_dans_sub_zone(~isnan(vec_n_points_mes_dans_sub_zone));
        ii_no_nan = find ( ~isnan(vec_n_points_mes_local_dans_sub_zone) );
        [~,ii_sort] = sort(vec_n_points_mes_local_dans_sub_zone(ii_no_nan));
        vec_n_points_mes_local_dans_sub_zone = vec_n_points_mes_local_dans_sub_zone(ii_no_nan(ii_sort));
        clear ii_no_nan ii_sort;
        vec_i_N_mes = vec_i_N_mes(~isnan(vec_i_N_mes));
        vec_j_N_mes = vec_j_N_mes(~isnan(vec_j_N_mes));
        vec_s_N_mes = vec_s_N_mes(~isnan(vec_s_N_mes));
        ni_N_mes = max(vec_i_N_mes);
        N_mes = sparse(vec_i_N_mes,vec_j_N_mes,vec_s_N_mes,ni_N_mes,nj_N_mes);
        %         D_mes = N_mes'*N_mes;
        % t_fin = cputime;
        %         disp(['                ' num2str(t_fin-t_ini) ' s']);
        
        % calcul du seconds membre R
        %         disp('            second membre');
        % t_ini = cputime;
        nb_mes = length(vec_n_points_mes_dans_sub_zone);
        vec_U_mes = zeros(nb_DDL_par_noeud*nb_mes,1);
        for n_DDL = 1:nb_DDL_par_noeud
            %             vec_U_mes(n_DDL:nb_DDL_par_noeud:end) = mat_U_mes(n_DDL,vec_n_points_mes_dans_sub_zone);
            vec_U_mes(n_DDL:nb_DDL_par_noeud:end) = mat_U_mes(n_DDL,vec_n_points_mes_dans_sub_zone(vec_n_points_mes_local_dans_sub_zone));
        end
        vec_F = zeros(size(K,1),1);
        %         vec_R = N_mes'*vec_U_mes;
        vec_R = D*(N_mes'*vec_U_mes);
        
        % --- affichage des deplacements projetes sur la grille de contraine
        %         vec_R_U = (D\N')*vec_U_mes;
        % % champs calcules
        %         vec_R_Ux = vec_R_U(1:nb_DDL_par_noeud:size(K,1));
        %         vec_R_Uy = vec_R_U(2:nb_DDL_par_noeud:size(K,1));
        %         vec_R_Uz = vec_R_U(3:nb_DDL_par_noeud:size(K,1));
        % %         mat_pos_maillage_sig_sub_zone = mat_pos_maillage_sig(vec_correspondance_n_noeud_sig_local_n_noeud_sig_global,:);
        %         mat_pos_maillage_sig_sub_zone = mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,:);
        %         vec_x_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,1);
        %         vec_y_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,2);
        %         vec_z_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,3);
        %         F_Ux_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_R_Ux),'linear','none');
        %         F_Ux_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_R_Ux),'linear','none');
        %         F_Uy_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_R_Uy),'linear','none');
        %         F_Uy_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_R_Uy),'linear','none');
        %         F_Uz_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_R_Uz),'linear','none');
        %         F_Uz_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_R_Uz),'linear','none');
        %         vec_grille_x_sig_sub_zone = unique(sort(vec_x_sig_sub_zone));
        %         vec_grille_y_sig_sub_zone = unique(sort(vec_y_sig_sub_zone));
        %         vec_grille_z_sig_sub_zone = unique(sort(vec_z_sig_sub_zone));
        %         [mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone] = meshgrid(vec_grille_x_sig_sub_zone,vec_grille_y_sig_sub_zone,vec_grille_z_sig_sub_zone);
        %         mat_Ux_real_sig_sub_zone_affichage = F_Ux_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Ux_imag_sig_sub_zone_affichage = F_Ux_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uy_real_sig_sub_zone_affichage = F_Uy_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uy_imag_sig_sub_zone_affichage = F_Uy_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uz_real_sig_sub_zone_affichage = F_Uz_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uz_imag_sig_sub_zone_affichage = F_Uz_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         coupe_x_sig = [min(vec_grille_x_sig_sub_zone) mean(vec_grille_x_sig_sub_zone) max(vec_grille_x_sig_sub_zone)];
        %         coupe_y_sig = max(vec_grille_y_sig_sub_zone);
        %         coupe_z_sig = [min(vec_grille_z_sig_sub_zone) mean(vec_grille_z_sig_sub_zone)];
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Ux projete), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Ux projete), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uy projete), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uy projete), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uz projete), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uz projete), (m)');
        %         clear vec_R_U vec_R_Ux vec_R_Uy vec_R_Uz mat_pos_maillage_sig_sub_zone vec_x_sig_sub_zone vec_y_sig_sub_zone vec_z_sig_sub_zone F_Ux_real_sig F_Ux_imag_sig F_Uy_real_sig F_Uy_imag_sig F_Uz_real_sig F_Uz_imag_sig vec_grille_x_sig_sub_zone vec_grille_y_sig_sub_zone vec_grille_z_sig_sub_zone mat_x_sig_sub_zone mat_y_sig_sub_zone mat_z_sig_sub_zone mat_Ux_real_sig_sub_zone_affichage mat_Ux_imag_sig_sub_zone_affichage mat_Uy_real_sig_sub_zone_affichage mat_Uy_imag_sig_sub_zone_affichage mat_Uz_real_sig_sub_zone_affichage mat_Uz_imag_sig_sub_zone_affichage coupe_x_sig coupe_y_sig coupe_z_sig;
        % t_fin = cputime;
        %         disp(['                ' num2str(t_fin-t_ini) ' s']);
        
        % determination de la raideur globale
        %         disp('            raideur globale');
        % t_ini = cputime;
        K_tilde = K-omega_mec^2*M; % (elastodynamique)
        [vec_i_K_tilde,vec_j_K_tilde,vec_s_K_tilde] = find(K_tilde);
        [vec_i_T,vec_j_T,vec_s_T] = find(T);
        [vec_i_D,vec_j_D,vec_s_D] = find(D);
        
        kappa_critique = max(abs(vec_s_K_tilde))/max(abs(vec_s_D));
        %  kappa = kappa_critique/1000;
        %  kappa = kappa_critique;
        %  kappa = kappa_critique*1000;
        if ( strcmp(type_identification,'FEMU_DFC') == 1 )
        elseif ( strcmp(type_identification,'FEMU_AFC') == 1 )
            kappa = -1;
            vec_i_global = [vec_i_K_tilde'                   , (vec_j_K_tilde+size(K_tilde,1))' , (vec_i_D+size(K_tilde,1))'];
            vec_j_global = [(vec_j_K_tilde+size(K_tilde,2))' , vec_i_K_tilde'                   , (vec_j_D+size(K_tilde,2))'];
            vec_s_global = [vec_s_K_tilde.'                  , conj(vec_s_K_tilde).'            , (-kappa*vec_s_D.')];
            K_global = sparse(vec_i_global,vec_j_global,vec_s_global,2*size(K_tilde,1),2*size(K_tilde,2));
            clear vec_i_K_tilde vec_j_K_tilde vec_s_K_tilde;
            clear vec_i_T vec_j_T vec_s_T;
            clear vec_i_D vec_j_D vec_s_D;
            clear vec_i_global vec_j_global vec_s_global;
            %   clear K M T K_tilde D;
            vec_F_global = zeros(size(K_global,1),1);
            vec_F_global(1:size(K_tilde,1)) = vec_F;
            vec_F_global(size(K_tilde,1)+1:end) = -kappa*vec_R;
            % MAJ du systeme sur K_global pour tenir compte des CL
            %            [Ks_global,Fs_global,U_impose_global,vec_n_DDL_conserves_global,vec_n_DDL_supprimes_global] = maj_matrices(K_global,vec_F_global,liste_DL_bloque);
            %  figure;spy(K_global);title('structure K global');
        elseif ( strcmp(type_identification,'MERC') == 1 )
            %             disp(['                valeur a donner a "kappa" pour avoir des poids equivalents sur "K_tilde" et "D" : ' num2str(kappa_critique)]);
            %             disp(['                valeur de kappa : ' num2str(kappa)]);
            vec_i_global = [vec_i_T'  , vec_i_K_tilde'                   , (vec_j_K_tilde+size(K_tilde,1))' , (vec_i_D+size(K_tilde,1))'];
            vec_j_global = [vec_j_T'  , (vec_j_K_tilde+size(K_tilde,2))' , vec_i_K_tilde'                   , (vec_j_D+size(K_tilde,2))'];
            vec_s_global = [vec_s_T.' , vec_s_K_tilde.'                  , conj(vec_s_K_tilde).'            , (-kappa(idx)*vec_s_D.')];
            K_global = sparse(vec_i_global,vec_j_global,vec_s_global,2*size(K_tilde,1),2*size(K_tilde,2));
            clear vec_i_K_tilde vec_j_K_tilde vec_s_K_tilde;
            clear vec_i_T vec_j_T vec_s_T;
            clear vec_i_D vec_j_D vec_s_D;
            clear vec_i_global vec_j_global vec_s_global;
            %             clear K M T K_tilde D;
            vec_F_global = zeros(size(K_global,1),1);
            vec_F_global(1:size(K_tilde,1)) = vec_F;
            vec_F_global(size(K_tilde,1)+1:end) = -kappa(idx)*vec_R;
            %  figure;spy(K_global);title('structure K global');
            % MAJ du systeme sur K_global pour tenir compte des CL
            % par substitution
            [Ks_global,Fs_global,U_impose_global,vec_n_DDL_conserves_global,vec_n_DDL_supprimes_global] = maj_matrices(K_global,vec_F_global,liste_DL_bloque);
            % % par des multiplicateurs de Lagrange
            %             [vec_i_global,vec_j_global,vec_s_global] = find ( K_global );
            %             n_deb_multiplicateur = 0;
            %             nb_multiplicateurs = 0;
            %             for n_CL = 1:length(liste_DL_bloque)
            %                 type_DL_bloque = liste_DL_bloque{n_CL}.type;
            %                 vec_n_DL_bloque = liste_DL_bloque{n_CL}.vec_n_DL_bloque;
            %                 vec_val_DL_bloque = liste_DL_bloque{n_CL}.vec_val_DL_bloque;
            %                 vec_n_DL_bloque_matrice = n_deb_multiplicateur+(1:length(vec_n_DL_bloque));
            %                 vec_i_global = [vec_i_global;vec_n_DL_bloque'];
            %                 vec_j_global = [vec_j_global;(2*size(K_tilde,1)+vec_n_DL_bloque_matrice)'];
            %                 vec_s_global = [vec_s_global;ones(size(vec_n_DL_bloque))'];
            %                 vec_i_global = [vec_i_global;(2*size(K_tilde,1)+vec_n_DL_bloque_matrice)'];
            %                 vec_j_global = [vec_j_global;vec_n_DL_bloque'];
            %                 vec_s_global = [vec_s_global;ones(size(vec_n_DL_bloque))'];
            %                 vec_F_global = [vec_F_global;vec_val_DL_bloque.'];
            %                 n_deb_multiplicateur = n_deb_multiplicateur+length(vec_n_DL_bloque);
            %                 nb_multiplicateurs = nb_multiplicateurs+length(vec_n_DL_bloque);
            %             end
            %             K_global = sparse(vec_i_global,vec_j_global,vec_s_global,2*size(K_tilde,1)+nb_multiplicateurs,2*size(K_tilde,2)+nb_multiplicateurs);
            %             Ks_global = K_global;
            %             Fs_global = vec_F_global;
            %             U_impose_global = vec_F_global((2*size(K_tilde,1)+1):end);
            %             vec_n_DDL_conserves_global = 1:2*size(K_tilde,1);
            %             vec_n_DDL_supprimes_global = [];
        end
        
        % t_fin = cputime;
        %         disp(['                ' num2str(t_fin-t_ini) ' s']);
        
        % TEST POUR VOIR SI LES IMPLEMENTATIONS DES MATRICES SONT CORRECTES =>  CALCUL DES (U) en CL imposees en deplacement
        % %          MAJ du systeme sur K_tilde pour tenir compte des CL
        %         vec_x_sig = mat_pos_maillage_sig(:,1);
        %         vec_y_sig = mat_pos_maillage_sig(:,2);
        %         vec_z_sig = mat_pos_maillage_sig(:,3);
        %         vec_x_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,1)),1,[]);
        %         vec_y_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,2)),1,[]);
        %         vec_z_mes = reshape(squeeze(mat_X_mes_3D(:,:,:,3)),1,[]);
        %         vec_ux_mes = reshape(squeeze(mat_U_mes_3D(:,:,:,1)),1,[]);
        %         vec_uy_mes = reshape(squeeze(mat_U_mes_3D(:,:,:,2)),1,[]);
        %         vec_uz_mes = reshape(squeeze(mat_U_mes_3D(:,:,:,3)),1,[]);
        %         vec_n_a_conserver = find ( ~isnan(vec_x_mes) & ~isnan() & ~isnan(vec_z_mes) );
        %         vec_x_mes = vec_x_mes(vec_n_a_conserver)';
        %         vec_y_mes = vec_y_mes(vec_n_a_conserver)';
        %         vec_z_mes = vec_z_mes(vec_n_a_conserver)';
        %         vec_ux_mes = vec_ux_mes(vec_n_a_conserver)';
        %         vec_uy_mes = vec_uy_mes(vec_n_a_conserver)';
        %         vec_uz_mes = vec_uz_mes(vec_n_a_conserver)';
        % %           F_Ux = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_ux_mes,'linear','none');
        % %           F_Uy = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_uy_mes,'linear','none');
        % %           F_Uz = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_uz_mes,'linear','none');
        % %           F_Ux = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_ux_mes,'linear','linear');
        % %           F_Uy = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_uy_mes,'linear','linear');
        % %           F_Uz = scatteredInterpolant(vec_x_mes,vec_y_mes,vec_z_mes,vec_uz_mes,'linear','linear');
        %         vec_ux_sig = F_Ux(vec_x_sig,vec_y_sig,vec_z_sig);
        %         vec_uy_sig = F_Uy(vec_x_sig,vec_y_sig,vec_z_sig);
        %         vec_uz_sig = F_Uz(vec_x_sig,vec_y_sig,vec_z_sig);
        %         mat_U_sig_3D = [vec_ux_sig.';vec_uy_sig.';vec_uz_sig.'].';
        %         clear vec_x_mes vec_y_mes vec_z_mes vec_ux_mes vec_uy_mes vec_uz_mes vec_n_a_conserver vec_x_sig vec_y_sig vec_z_sig vec_ux_sig vec_uy_sig vec_uz_sig F_Ux F_Uy F_Uz;
        %         liste_DL_bloque_U_impose = cell(1,nb_DDL_par_noeud);
        %         for n_DL_imp = 1:nb_DDL_par_noeud
        %            vec_n_DL_bloque_U_impose = (vec_n_noeuds_frontieres_local-1)*nb_DDL_par_noeud+n_DL_imp;
        %            vec_val_DL_bloque_U_impose = mat_U_sig_3D(vec_noeuds_frontieres,n_DL_imp)';
        %            liste_DL_bloque_U_impose{n_DL_imp} = struct('type','U','vec_n_DL_bloque',vec_n_DL_bloque_U_impose,'vec_val_DL_bloque',vec_val_DL_bloque_U_impose,'vec_n_noeud',vec_n_noeuds_frontieres_local);
        %         end
        %         vec_F_U_impose = zeros(size(K_tilde,1),1);
        %         [Ks_U_impose,Fs_U_impose,U_impose_local,vec_n_DDL_conserves_U_impose,vec_n_DDL_supprimes_U_impose] = maj_matrices(K_tilde,vec_F_U_impose,liste_DL_bloque_U_impose);
        %         Us_local = Ks_U_impose\Fs_U_impose;
        %         U_local = CL_assemblage(Us_local,U_impose_local,vec_n_DDL_conserves_U_impose,vec_n_DDL_supprimes_U_impose);
        %         Ux_local = U_local(1:nb_DDL_par_noeud:end);
        %         Uy_local = U_local(2:nb_DDL_par_noeud:end);
        %         Uz_local = U_local(3:nb_DDL_par_noeud:end);
        %         figure;hold on;plot(real(Ux_local),'r');plot(real(Uy_local),'g');plot(real(Uz_local),'b');title('real(U) local');legend('Ux','Uy','Uz');
        %         figure;hold on;plot(imag(Ux_local),'r');plot(imag(Uy_local),'g');plot(imag(Uz_local),'b');title('imag(U) local');legend('Ux','Uy','Uz');
        % %         % AFFICHAGE DES CHAMPS CALCULES POUR LA SUB-ZONE COURANTE
        % %         % deplacements mesures
        %         vec_Ux_mes_sub_zone = mat_U_mes(1,vec_n_noeuds_mes_sub_zone);
        %         vec_Uy_mes_sub_zone = mat_U_mes(2,vec_n_noeuds_mes_sub_zone);
        %         vec_Uz_mes_sub_zone = mat_U_mes(3,vec_n_noeuds_mes_sub_zone);
        %         vec_i_mes_sub_zone = vec_i_mes(vec_n_noeuds_mes_sub_zone);
        %         vec_j_mes_sub_zone = vec_j_mes(vec_n_noeuds_mes_sub_zone);
        %         vec_k_mes_sub_zone = vec_k_mes(vec_n_noeuds_mes_sub_zone);
        %         vec_np_mes_sub_zone_global = vec_i_mes_sub_zone+ni_mes*((vec_j_mes_sub_zone-1)+nj_mes*(vec_k_mes_sub_zone-1));
        %         vec_i_mes_sub_zone = vec_i_mes_sub_zone-min(vec_i_mes_sub_zone)+1;
        %         vec_j_mes_sub_zone = vec_j_mes_sub_zone-min(vec_j_mes_sub_zone)+1;
        %         vec_k_mes_sub_zone = vec_k_mes_sub_zone-min(vec_k_mes_sub_zone)+1;
        %         ni_mes_sub_zone = max(vec_i_mes_sub_zone);
        %         nj_mes_sub_zone = max(vec_j_mes_sub_zone);
        %         nk_mes_sub_zone = max(vec_k_mes_sub_zone);
        %         vec_np_mes_sub_zone_local = vec_i_mes_sub_zone+ni_mes_sub_zone*((vec_j_mes_sub_zone-1)+nj_mes_sub_zone*(vec_k_mes_sub_zone-1));
        %         mat_Ux_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Ux_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Uy_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Uy_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Uz_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Uz_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Ux_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Ux_mes_sub_zone);
        %         mat_Ux_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Ux_mes_sub_zone);
        %         mat_Uy_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Uy_mes_sub_zone);
        %         mat_Uy_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Uy_mes_sub_zone);
        %         mat_Uz_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Uz_mes_sub_zone);
        %         mat_Uz_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Uz_mes_sub_zone);
        %         vec_x_grille_mes_sub_zone = struct_grille_mes.x_min+struct_grille_mes.dx*((min(vec_i_mes(vec_np_mes_sub_zone_global)):max(vec_i_mes(vec_np_mes_sub_zone_global)))-1);
        %         vec_y_grille_mes_sub_zone = struct_grille_mes.y_min+struct_grille_mes.dy*((min(vec_j_mes(vec_np_mes_sub_zone_global)):max(vec_j_mes(vec_np_mes_sub_zone_global)))-1);
        %         vec_z_grille_mes_sub_zone = struct_grille_mes.z_min+struct_grille_mes.dz*((min(vec_k_mes(vec_np_mes_sub_zone_global)):max(vec_k_mes(vec_np_mes_sub_zone_global)))-1);
        %         [mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone] = meshgrid(vec_y_grille_mes_sub_zone,vec_x_grille_mes_sub_zone,vec_z_grille_mes_sub_zone);
        %         coupe_x_mes = [min(vec_x_grille_mes_sub_zone) mean(vec_x_grille_mes_sub_zone) max(vec_x_grille_mes_sub_zone)];
        %         coupe_y_mes = max(vec_y_grille_mes_sub_zone);
        %         coupe_z_mes = [min(vec_z_grille_mes_sub_zone) mean(vec_z_grille_mes_sub_zone)];
        %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Ux_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Ux mes), (m)');
        %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Ux_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Ux mes), (m)');
        %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uy_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Uy mes), (m)');
        %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uy_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Uy mes), (m)');
        %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uz_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Uz mes), (m)');
        %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uz_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Uz mes), (m)');
        % %         champs calcules
        %         vec_Ux_sub_zone = Ux;
        %         vec_Uy_sub_zone = Uy;
        %         vec_Uz_sub_zone = Uz;
        % %         mat_pos_maillage_sig_sub_zone = mat_pos_maillage_sig(vec_correspondance_n_noeud_sig_local_n_noeud_sig_global,:);
        %         mat_pos_maillage_sig_sub_zone = mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,:);
        %         vec_x_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,1);
        %         vec_y_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,2);
        %         vec_z_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,3);
        %         F_Ux_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Ux_sub_zone),'linear','none');
        %         F_Ux_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Ux_sub_zone),'linear','none');
        %         F_Uy_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Uy_sub_zone),'linear','none');
        %         F_Uy_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Uy_sub_zone),'linear','none');
        %         F_Uz_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Uz_sub_zone),'linear','none');
        %         F_Uz_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Uz_sub_zone),'linear','none');
        %         vec_grille_x_sig_sub_zone = unique(sort(vec_x_sig_sub_zone));
        %         vec_grille_y_sig_sub_zone = unique(sort(vec_y_sig_sub_zone));
        %         vec_grille_z_sig_sub_zone = unique(sort(vec_z_sig_sub_zone));
        %         [mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone] = meshgrid(vec_grille_x_sig_sub_zone,vec_grille_y_sig_sub_zone,vec_grille_z_sig_sub_zone);
        %         mat_Ux_real_sig_sub_zone_affichage = F_Ux_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Ux_imag_sig_sub_zone_affichage = F_Ux_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uy_real_sig_sub_zone_affichage = F_Uy_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uy_imag_sig_sub_zone_affichage = F_Uy_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uz_real_sig_sub_zone_affichage = F_Uz_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uz_imag_sig_sub_zone_affichage = F_Uz_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         coupe_x_sig = [min(vec_grille_x_sig_sub_zone) mean(vec_grille_x_sig_sub_zone) max(vec_grille_x_sig_sub_zone)];
        %         coupe_y_sig = max(vec_grille_y_sig_sub_zone);
        %         coupe_z_sig = [min(vec_grille_z_sig_sub_zone) mean(vec_grille_z_sig_sub_zone)];
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Ux), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Ux), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uy), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uy), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uz), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uz), (m)');
        
        % resolution et reecriture de U (assemblage du vecteur resultat)
        %         disp('            resolution');
        % t_ini = cputime;
        nb_DDL_K = size(K,1);
        Us_global = Ks_global\Fs_global;
        % reintroduction des CL
        % par substitution
        U_global = CL_assemblage(Us_global,U_impose_global,vec_n_DDL_conserves_global,vec_n_DDL_supprimes_global);
        % % par multiplicateurs de Lagrange
        %         U_global = Us_global(1:2*size(K_tilde,1));
        % definition des vecteurs U et W
        Wx = U_global(1:nb_DDL_par_noeud:nb_DDL_K);
        Wy = U_global(2:nb_DDL_par_noeud:nb_DDL_K);
        Wz = U_global(3:nb_DDL_par_noeud:nb_DDL_K);
        Ux = U_global((nb_DDL_K+1):nb_DDL_par_noeud:end);
        Uy = U_global((nb_DDL_K+2):nb_DDL_par_noeud:end);
        Uz = U_global((nb_DDL_K+3):nb_DDL_par_noeud:end);
        %         figure;hold on;plot(real(Ux),'r');plot(real(Uy),'g');plot(real(Uz),'b');title('real(U)');legend('Ux','Uy','Uz');title('Re(U calc))');
        %         figure;hold on;plot(imag(Ux),'r');plot(imag(Uy),'g');plot(imag(Uz),'b');title('imag(U)');legend('Ux','Uy','Uz');title('Im(U calc))');
        %         figure;hold on;plot(real(Wx),'r');plot(real(Wy),'g');plot(real(Wz),'b');title('real(W)');legend('Wx','Wy','Wz');title('Re(W calc))');
        %         figure;hold on;plot(imag(Wx),'r');plot(imag(Wy),'g');plot(imag(Wz),'b');title('imag(W)');legend('Wx','Wy','Wz');title('Im(W calc))');
        % %         figure;hold on;plot(real(Ux(vec_n_noeuds_frontieres_local)),'-r');plot(real(Uy(vec_n_noeuds_frontieres_local)),'-g');plot(real(Uz(vec_n_noeuds_frontieres_local)),'-b');grid;legend('Ux','Uy','Uz');title('Re(U calc frontiere))');
        % %         figure;hold on;plot(imag(Ux(vec_n_noeuds_frontieres_local)),'-r');plot(imag(Uy(vec_n_noeuds_frontieres_local)),'-g');plot(imag(Uz(vec_n_noeuds_frontieres_local)),'-b');grid;legend('Ux','Uy','Uz');title('Im(U calc frontiere))');
        % %         figure;hold on;plot(real(Wx(vec_n_noeuds_frontieres_local)),'-r');plot(real(Wy(vec_n_noeuds_frontieres_local)),'-g');plot(real(Wz(vec_n_noeuds_frontieres_local)),'-b');grid;legend('Wx','Wy','Wz');title('Re(W calc frontiere))');
        % %         figure;hold on;plot(imag(Wx(vec_n_noeuds_frontieres_local)),'-r');plot(imag(Wy(vec_n_noeuds_frontieres_local)),'-g');plot(imag(Wz(vec_n_noeuds_frontieres_local)),'-b');grid;legend('Wx','Wy','Wz');title('Im(W calc frontiere))');
        % t_fin = cputime;
        %         disp(['                ' num2str(t_fin-t_ini) ' s']);
        
        % AFFICHAGE DES CHAMPS CALCULES POUR LA SUB-ZONE COURANTE
        % % deplacements mesures
        %         vec_Ux_mes_sub_zone = mat_U_mes(1,vec_n_noeuds_mes_sub_zone);
        %         vec_Uy_mes_sub_zone = mat_U_mes(2,vec_n_noeuds_mes_sub_zone);
        %         vec_Uz_mes_sub_zone = mat_U_mes(3,vec_n_noeuds_mes_sub_zone);
        %         vec_i_mes_sub_zone = vec_i_mes(vec_n_noeuds_mes_sub_zone);
        %         vec_j_mes_sub_zone = vec_j_mes(vec_n_noeuds_mes_sub_zone);
        %         vec_k_mes_sub_zone = vec_k_mes(vec_n_noeuds_mes_sub_zone);
        %         vec_np_mes_sub_zone_global = vec_i_mes_sub_zone+ni_mes*((vec_j_mes_sub_zone-1)+nj_mes*(vec_k_mes_sub_zone-1));
        %         vec_i_mes_sub_zone = vec_i_mes_sub_zone-min(vec_i_mes_sub_zone)+1;
        %         vec_j_mes_sub_zone = vec_j_mes_sub_zone-min(vec_j_mes_sub_zone)+1;
        %         vec_k_mes_sub_zone = vec_k_mes_sub_zone-min(vec_k_mes_sub_zone)+1;
        %         ni_mes_sub_zone = max(vec_i_mes_sub_zone);
        %         nj_mes_sub_zone = max(vec_j_mes_sub_zone);
        %         nk_mes_sub_zone = max(vec_k_mes_sub_zone);
        %         vec_np_mes_sub_zone_local = vec_i_mes_sub_zone+ni_mes_sub_zone*((vec_j_mes_sub_zone-1)+nj_mes_sub_zone*(vec_k_mes_sub_zone-1));
        %         mat_Ux_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Ux_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Uy_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Uy_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Uz_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Uz_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
        %         mat_Ux_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Ux_mes_sub_zone);
        %         mat_Ux_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Ux_mes_sub_zone);
        %         mat_Uy_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Uy_mes_sub_zone);
        %         mat_Uy_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Uy_mes_sub_zone);
        %         mat_Uz_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Uz_mes_sub_zone);
        %         mat_Uz_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Uz_mes_sub_zone);
        %         vec_x_grille_mes_sub_zone = struct_grille_mes.x_min+struct_grille_mes.dx*((min(vec_i_mes(vec_np_mes_sub_zone_global)):max(vec_i_mes(vec_np_mes_sub_zone_global)))-1);
        %         vec_y_grille_mes_sub_zone = struct_grille_mes.y_min+struct_grille_mes.dy*((min(vec_j_mes(vec_np_mes_sub_zone_global)):max(vec_j_mes(vec_np_mes_sub_zone_global)))-1);
        %         vec_z_grille_mes_sub_zone = struct_grille_mes.z_min+struct_grille_mes.dz*((min(vec_k_mes(vec_np_mes_sub_zone_global)):max(vec_k_mes(vec_np_mes_sub_zone_global)))-1);
        %         [mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone] = meshgrid(vec_y_grille_mes_sub_zone,vec_x_grille_mes_sub_zone,vec_z_grille_mes_sub_zone);
        %         coupe_x_mes = [min(vec_x_grille_mes_sub_zone) mean(vec_x_grille_mes_sub_zone) max(vec_x_grille_mes_sub_zone)];
        %         coupe_y_mes = max(vec_y_grille_mes_sub_zone);
        %         coupe_z_mes = [min(vec_z_grille_mes_sub_zone) mean(vec_z_grille_mes_sub_zone)];
        % %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Ux_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Ux mes), (m)');
        % %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Ux_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Ux mes), (m)');
        % %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uy_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Uy mes), (m)');
        % %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uy_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Uy mes), (m)');
        % %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uz_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Uz mes), (m)');
        % %         figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uz_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Uz mes), (m)');
        % % champs calcules
        %         vec_Ux_sub_zone = Ux;
        %         vec_Uy_sub_zone = Uy;
        %         vec_Uz_sub_zone = Uz;
        %         vec_Wx_sub_zone = Wx;
        %         vec_Wy_sub_zone = Wy;
        %         vec_Wz_sub_zone = Wz;
        %         mat_pos_maillage_sig_sub_zone = mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,:);
        %         vec_x_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,1);
        %         vec_y_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,2);
        %         vec_z_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,3);
        %         F_Ux_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Ux_sub_zone),'linear','none');
        %         F_Ux_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Ux_sub_zone),'linear','none');
        %         F_Uy_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Uy_sub_zone),'linear','none');
        %         F_Uy_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Uy_sub_zone),'linear','none');
        %         F_Uz_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Uz_sub_zone),'linear','none');
        %         F_Uz_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Uz_sub_zone),'linear','none');
        %         F_Wx_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Wx_sub_zone),'linear','none');
        %         F_Wx_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Wx_sub_zone),'linear','none');
        %         F_Wy_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Wy_sub_zone),'linear','none');
        %         F_Wy_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Wy_sub_zone),'linear','none');
        %         F_Wz_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Wz_sub_zone),'linear','none');
        %         F_Wz_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Wz_sub_zone),'linear','none');
        %         vec_grille_x_sig_sub_zone = unique(sort(vec_x_sig_sub_zone));
        %         vec_grille_y_sig_sub_zone = unique(sort(vec_y_sig_sub_zone));
        %         vec_grille_z_sig_sub_zone = unique(sort(vec_z_sig_sub_zone));
        %         [mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone] = meshgrid(vec_grille_x_sig_sub_zone,vec_grille_y_sig_sub_zone,vec_grille_z_sig_sub_zone);
        %         mat_Ux_real_sig_sub_zone_affichage = F_Ux_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Ux_imag_sig_sub_zone_affichage = F_Ux_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uy_real_sig_sub_zone_affichage = F_Uy_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uy_imag_sig_sub_zone_affichage = F_Uy_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uz_real_sig_sub_zone_affichage = F_Uz_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Uz_imag_sig_sub_zone_affichage = F_Uz_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Wx_real_sig_sub_zone_affichage = F_Wx_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Wx_imag_sig_sub_zone_affichage = F_Wx_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Wy_real_sig_sub_zone_affichage = F_Wy_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Wy_imag_sig_sub_zone_affichage = F_Wy_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Wz_real_sig_sub_zone_affichage = F_Wz_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         mat_Wz_imag_sig_sub_zone_affichage = F_Wz_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
        %         coupe_x_sig = [min(vec_grille_x_sig_sub_zone) mean(vec_grille_x_sig_sub_zone) max(vec_grille_x_sig_sub_zone)];
        %         coupe_y_sig = max(vec_grille_y_sig_sub_zone);
        %         coupe_z_sig = [min(vec_grille_z_sig_sub_zone) mean(vec_grille_z_sig_sub_zone)];
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Ux), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Ux), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uy), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uy), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uz), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uz), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wx_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Wx), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wx_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Wx), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wy_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Wy), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wy_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Wy), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wz_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Wz), (m)');
        %         figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wz_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Wz), (m)');
        
        % verification de la numerotation a 3 indices
        % %         mat_V = nan(3,5,7);
        % %         n = 0;
        % %         vec_i = zeros(1,size(mat_V,1)*size(mat_V,2)*size(mat_V,3));
        % %         vec_j = zeros(1,size(mat_V,1)*size(mat_V,2)*size(mat_V,3));
        % %         vec_k = zeros(1,size(mat_V,1)*size(mat_V,2)*size(mat_V,3));
        % %         vec_V = zeros(1,size(mat_V,1)*size(mat_V,2)*size(mat_V,3));
        % %         for i = 1:size(mat_V,1)
        % %          for j = 1:size(mat_V,2)
        % %           for k = 1:size(mat_V,3)
        % %            n = n+1;
        % %            mat_V(i,j,k) = n^2;
        % %            vec_i(n) = i;
        % %            vec_j(n) = j;
        % %            vec_k(n) = k;
        % %            vec_V(n) = mat_V(i,j,k);
        % %           end
        % %          end
        % %         end
        % %         vec_n = vec_i+size(mat_V,1)*((vec_j-1)+size(mat_V,2)*(vec_k-1));
        % %         mat_V_tilde = nan(size(mat_V));
        % %         mat_V_tilde(vec_n) = vec_V;
        % %         k = 3;figure;subplot(1,2,1);imagesc(squeeze(mat_V(:,:,k)));colorbar;subplot(1,2,2);imagesc(squeeze(mat_V_tilde(:,:,k)));colorbar;
        % %         for k = 1:size(mat_V,3),figure;imagesc(squeeze(mat_V(:,:,k)-mat_V_tilde(:,:,k)));colorbar;end;
        
        %         disp('            mise a jour des proprietes mecaniques');
        % t_ini = cputime;
        if ( strcmp(type_identification,'FEMU_DFC') == 1 )
            
        elseif ( strcmp(type_identification,'FEMU_AFC') == 1 )
            % calcul du gradient
            mat_gradient_A = zeros(size(d_K_d_p,1),size(d_K_d_p,3));
            W_local = U_global((1:nb_DDL_K));
            U_local = U_global(nb_DDL_K+(1:nb_DDL_K));
            for l = 1:size(d_K_d_p,3)
                mat_gradient_A(:,l) = sum(squeeze(d_K_d_p(:,:,l)).*(ones(size(d_K_d_p,1),1)*(U_local.*W_local).'),2);
            end
            % minimisation de la fonction cout en (u-u_m) par rapport aux proprietes materielles
            %   => algorithme du gradient conjugue
            mat_d = -mat_gradient_A;
            
            %%%%%%%%%%%%%%%%%
            %%% A CONTINUER ...
            %%%%%%%%%%%%%%%%%
            
        elseif( ( strcmp(type_identification,'MERC') == 1 ) )
            % determination des numeros des parametres "lambda" et "mu"
            n_param_lambda = 0;
            test_param = false;
            while ( ~test_param )
                n_param_lambda = n_param_lambda+1;
                if ( strcmp(struct_param_comportement_a_identifier.liste_parametres_comportement{struct_param_comportement_a_identifier.vec_numeros_parametres_comportement(n_param_lambda)},'lambda') == 1 )
                    test_param = true;
                end
            end
            clear test_param;
            n_param_mu = 0;
            test_param = false;
            while ( ~test_param )
                n_param_mu = n_param_mu+1;
                if ( strcmp(struct_param_comportement_a_identifier.liste_parametres_comportement{struct_param_comportement_a_identifier.vec_numeros_parametres_comportement(n_param_mu)},'mu') == 1 )
                    test_param = true;
                end
            end
            clear test_param;
            
            % boucle sur tous les elements de contraintes et tous les points de Gauss pour identifier les proprietes
            vec_x_G = nan(1,length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds);
            vec_y_G = nan(1,length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds);
            vec_z_G = nan(1,length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds);
            vec_tr_epsilon_U_G = nan(1,length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds);
            vec_tr_sigma_G = nan(1,length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds);
            vec_deviateur_epsilon_U_deviateur_epsilon_conjugue_U_G = nan(1,length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds);
            vec_deviateur_sigma_deviateur_epsilon_conjugue_U_G = nan(1,length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds);
            mat_coef_mat = zeros(length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds*nb_parametres_comportement_a_identifier,nb_parametres_comportement_a_identifier*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local));
            vec_second_membre_mat = zeros(length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds*nb_parametres_comportement_a_identifier,1);
            vec_n_elem_sig_systeme = zeros(1,length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds*nb_parametres_comportement_a_identifier);
            vec_n_elem_pha_systeme = zeros(1,length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds*nb_parametres_comportement_a_identifier);
            vec_n_param_systeme = zeros(1,length(liste_elem_sig_sub_zone)*elem_sig_ref.nb_noeuds*nb_parametres_comportement_a_identifier);
            
            n_G_local = 0;
            for n_sig = 1:length(liste_elem_sig_sub_zone)
                elem_sig = liste_elem_sig_sub_zone{n_sig};
                elem_ref_sig = liste_elem_ref{elem_sig.n_elem_ref};
                vec_U_local = nan(3*length(elem_sig.vec_n_noeuds),1);
                vec_U_local(1:3:end) = Ux(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)');
                vec_U_local(2:3:end) = Uy(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)');
                vec_U_local(3:3:end) = Uz(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)');
                vec_conjugue_U_local = conj(vec_U_local);
                vec_W_local = nan(3*length(elem_sig.vec_n_noeuds),1);
                vec_W_local(1:3:end) = Wx(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)');
                vec_W_local(2:3:end) = Wy(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)');
                vec_W_local(3:3:end) = Wz(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)');
                vec_poids_Gauss = liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.poids_Gauss;
                
                for n_G = 1:size(elem_sig.mat_pos_pha_G_K,2)
                    n_pha = elem_sig.vec_n_pha_G_K(n_G);
                    vec_pos_G_ref = elem_sig.mat_pos_pha_G_K(:,n_G);
                    elem_pha = liste_elem_pha{n_pha};
                    vec_n_noeuds_pha = elem_pha.vec_n_noeuds;
                    elem_ref_pha = liste_elem_ref{elem_pha.n_elem_ref};
                    vec_f_Nf_pha = squeeze(elem_ref_pha.f_Nf(vec_pos_G_ref(1),vec_pos_G_ref(2),vec_pos_G_ref(3)))';
                    % calcul de la deformation au point de Gauss de l'element de contrainte
                    mat_d_f_Nf_sig = squeeze(elem_ref_sig.d_f_Nf(elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,1),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,2),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,3)))';
                    Jaco = mat_d_f_Nf_sig*mat_pos_maillage_sig(elem_sig.vec_n_noeuds,:); % matrice 3x3. Produit du gradient fonction de forme associ? et coordonn?es des noeuds
                    J = det(Jaco); % d?terminant
                    % matrice du gradient des fonctions de formes dans la configuration reelle
                    % permettant de calculer le tenseur de deformation en notation de "Voigt modifiee" : (exx,eyy,ezz,2*exy,2*exz,2*eyz)
                    [Be] = calcul_Be(mat_d_f_Nf_sig,Jaco,nb_DDL_par_noeud);
                    vec_epsilon_U_Voigt = Be*vec_U_local;
                    vec_epsilon_conjugue_U_Voigt = Be*vec_conjugue_U_local;
                    vec_epsilon_W_Voigt = Be*vec_W_local;
                    % calcul des fonctions de forme sur les phases au point de Gauss de l'element de contrainte
                    A_visco = struct_param_comportement_a_identifier.f_C(struct_param_comportement_a_identifier.mat_param(struct_param_comportement_a_identifier.vec_numeros_parametres_comportement,elem_pha.vec_n_noeuds)*vec_f_Nf_pha);
                    A_elastique = struct_param_comportement_normalisation.f_C(struct_param_comportement_normalisation.mat_param(struct_param_comportement_normalisation.vec_numeros_parametres_comportement,elem_pha.vec_n_noeuds)*vec_f_Nf_pha);
                    % % calcul de sigma : sigma = C:epsilon[u] + P:epsilon[w]
                    vec_sigma_Voigt = A_visco*vec_epsilon_U_Voigt+A_elastique*vec_epsilon_W_Voigt;
                    % pour tester la formule d'identification, on peut calculer "sigma = A*epsilon[u]" et verifier qu'on obtient les memes modules que ceux utilises pour A
                    %                     vec_sigma_Voigt = A_visco*vec_epsilon_U_Voigt;
                    % calcul des quantites necessaires a la mise a jour des proprietes des phases
                    deviateur_epsilon_U = calcul_deviateur(vec_epsilon_U_Voigt./[1 1 1 2 2 2]');
                    deviateur_epsilon_conjugue_U = calcul_deviateur(vec_epsilon_conjugue_U_Voigt./[1 1 1 2 2 2]');
                    deviateur_sigma = calcul_deviateur(vec_sigma_Voigt);
                    tr_epsilon_U = vec_epsilon_U_Voigt(1)+vec_epsilon_U_Voigt(2)+vec_epsilon_U_Voigt(3);
                    tr_sigma = vec_sigma_Voigt(1)+vec_sigma_Voigt(2)+vec_sigma_Voigt(3);
                    deviateur_epsilon_U_deviateur_epsilon_conjugue_U = sum(sum(deviateur_epsilon_U.*deviateur_epsilon_conjugue_U));
                    deviateur_sigma_deviateur_epsilon_conjugue_U = sum(sum(deviateur_sigma.*deviateur_epsilon_conjugue_U));
                    deviateur_sigma_deviateur_sigma_conjugue = sum(sum(deviateur_sigma.*conj(deviateur_sigma)));
                    %                     deviateur_sigma_deviateur_epsilon_conjugue_U/(2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U)
                    
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
                    % equation sur lambda
                    %                     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*3*tr_epsilon_U*vec_f_Nf_pha; % lambda
                    %                     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*2*tr_epsilon_U*vec_f_Nf_pha; % mu
                    mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*3*tr_epsilon_U; % lambda
                    mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*2*tr_epsilon_U; % mu
                    vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = vec_poids_Gauss(n_G)*J*tr_sigma;
                    vec_n_elem_sig_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_sig;
                    vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_pha;
                    vec_n_param_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_param_lambda;
                    % equation sur mu
                    %                     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2,max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U*vec_f_Nf_pha; % mu
                    %                     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2,max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U; % mu
                    mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2,max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U; % mu
                    vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = vec_poids_Gauss(n_G)*J*deviateur_sigma_deviateur_epsilon_conjugue_U;
                    vec_n_elem_sig_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_sig;
                    vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_pha;
                    vec_n_param_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_param_mu;
                    % % methode (ii) ecriture des equations en (B,G)
                    % % equation sur B
                    % %                     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*3*tr_epsilon_U*vec_f_Nf_pha; % B
                    %                     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*3*tr_epsilon_U; % B
                    %                     vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = vec_poids_Gauss(n_G)*J*tr_sigma;
                    %                     vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_pha;
                    %                     vec_n_elem_sig_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_sig;
                    %                     vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_pha;
                    %                     vec_n_param_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_param_lambda;
                    % % equation sur G
                    % %                     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2,max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U*vec_f_Nf_pha; % G
                    %                     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2,max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U; % G
                    %                     vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = vec_poids_Gauss(n_G)*J*deviateur_sigma_deviateur_epsilon_conjugue_U;
                    %                     vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_pha;
                    %                     vec_n_elem_sig_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_sig;
                    %                     vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_pha;
                    %                     vec_n_param_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_param_mu;
                end
            end
            
            % on somme les equations par phase
            vec_n_pha_sub_zone = unique(vec_n_elem_pha_systeme(vec_n_elem_pha_systeme > 0));
            mat_coef_mat_tempo = zeros(2*length(vec_n_pha_sub_zone),size(mat_coef_mat,2));
            vec_second_membre_mat_tempo = zeros(2*length(vec_n_pha_sub_zone),1);
            vec_nb_eq_par_param_pha = zeros(2*length(vec_n_pha_sub_zone),1);
            for nn_pha = 1:length(vec_n_pha_sub_zone)
                n_pha = vec_n_pha_sub_zone(nn_pha);
                % equations en "lambda"
                ii_sel = find ( (vec_n_elem_pha_systeme == n_pha) & (vec_n_param_systeme == n_param_lambda) );
                mat_coef_mat_tempo((nn_pha-1)*2+1,:) = sum(mat_coef_mat(ii_sel,:),1);
                vec_second_membre_mat_tempo((nn_pha-1)*2+1,1) = sum(vec_second_membre_mat(ii_sel,:),1);
                vec_nb_eq_par_param_pha((nn_pha-1)*2+1,1) = length(ii_sel);
                % equations en "mu"
                ii_sel = find ( (vec_n_elem_pha_systeme == n_pha) & (vec_n_param_systeme == n_param_mu) );
                mat_coef_mat_tempo((nn_pha-1)*2+2,:) = sum(mat_coef_mat(ii_sel,:),1);
                vec_second_membre_mat_tempo((nn_pha-1)*2+2,1) = sum(vec_second_membre_mat(ii_sel,:),1);
                vec_nb_eq_par_param_pha((nn_pha-1)*2+2,1) = length(ii_sel);
            end
            mat_coef_mat = mat_coef_mat_tempo;
            vec_second_membre_mat = vec_second_membre_mat_tempo;
            %             mat_coef_mat_SAV = mat_coef_mat;
            %             vec_second_membre_mat_SAV = vec_second_membre_mat;
            
            % on reagence le systeme en fonction des parametres a identifier
            test_identification_lambda = false;
            test_identification_mu = false;
            n_param_lambda_ident = nan;
            n_param_mu_ident = nan;
            for n_param = 1:length(struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier)
                if ( strcmp(struct_param_comportement_a_identifier.liste_parametres_comportement{struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier(n_param)},'lambda') == 1 )
                    test_identification_lambda = true;
                    n_param_lambda_ident = n_param;
                end
                if ( strcmp(struct_param_comportement_a_identifier.liste_parametres_comportement{struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier(n_param)},'mu') == 1 )
                    test_identification_mu = true;
                    n_param_mu_ident = n_param;
                end
            end
            if ( ~test_identification_lambda ) % on n'identifie pas "lambda"
                vec_second_membre_mat = vec_second_membre_mat-mat_coef_mat(:,(1:nb_parametres_comportement_a_identifier*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)))*(struct_param_comportement_a_identifier.mat_param(n_param_lambda,vec_n_noeuds_pha_sub_zone)).';
                mat_coef_mat = mat_coef_mat(:,nb_parametres_comportement_a_identifier*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+(1:nb_parametres_comportement_a_identifier*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)));
                mat_coef_mat = mat_coef_mat(2:2:end,:);
                vec_second_membre_mat = vec_second_membre_mat(2:2:end);
            end
            if ( ~test_identification_mu ) % on n'identifie pas "mu"
                vec_second_membre_mat = vec_second_membre_mat-mat_coef_mat(:,nb_parametres_comportement_a_identifier*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+(1:nb_parametres_comportement_a_identifier*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)))*(struct_param_comportement_a_identifier.mat_param(n_param_mu,vec_n_noeuds_pha_sub_zone)).';
                mat_coef_mat = mat_coef_mat(:,(1:nb_parametres_comportement_a_identifier*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)));
                mat_coef_mat = mat_coef_mat(1:2:end,:);
                vec_second_membre_mat = vec_second_membre_mat(1:2:end);
            end
            %             vec_sol = mat_coef_mat\vec_second_membre_mat;
            mat_systeme = (mat_coef_mat'*mat_coef_mat);
            vec_systeme = (mat_coef_mat'*vec_second_membre_mat);
            vec_norme_mat_systeme = max(abs(mat_systeme),[],2);
            mat_systeme = mat_systeme./(vec_norme_mat_systeme*ones(1,size(mat_systeme,2)));
            vec_systeme = vec_systeme./vec_norme_mat_systeme;
            vec_sol = mat_systeme\vec_systeme;
            %             sum(vec_deviateur_sigma_deviateur_epsilon_conjugue_U_G)/sum((2*vec_deviateur_epsilon_U_deviateur_epsilon_conjugue_U_G))
            %             vec_sol = real(vec_sol)+1i*imag(struct_param_comportement_a_identifier.mat_param(n_param_mu,vec_n_noeuds_pha_sub_zone));
            %             figure;plot(abs(vec_sol));grid;
            %             figure;plot(abs(mat_systeme*vec_sol-vec_systeme));grid;
            %             figure;hold on;plot(real(Ux),'-r');plot(real(Wx),'-b');grid;title(['iteration ' int2str(n_iter_LDC) ', mu = ' num2str(real(vec_sol(1))) '+i*' num2str(imag(vec_sol(1)))]);
            %             figure;plot(abs(U_global(nb_DDL_K+(1:nb_DDL_K))-(N_mes'*vec_U_mes)));
            %             figure;plot(abs(U_global(nb_DDL_K+(1:nb_DDL_K))));
            %             figure;hold on;plot(abs(T*U_global(1:nb_DDL_K)),'-r');plot(abs(K_tilde*U_global(nb_DDL_K+(1:nb_DDL_K))),'-b');
            %             figure;hold on;plot(abs(K_tilde'*U_global(1:nb_DDL_K)),'-r');plot(abs(kappa*(D*U_global(nb_DDL_K+(1:nb_DDL_K))-vec_R)),'-b');
            % test de l'admissibilite des valeurs identifiees par rapport aux bornes d'identification
            if ( test_identification_lambda && test_identification_mu )
                vec_lambda_noeuds_pha_identifies = vec_sol((1:max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)));
                val_borne_min_lambda = struct_param_comportement_a_identifier.vec_borne_min_param_identification(n_param_lambda_ident);
                val_borne_max_lambda = struct_param_comportement_a_identifier.vec_borne_max_param_identification(n_param_lambda_ident);
                vec_n_param_min_lambda_reel = find ( real(vec_lambda_noeuds_pha_identifies) < real(val_borne_min_lambda) );
                vec_n_param_max_lambda_reel = find ( real(vec_lambda_noeuds_pha_identifies) > real(val_borne_max_lambda) );
                vec_n_param_min_lambda_imag = find ( imag(vec_lambda_noeuds_pha_identifies) < imag(val_borne_min_lambda) );
                vec_n_param_max_lambda_imag = find ( imag(vec_lambda_noeuds_pha_identifies) > imag(val_borne_max_lambda) );
                if ( ~isempty(vec_n_param_min_lambda_reel) )
                    vec_lambda_noeuds_pha_identifies(vec_n_param_min_lambda_reel) = real(liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_lambda_reel)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*real((val_borne_min_lambda-liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_lambda_reel))))+1i*imag(vec_lambda_noeuds_pha_identifies(vec_n_param_min_lambda_reel)).';
                end
                if ( ~isempty(vec_n_param_max_lambda_reel) )
                    vec_lambda_noeuds_pha_identifies(vec_n_param_max_lambda_reel) = real(liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_lambda_reel)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*real((val_borne_max_lambda-liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_lambda_reel))))+1i*imag(vec_lambda_noeuds_pha_identifies(vec_n_param_max_lambda_reel)).';
                end
                if ( ~isempty(vec_n_param_min_lambda_imag) )
                    vec_lambda_noeuds_pha_identifies(vec_n_param_min_lambda_imag) = real(vec_lambda_noeuds_pha_identifies(vec_n_param_min_lambda_imag)).'+1i*imag(liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_lambda_imag)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*imag((val_borne_min_lambda-liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_lambda_imag))));
                end
                if ( ~isempty(vec_n_param_max_lambda_imag) )
                    vec_lambda_noeuds_pha_identifies(vec_n_param_max_lambda_imag) = real(vec_lambda_noeuds_pha_identifies(vec_n_param_max_lambda_imag)).'+1i*imag(liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_lambda_imag)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*imag((val_borne_max_lambda-liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_lambda_imag))));
                end
                vec_mu_noeuds_pha_identifies = vec_sol(max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+(1:max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)));
                val_borne_min_mu = struct_param_comportement_a_identifier.vec_borne_min_param_identification(n_param_mu_ident);
                val_borne_max_mu = struct_param_comportement_a_identifier.vec_borne_max_param_identification(n_param_mu_ident);
                vec_n_param_min_mu_reel = find ( real(vec_mu_noeuds_pha_identifies) < real(val_borne_min_mu) );
                vec_n_param_max_mu_reel = find ( real(vec_mu_noeuds_pha_identifies) > real(val_borne_max_mu) );
                vec_n_param_min_mu_imag = find ( imag(vec_mu_noeuds_pha_identifies) < imag(val_borne_min_mu) );
                vec_n_param_max_mu_imag = find ( imag(vec_mu_noeuds_pha_identifies) > imag(val_borne_max_mu) );
                if ( ~isempty(vec_n_param_min_mu_reel) )
                    vec_mu_noeuds_pha_identifies(vec_n_param_min_mu_reel) = real(liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_mu_reel)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*real((val_borne_min_mu-liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_mu_reel))))+1i*imag(vec_mu_noeuds_pha_identifies(vec_n_param_min_mu_reel)).';
                end
                if ( ~isempty(vec_n_param_max_mu_reel) )
                    vec_mu_noeuds_pha_identifies(vec_n_param_max_mu_reel) = real(liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_mu_reel)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*real((val_borne_max_mu-liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_mu_reel))))+1i*imag(vec_mu_noeuds_pha_identifies(vec_n_param_max_mu_reel)).';
                end
                if ( ~isempty(vec_n_param_min_mu_imag) )
                    vec_mu_noeuds_pha_identifies(vec_n_param_min_mu_imag) = real(vec_mu_noeuds_pha_identifies(vec_n_param_min_mu_imag)).'+1i*imag(liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_mu_imag)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*imag((val_borne_min_mu-liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_mu_imag))));
                end
                if ( ~isempty(vec_n_param_max_mu_imag) )
                    vec_mu_noeuds_pha_identifies(vec_n_param_max_mu_imag) = real(vec_mu_noeuds_pha_identifies(vec_n_param_max_mu_imag)).'+1i*imag(liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_mu_imag)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*imag((val_borne_max_mu-liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_mu_imag))));
                end
                [vec_n_noeud_pha_global] = find ( vec_correspondance_n_noeud_pha_global_n_noeud_pha_local > 0);
                mat_proprietes_identifies_sub_zones(n_param_lambda_ident,n_sub_zone,vec_n_noeud_pha_global) = vec_lambda_noeuds_pha_identifies.';
                mat_proprietes_identifies_sub_zones(n_param_mu_ident,n_sub_zone,vec_n_noeud_pha_global) = vec_mu_noeuds_pha_identifies.';
                mat_test_proprietes_identifies_sub_zones(n_sub_zone,vec_n_noeud_pha_global) = true;
            elseif ( test_identification_lambda )
                vec_lambda_noeuds_pha_identifies = vec_sol((1:max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)));
                val_borne_min_lambda = struct_param_comportement_a_identifier.vec_borne_min_param_identification(n_param_lambda_ident);
                val_borne_max_lambda = struct_param_comportement_a_identifier.vec_borne_max_param_identification(n_param_lambda_ident);
                vec_n_param_min_lambda_reel = find ( real(vec_lambda_noeuds_pha_identifies) < real(val_borne_min_lambda) );
                vec_n_param_max_lambda_reel = find ( real(vec_lambda_noeuds_pha_identifies) > real(val_borne_max_lambda) );
                vec_n_param_min_lambda_imag = find ( imag(vec_lambda_noeuds_pha_identifies) < imag(val_borne_min_lambda) );
                vec_n_param_max_lambda_imag = find ( imag(vec_lambda_noeuds_pha_identifies) > imag(val_borne_max_lambda) );
                if ( ~isempty(vec_n_param_min_lambda_reel) )
                    vec_lambda_noeuds_pha_identifies(vec_n_param_min_lambda_reel) = real(liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_lambda_reel)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*real((val_borne_min_lambda-liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_lambda_reel))))+1i*imag(vec_lambda_noeuds_pha_identifies(vec_n_param_min_lambda_reel)).';
                end
                if ( ~isempty(vec_n_param_max_lambda_reel) )
                    vec_lambda_noeuds_pha_identifies(vec_n_param_max_lambda_reel) = real(liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_lambda_reel)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*real((val_borne_max_lambda-liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_lambda_reel))))+1i*imag(vec_lambda_noeuds_pha_identifies(vec_n_param_max_lambda_reel)).';
                end
                if ( ~isempty(vec_n_param_min_lambda_imag) )
                    vec_lambda_noeuds_pha_identifies(vec_n_param_min_lambda_imag) = real(vec_lambda_noeuds_pha_identifies(vec_n_param_min_lambda_imag)).'+1i*imag(liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_lambda_imag)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*imag((val_borne_min_lambda-liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_lambda_imag))));
                end
                if ( ~isempty(vec_n_param_max_lambda_imag) )
                    vec_lambda_noeuds_pha_identifies(vec_n_param_max_lambda_imag) = real(vec_lambda_noeuds_pha_identifies(vec_n_param_max_lambda_imag)).'+1i*imag(liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_lambda_imag)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*imag((val_borne_max_lambda-liste_proprietes_iterations{n_iter_LDC}(n_param_lambda_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_lambda_imag))));
                end
                [vec_n_noeud_pha_global] = find ( vec_correspondance_n_noeud_pha_global_n_noeud_pha_local > 0);
                mat_proprietes_identifies_sub_zones(n_param_lambda_ident,n_sub_zone,vec_n_noeud_pha_global) = vec_lambda_noeuds_pha_identifies.';
                mat_test_proprietes_identifies_sub_zones(n_sub_zone,vec_n_noeud_pha_global) = true;
            elseif ( test_identification_mu )
                vec_mu_noeuds_pha_identifies = vec_sol((1:max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)));
                val_borne_min_mu = struct_param_comportement_a_identifier.vec_borne_min_param_identification(n_param_mu_ident);
                val_borne_max_mu = struct_param_comportement_a_identifier.vec_borne_max_param_identification(n_param_mu_ident);
                vec_n_param_min_mu_reel = find ( real(vec_mu_noeuds_pha_identifies) < real(val_borne_min_mu) );
                vec_n_param_max_mu_reel = find ( real(vec_mu_noeuds_pha_identifies) > real(val_borne_max_mu) );
                vec_n_param_min_mu_imag = find ( imag(vec_mu_noeuds_pha_identifies) < imag(val_borne_min_mu) );
                vec_n_param_max_mu_imag = find ( imag(vec_mu_noeuds_pha_identifies) > imag(val_borne_max_mu) );
                if ( ~isempty(vec_n_param_min_mu_reel) )
                    vec_mu_noeuds_pha_identifies(vec_n_param_min_mu_reel) = real(liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_mu_reel)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*real((val_borne_min_mu-liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_mu_reel))))+1i*imag(vec_mu_noeuds_pha_identifies(vec_n_param_min_mu_reel)).';
                end
                if ( ~isempty(vec_n_param_max_mu_reel) )
                    vec_mu_noeuds_pha_identifies(vec_n_param_max_mu_reel) = real(liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_mu_reel)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*real((val_borne_max_mu-liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_mu_reel))))+1i*imag(vec_mu_noeuds_pha_identifies(vec_n_param_max_mu_reel)).';
                end
                if ( ~isempty(vec_n_param_min_mu_imag) )
                    vec_mu_noeuds_pha_identifies(vec_n_param_min_mu_imag) = real(vec_mu_noeuds_pha_identifies(vec_n_param_min_mu_imag)).'+1i*imag(liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_mu_imag)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*imag((val_borne_min_mu-liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_min_mu_imag))));
                end
                if ( ~isempty(vec_n_param_max_mu_imag) )
                    vec_mu_noeuds_pha_identifies(vec_n_param_max_mu_imag) = real(vec_mu_noeuds_pha_identifies(vec_n_param_max_mu_imag)).'+1i*imag(liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_mu_imag)))+struct_param_comportement_a_identifier.seuil_relaxation_bornes_min_max*imag((val_borne_max_mu-liste_proprietes_iterations{n_iter_LDC}(n_param_mu_ident,vec_n_noeuds_pha_sub_zone(vec_n_param_max_mu_imag))));
                end
                [vec_n_noeud_pha_global] = find ( vec_correspondance_n_noeud_pha_global_n_noeud_pha_local > 0);
                mat_proprietes_identifies_sub_zones(n_param_mu_ident,n_sub_zone,vec_n_noeud_pha_global) = vec_mu_noeuds_pha_identifies.';
                mat_test_proprietes_identifies_sub_zones(n_sub_zone,vec_n_noeud_pha_global) = true;
            end
            
            % % calcul de la fonction co?ut (moindres carres) a l'optimum
            % %             F_cout_U_prec = norm(vec_R-D*U_global((nb_DDL_K+1):end));
            %             F_cout_U_prec = norm((N_mes'*vec_U_mes)-U_global((nb_DDL_K+1):end));
            %             struct_param_comportement_a_identifier_SAV = struct_param_comportement_a_identifier;
            %             struct_param_comportement_a_identifier.mat_param(struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier,vec_n_noeud_pha_global) = squeeze(mat_proprietes_identifies_sub_zones(:,n_sub_zone,vec_n_noeud_pha_global));
            %             [K,T,M,d_K_d_p] = raideur_ERCM(liste_elem_pha,liste_elem_sig_sub_zone,liste_elem_ref,struct_param_masse_raideur,vec_correspondance_n_noeud_sig_global_n_noeud_sig_local,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local,mat_pos_maillage_sig,nb_DDL_par_noeud,struct_param_comportement_a_identifier,struct_param_comportement_normalisation);
            %             K_tilde = K-omega_mec^2*M; % (elastodynamique)
            %             struct_param_comportement_a_identifier = struct_param_comportement_a_identifier_SAV;
            %             clear struct_param_comportement_a_identifier_SAV;
            %             [vec_i_K_tilde,vec_j_K_tilde,vec_s_K_tilde] = find(K_tilde);
            %             [vec_i_T,vec_j_T,vec_s_T] = find(T);
            %             [vec_i_D,vec_j_D,vec_s_D] = find(D);
            %             vec_i_global = [vec_i_T'  , vec_i_K_tilde'                   , (vec_j_K_tilde+size(K_tilde,1))' , (vec_i_D+size(K_tilde,1))'];
            %             vec_j_global = [vec_j_T'  , (vec_j_K_tilde+size(K_tilde,2))' , vec_i_K_tilde'                   , (vec_j_D+size(K_tilde,2))'];
            %             vec_s_global = [vec_s_T.' , vec_s_K_tilde.'                  , conj(vec_s_K_tilde).'            , (-kappa*vec_s_D.')];
            %             K_global = sparse(vec_i_global,vec_j_global,vec_s_global,2*size(K_tilde,1),2*size(K_tilde,2));
            %             clear vec_i_K_tilde vec_j_K_tilde vec_s_K_tilde;
            %             clear vec_i_T vec_j_T vec_s_T;
            %             clear vec_i_D vec_j_D vec_s_D;
            %             clear vec_i_global vec_j_global vec_s_global;
            % %             clear K M T K_tilde D;
            %             vec_F_global = zeros(size(K_global,1),1);
            %             vec_F_global(1:size(K_tilde,1)) = vec_F;
            %             vec_F_global(size(K_tilde,1)+1:end) = -kappa*vec_R;
            % % MAJ du systeme sur K_global pour tenir compte des CL
            % % par substitution
            %             [Ks_global,Fs_global,U_impose_global,vec_n_DDL_conserves_global,vec_n_DDL_supprimes_global] = maj_matrices(K_global,vec_F_global,liste_DL_bloque);
            % % resolution et reecriture de U (assemblage du vecteur resultat)
            %             Us_global = Ks_global\Fs_global;
            % % reintroduction des CL
            % % par substitution
            %             U_global = CL_assemblage(Us_global,U_impose_global,vec_n_DDL_conserves_global,vec_n_DDL_supprimes_global);
            % %             F_cout_U_new = norm(vec_R-D*U_global((nb_DDL_K+1):end));
            %             F_cout_U_new = norm((N_mes'*vec_U_mes)-U_global((nb_DDL_K+1):end));
            %             disp(['F_cout_prec = ' num2str(F_cout_U_prec) ', F_cout_U_new - F_cout_U_prec = ' num2str(F_cout_U_new-F_cout_U_prec)]);
            %             if ( F_cout_U_new < F_cout_U_prec ) % on a bien minimise la fonction cout => on conserve les proprietes
            %             else % on n'a pas minimise la fonction cout => on ne conserve pas les proprietes et on reprend celles qui avaient servi au premier calcul
            %                 mat_proprietes_identifies_sub_zones(:,n_sub_zone,vec_n_noeud_pha_global) = struct_param_comportement_a_identifier.mat_param(struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier,vec_n_noeud_pha_global);
            %             end
            
        end
        
        % t_fin = cputime;
        %         disp(['                ' num2str(t_fin-t_ini) ' s']);
        
        % mise a jour de la liste des candidats possibles pour etre le centre d'une sub-zone => vecteur "vec_test_n_elem_sig_centre_sub_zone"
        i_min_recouvrement_sub_zone = ceil(i_sig_centre_sub_zone-floor(L_x_sub_zone/2)*(1+1-taux_recouvrement_sub_zones_par_MAJ_materielle));
        i_max_recouvrement_sub_zone = floor(i_sig_centre_sub_zone+(L_x_sub_zone-1-floor(L_x_sub_zone/2))*(1+1-taux_recouvrement_sub_zones_par_MAJ_materielle));
        j_min_recouvrement_sub_zone = ceil(j_sig_centre_sub_zone-floor(L_y_sub_zone/2)*(1+1-taux_recouvrement_sub_zones_par_MAJ_materielle));
        j_max_recouvrement_sub_zone = floor(j_sig_centre_sub_zone+(L_y_sub_zone-1-floor(L_y_sub_zone/2))*(1+1-taux_recouvrement_sub_zones_par_MAJ_materielle));
        k_min_recouvrement_sub_zone = ceil(k_sig_centre_sub_zone-floor(L_z_sub_zone/2)*(1+1-taux_recouvrement_sub_zones_par_MAJ_materielle));
        k_max_recouvrement_sub_zone = floor(k_sig_centre_sub_zone+(L_z_sub_zone-1-floor(L_z_sub_zone/2))*(1+1-taux_recouvrement_sub_zones_par_MAJ_materielle));
        for i = i_min_recouvrement_sub_zone:i_max_recouvrement_sub_zone
            for j = j_min_recouvrement_sub_zone:j_max_recouvrement_sub_zone
                for k = k_min_recouvrement_sub_zone:k_max_recouvrement_sub_zone
                    if ( (i >=1) && (i <= ni_elem_sig) && (j >=1) && (j <= nj_elem_sig) && (k >=1) && (k <= nk_elem_sig) )
                        n_elem_sig_local = mat_correspondance_i_sig_j_sig_k_sig_n_elem_sig(i,j,k);
                        if ( ~isnan(n_elem_sig_local) )
                            vec_test_n_elem_sig_centre_sub_zone(n_elem_sig_local) = true;
                        end
                    end
                end
            end
        end
        
        % test pour savoir si on a bien balaye toutes les subzones pour atteindre le taux de recouvrement souhaite
        if ( prod(vec_test_n_elem_sig_centre_sub_zone) == 1 )
            % on regarde si on a bien identifie au moins une fois tous les param?tres de toutes les sub-zones
            vec_test_noeuds_pha_identifies = ( sum(mat_test_proprietes_identifies_sub_zones(1:n_sub_zone,:),1) > 0 );
            [vec_n_noeuds_pha_non_identifies] = find ( ~vec_test_noeuds_pha_identifies );
            if ( ~isempty(vec_n_noeuds_pha_non_identifies) )
                mat_test_n_noeuds_pha_non_identifies_par_pha = false(size(mat_n_noeuds_pha_par_pha));
                for n_noeud_pha = 1:size(mat_n_noeuds_pha_par_pha,1)
                    for nn_noeud_non_identifie = 1:length(vec_n_noeuds_pha_non_identifies)
                        [ii_tempo] = find (mat_n_noeuds_pha_par_pha(n_noeud_pha,:) == vec_n_noeuds_pha_non_identifies(nn_noeud_non_identifie));
                        if ( ~isempty(ii_tempo) )
                            mat_test_n_noeuds_pha_non_identifies_par_pha(n_noeud_pha,ii_tempo) = true;
                        end
                    end
                end
                vec_n_pha_non_identifiees = find ( sum(mat_test_n_noeuds_pha_non_identifies_par_pha,1) > 0 );
                for nn_pha_non_identifiees = 1:length(vec_n_pha_non_identifiees)
                    n_pha = vec_n_pha_non_identifiees(nn_pha_non_identifiees);
                    vec_test_n_elem_sig_centre_sub_zone(liste_elem_pha{n_pha}.vec_n_sig_K) = false;
                    vec_test_n_elem_sig_centre_sub_zone(liste_elem_pha{n_pha}.vec_n_sig_M) = false;
                end
                if ( prod(vec_test_n_elem_sig_centre_sub_zone) == 1 )
                    test_fin_sub_zones = true;
                end
            else
                test_fin_sub_zones = true;
                mat_test_n_noeuds_pha_non_identifies_par_pha = false(size(mat_n_noeuds_pha_par_pha));
                for n_noeud_pha = 1:size(mat_n_noeuds_pha_par_pha,1)
                    for nn_noeud_non_identifie = 1:length(vec_n_noeuds_pha_non_identifies)
                        [ii_tempo] = find (mat_n_noeuds_pha_par_pha(n_noeud_pha,:) == vec_n_noeuds_pha_non_identifies(nn_noeud_non_identifie));
                        if ( ~isempty(ii_tempo) )
                            mat_test_n_noeuds_pha_non_identifies_par_pha(n_noeud_pha,ii_tempo) = true;
                        end
                    end
                end
                vec_n_pha_non_identifiees = find ( sum(mat_test_n_noeuds_pha_non_identifies_par_pha,1) > 0 );
            end
        end
        
        %     end
        
        % calcul des proprietes moyennes sur toutes les sub-zones et conservation des valeurs initiales pour les parametres non identifies
        nb_sub_zones = max(find ( sum(mat_test_proprietes_identifies_sub_zones,2) > 0 ));
        mat_proprietes_identifies_moyennes_sub_zones = zeros(nb_parametres_comportement_a_identifier,size(struct_param_comportement_a_identifier.mat_param,2));
        vec_test_parametres_identifies = sum(mat_test_proprietes_identifies_sub_zones,1) > 0;
        if ( ~exist('vec_n_pha_non_identifiees','var') )
            vec_test_noeuds_pha_identifies = ( sum(mat_test_proprietes_identifies_sub_zones(1:n_sub_zone,:),1) > 0 );
            [vec_n_noeuds_pha_non_identifies] = find ( ~vec_test_noeuds_pha_identifies );
            mat_test_n_noeuds_pha_non_identifies_par_pha = false(size(mat_n_noeuds_pha_par_pha));
            for n_noeud_pha = 1:size(mat_n_noeuds_pha_par_pha,1)
                for nn_noeud_non_identifie = 1:length(vec_n_noeuds_pha_non_identifies)
                    [ii_tempo] = find (mat_n_noeuds_pha_par_pha(n_noeud_pha,:) == vec_n_noeuds_pha_non_identifies(nn_noeud_non_identifie));
                    if ( ~isempty(ii_tempo) )
                        mat_test_n_noeuds_pha_non_identifies_par_pha(n_noeud_pha,ii_tempo) = true;
                    end
                end
            end
            vec_n_pha_non_identifiees = find ( sum(mat_test_n_noeuds_pha_non_identifies_par_pha,1) > 0 );
        end
        for n_param = 1:nb_parametres_comportement_a_identifier
            mat_proprietes_identifies_moyennes_sub_zones(n_param,:) = sum(reshape(mat_proprietes_identifies_sub_zones(n_param,1:nb_sub_zones,:),nb_sub_zones,size(struct_param_comportement_a_identifier.mat_param,2)).*mat_test_proprietes_identifies_sub_zones(1:nb_sub_zones,:),1)./max([sum(mat_test_proprietes_identifies_sub_zones(1:nb_sub_zones,:),1);ones(1,size(struct_param_comportement_a_identifier.mat_param,2))],[],1);
            mat_proprietes_identifies_moyennes_sub_zones(n_param,vec_n_pha_non_identifiees) = struct_param_comportement_a_identifier.mat_param(struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier(n_param),vec_n_pha_non_identifiees);
            struct_param_comportement_a_identifier.mat_param(struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier(n_param),:) = mat_proprietes_identifies_moyennes_sub_zones(n_param,:);
        end
        
        
        % test de la convergence
        vec_difference_proprietes = liste_proprietes_iterations{n_iter_LDC}-mat_proprietes_identifies_moyennes_sub_zones;
        if ( norm(vec_difference_proprietes) < tolerance_LDC*norm(liste_proprietes_iterations{n_iter_LDC}) )
            test_convergence_LDC = true;
        end
        
        % % affichage des noeuds des differentes sub-zones
        %     figure;
        %     hold on;
        %     mat_coul = jet(nb_sub_zones);
        %     liste_marqueurs = {'x','+','o','d','s','*','>','>','^'};
        %     for n_sub_zone = 1:nb_sub_zones
        %         n_marqueur = randi(length(liste_marqueurs));
        %         vec_n_noeuds_sig_sub_zone = liste_sub_zones{n_sub_zone}.vec_correspondance_n_noeud_sig_local_n_noeud_sig_global;
        %         hp_sig = plot3(mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,1),mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,2),mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,3),'xk');set(hp_sig,'Color',mat_coul(n_sub_zone,:),'Marker',liste_marqueurs{n_marqueur});
        % %        hp_mes = plot3(liste_sub_zones{n_sub_zone}.vec_x_noeuds_mes_sub_zone,liste_sub_zones{n_sub_zone}.vec_y_noeuds_mes_sub_zone,liste_sub_zones{n_sub_zone}.vec_z_noeuds_mes_sub_zone,'or');set(hp_mes,'Color',mat_coul(n_sub_zone,:));
        %     end
        %     grid;
        %     xlabel('x (m)');
        %     ylabel('y (m)');
        %     zlabel('z (m)');
        %     title('maillage toutes sub-zones');
        
        disp(['        norme 1 valeurs identifies = ' num2str(sum(abs(mat_proprietes_identifies_moyennes_sub_zones),2)/size(mat_proprietes_identifies_moyennes_sub_zones,2)) ', norme relative de la correction = ' num2str(norm(vec_difference_proprietes)/norm(liste_proprietes_iterations{n_iter_LDC}))]);
        disp(' ');
        
        % mise a jour des proprietes
        n_iter_LDC = n_iter_LDC+1;
        liste_proprietes_iterations{n_iter_LDC} = mat_proprietes_identifies_moyennes_sub_zones;
        
    end

    sTime(1,idx) = toc;

    t_fin_identification(1,idx) = cputime - t_ini_identification(1,idx);

    valKappa(idx,1:length(cell2mat(liste_proprietes_iterations))) = cell2mat(liste_proprietes_iterations);

    cd(path_dir{3});

    n_iter_LDC_max = n_iter_LDC;
    % n_iter_LDC = n_iter_LDC_max;

    close all;

    for nn_param = 1:size(liste_proprietes_iterations,1)

        n_param = struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier(nn_param);
        nom_param = struct_param_comportement_a_identifier.liste_parametres_comportement{n_param};
        gcf = figure;
        hold on;
        plot(real(liste_proprietes_iterations{n_iter_LDC}(nn_param,:)),'-r');
        plot(imag(liste_proprietes_iterations{n_iter_LDC}(nn_param,:)),'-b');
        grid;
        xl = xlabel('Phase number node','interpreter','latex');
        yl = ylabel('$\mu$ [Pa]','interpreter','latex');
        lg = legend('Real','Imag','interpreter','latex');
        [xl.FontSize, yl.FontSize] = deal(12);
        lg.FontSize = 11;
        saveas(gcf,sprintf('phaseNum_kappa_%d.png',idx));
        close gcf;

    end
    
    vec_param_identifie_moyen = zeros(size(liste_proprietes_iterations{n_iter_LDC},1),n_iter_LDC_max);
    
    for n_iter_LDC = 1:n_iter_LDC_max
        for n_param = 1:size(liste_proprietes_iterations{n_iter_LDC},1)
            vec_param_identifie_moyen(n_param,n_iter_LDC) = mean(liste_proprietes_iterations{n_iter_LDC}(n_param,:));
        end
    end
    
    for n_param = 1:size(liste_proprietes_iterations{n_iter_LDC},1)
        % nom_param = struct_param_comportement_a_identifier.liste_parametres_comportement{struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier(n_param)};
        
        gcf = figure;
        hold on;

        plot(real(vec_param_identifie_moyen(n_param,:)),'--r');
        plot(imag(vec_param_identifie_moyen(n_param,:)),'--b');

        plot(1743*ones(size(vec_param_identifie_moyen(n_param,:))),'-k');
        plot(174.3*ones(size(vec_param_identifie_moyen(n_param,:))),'-k');

        grid;
        
        tl = title(sprintf('Material property $\\mu$ ($\\kappa$ = %0.0e)',kappa(idx)),'interpreter','latex');
        xl = xlabel('Number of iterations','interpreter','latex');
        yl = ylabel('$\mu$ [Pa]','interpreter','latex');
        lg = legend({'Re $\left( \tilde{\mu} \right)$', 'Im $\left( \tilde{\mu} \right)$','Re $\left( \mu \right)$','Im $\left( \mu \right)$'},'interpreter','latex');
        
        [tl.FontSize, xl.FontSize, yl.FontSize] = deal(12);
        lg.FontSize = 11;

        saveas(gcf,sprintf('results_kappa_%d.png',idx));

        close gcf;
        
    end

    if idx ~= length(kappa)
        cd(path_dir{2});
    end

end

gcf = figure;
hold on;

plot(1:length(sTime), sTime, '-k');
tl = title('Code performance (tic-toc)', 'interpreter', 'latex');
xl = xlabel('Number of kappa indices', 'interpreter', 'latex');
yl = ylabel('Simulation time [s]', 'interpreter', 'latex');
[tl.FontSize, xl.FontSize, yl.FontSize] = deal(12);
saveas(gcf,'simTime (t-t).png');
close gcf;


gcf = figure;
hold on;

plot(1:length(t_fin_identification), t_fin_identification, '-k');
tl = title('Code performance (cputime)', 'interpreter', 'latex');
xl = xlabel('Number of kappa indices', 'interpreter', 'latex');
yl = ylabel('Simulation time [s]', 'interpreter', 'latex');
[tl.FontSize, xl.FontSize, yl.FontSize] = deal(12);
saveas(gcf, 'simTime');
close gcf;

save('resultsKappa.mat');

% for n_iter_LDC = 1:n_iter_LDC_max;
%  for nn_param = 1:size(liste_proprietes_iterations{n_iter_LDC},1)
%   n_param = struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier(nn_param);
%   nom_param = struct_param_comportement_a_identifier.liste_parametres_comportement{n_param};
%   figure;
%   hold on;
%   plot(real(liste_proprietes_iterations{n_iter_LDC}(nn_param,:)),'-r');
%   plot(imag(liste_proprietes_iterations{n_iter_LDC}(nn_param,:)),'-b');
%   grid;
%   xlabel('numero noeud phase');ylabel([nom_param ' (Pa)']);title(['n iter LDC = ' num2str(n_iter_LDC)]);legend('reel','imag');
%  end
% end

% for n_iter_LDC = 2:n_iter_LDC_max;
%  for nn_param = 1:size(liste_proprietes_iterations{n_iter_LDC},1)
%   n_param = struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier(nn_param);
%   nom_param = struct_param_comportement_a_identifier.liste_parametres_comportement{n_param};
%   figure;
%   hold on;
%   plot(abs(real(liste_proprietes_iterations{n_iter_LDC}(nn_param,:)-liste_proprietes_iterations{n_iter_LDC-1}(nn_param,:))),'-r');
%   plot(abs(imag(liste_proprietes_iterations{n_iter_LDC}(nn_param,:)-liste_proprietes_iterations{n_iter_LDC-1}(nn_param,:))),'-b');
%   grid;
%   xlabel('numero noeud phase');ylabel(['delta' nom_param ' (Pa)']);title(['n iter LDC = ' num2str(n_iter_LDC)]);legend('reel','imag');
%  end
% end