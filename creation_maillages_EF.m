function [liste_elem_pha,mat_pos_maillage_pha,mat_pos_pha,mat_n_pha,liste_elem_sig,mat_pos_maillage_sig,mat_pos_sig,mat_n_sig] = creation_maillages_EF(mat_X_mes_3D,elem_pha_ref,elem_sig_ref,n_integration_K,n_integration_M,struct_parametres_maillage_EF,struct_grille_mes)

% seuil_delta = 0.1;

facteur_tolerance_position = struct_grille_mes.facteur_tolerance_position;

% points de Gauss des matrices de raideur "K" et de masse "M" des elements de contrainte
vec_x_G_ref_sig_K = elem_sig_ref.liste_parametres_integration{n_integration_K}.pos_Gauss(:,1);
vec_y_G_ref_sig_K = elem_sig_ref.liste_parametres_integration{n_integration_K}.pos_Gauss(:,2);
vec_z_G_ref_sig_K = elem_sig_ref.liste_parametres_integration{n_integration_K}.pos_Gauss(:,3);
vec_x_G_ref_sig_M = elem_sig_ref.liste_parametres_integration{n_integration_M}.pos_Gauss(:,1);
vec_y_G_ref_sig_M = elem_sig_ref.liste_parametres_integration{n_integration_M}.pos_Gauss(:,2);
vec_z_G_ref_sig_M = elem_sig_ref.liste_parametres_integration{n_integration_M}.pos_Gauss(:,3);

% caracteristiques du maillage de mesure
dx_mes = struct_grille_mes.dx;
x_mes_min = struct_grille_mes.x_min;
x_mes_max = x_mes_min+(size(mat_X_mes_3D,1)-1)*dx_mes;
i_mes_min = 1;
dy_mes = struct_grille_mes.dy;
y_mes_min = struct_grille_mes.y_min;
y_mes_max = y_mes_min+(size(mat_X_mes_3D,2)-1)*dy_mes;
j_mes_min = 1;
dz_mes = struct_grille_mes.dz;
z_mes_min = struct_grille_mes.z_min;
z_mes_max = z_mes_min+(size(mat_X_mes_3D,3)-1)*dz_mes;
k_mes_min = 1;
Lx_tot = x_mes_max-x_mes_min;
Ly_tot = y_mes_max-y_mes_min;
Lz_tot = z_mes_max-z_mes_min;
L_tot = [Lx_tot,Ly_tot,Lz_tot];
tolerance_position = max(L_tot)/facteur_tolerance_position;
clear Lx_tot Ly_tot Lz_tot L_tot;

mat_i_mes_3D = nan(size(mat_X_mes_3D));
mat_j_mes_3D = nan(size(mat_X_mes_3D));
mat_k_mes_3D = nan(size(mat_X_mes_3D));
for i = 1:size(mat_X_mes_3D,1)
 mat_i_mes_3D(i,:,:) = i;
end
for j = 1:size(mat_X_mes_3D,2)
 mat_j_mes_3D(:,j,:) = j;
end
for k = 1:size(mat_X_mes_3D,3)
 mat_k_mes_3D(:,:,k) = k;
end

% caracteristiques du maillage de phases
if ( elem_pha_ref.n_elem ~= 1 )
 vec_x_noeuds_pha_ref = unique(elem_pha_ref.pos_noeuds(1,:));
 vec_y_noeuds_pha_ref = unique(elem_pha_ref.pos_noeuds(2,:));
 vec_z_noeuds_pha_ref = unique(elem_pha_ref.pos_noeuds(3,:));
else
 liste_elem_pha_ref_tempo = creation_elem_ref();
 for nn_ref = 1:length(liste_elem_pha_ref_tempo)
  if ( strcmp(liste_elem_pha_ref_tempo{nn_ref}.type_elem,'HEX8') == 1 )
   vec_x_noeuds_pha_ref = unique(liste_elem_pha_ref_tempo{nn_ref}.pos_noeuds(1,:));
   vec_y_noeuds_pha_ref = unique(liste_elem_pha_ref_tempo{nn_ref}.pos_noeuds(2,:));
   vec_z_noeuds_pha_ref = unique(liste_elem_pha_ref_tempo{nn_ref}.pos_noeuds(3,:));
  end
 end
 clear liste_elem_pha_ref_tempo;
end
di_pha_ref = (length(vec_x_noeuds_pha_ref)-1);
dj_pha_ref = (length(vec_y_noeuds_pha_ref)-1);
dk_pha_ref = (length(vec_z_noeuds_pha_ref)-1);
vec_i_pha_ref = 0:di_pha_ref;
vec_j_pha_ref = 0:dj_pha_ref;
vec_k_pha_ref = 0:dk_pha_ref; 
[vec_j_pha_ref,vec_k_pha_ref,vec_i_pha_ref] = meshgrid(vec_i_pha_ref,vec_j_pha_ref,vec_k_pha_ref);
vec_i_pha_ref = vec_i_pha_ref(:)';
vec_j_pha_ref = vec_j_pha_ref(:)';
vec_k_pha_ref = vec_k_pha_ref(:)';

step_noeuds_pha_elem_i = max([di_pha_ref 1]);
step_noeuds_pha_elem_j = max([dj_pha_ref 1]);
step_noeuds_pha_elem_k = max([dk_pha_ref 1]);

if ( elem_pha_ref.n_elem == 1 )
 vec_ordre_pha_ref = [1 5 4 8 2 6 3 7];
 vec_ordre_pha_local = 1:8;
elseif ( elem_pha_ref.n_elem == 2 )
 vec_ordre_pha_ref = [1 5 4 8 2 6 3 7];
 vec_ordre_pha_local = 1:8;
elseif ( elem_pha_ref.n_elem == 3 )
 vec_ordre_pha_ref = [1 13 5 12 20 4 16 8 9 17 11 19 2 14 6 10 18 3 15 7];
 vec_ordre_pha_local = [1:4 6:10 12 16 18:22 24:27];
elseif ( elem_pha_ref.n_elem == 4 )
 vec_ordre_pha_ref = [1 13 5 12 25 20 4 16 8 9 22 17 21 27 26 11 24 19 2 14 6 10 23 18 3 15 7];
 vec_ordre_pha_local = 1:27;
end

% caracteristiques du maillage de contraintes
if ( elem_sig_ref.n_elem ~= 1 )
 vec_x_noeuds_sig_ref = unique(elem_sig_ref.pos_noeuds(1,:));
 vec_y_noeuds_sig_ref = unique(elem_sig_ref.pos_noeuds(2,:));
 vec_z_noeuds_sig_ref = unique(elem_sig_ref.pos_noeuds(3,:));
else
 liste_elem_sig_ref_tempo = creation_elem_ref();
 for nn_ref = 1:length(liste_elem_sig_ref_tempo)
  if ( strcmp(liste_elem_sig_ref_tempo{nn_ref}.type_elem,'HEX8') == 1 )
   vec_x_noeuds_sig_ref = unique(liste_elem_sig_ref_tempo{nn_ref}.pos_noeuds(1,:));
   vec_y_noeuds_sig_ref = unique(liste_elem_sig_ref_tempo{nn_ref}.pos_noeuds(2,:));
   vec_z_noeuds_sig_ref = unique(liste_elem_sig_ref_tempo{nn_ref}.pos_noeuds(3,:));
  end
 end
 clear liste_elem_sig_ref_tempo;
end
di_sig_ref = (length(vec_x_noeuds_sig_ref)-1);
dj_sig_ref = (length(vec_y_noeuds_sig_ref)-1);
dk_sig_ref = (length(vec_z_noeuds_sig_ref)-1);
vec_i_sig_ref = 0:di_sig_ref;
vec_j_sig_ref = 0:dj_sig_ref;
vec_k_sig_ref = 0:dk_sig_ref; 
[vec_j_sig_ref,vec_k_sig_ref,vec_i_sig_ref] = meshgrid(vec_i_sig_ref,vec_j_sig_ref,vec_k_sig_ref);
vec_i_sig_ref = vec_i_sig_ref(:)';
vec_j_sig_ref = vec_j_sig_ref(:)';
vec_k_sig_ref = vec_k_sig_ref(:)';

step_noeuds_sig_elem_i = max([di_sig_ref 1]);
step_noeuds_sig_elem_j = max([dj_sig_ref 1]);
step_noeuds_sig_elem_k = max([dk_sig_ref 1]);

if ( elem_sig_ref.n_elem == 1 )
 vec_ordre_sig_ref = [1 5 4 8 2 6 3 7];
 vec_ordre_sig_local = 1:8;
elseif ( elem_sig_ref.n_elem == 2 )
 vec_ordre_sig_ref = [1 5 4 8 2 6 3 7];
 vec_ordre_sig_local = 1:8;
elseif ( elem_sig_ref.n_elem == 3 )
 vec_ordre_sig_ref = [1 13 5 12 20 4 16 8 9 17 11 19 2 14 6 10 18 3 15 7];
 vec_ordre_sig_local = [1:4 6:10 12 16 18:22 24:27];
elseif ( elem_sig_ref.n_elem == 4 )
 vec_ordre_sig_ref = [1 13 5 12 25 20 4 16 8 9 22 17 21 27 26 11 24 19 2 14 6 10 23 18 3 15 7];
 vec_ordre_sig_local = 1:27;
end

% determination du maillage "elements-finis"
%x_EF_min = x_mes_min-struct_parametres_maillage_EF.dx_moins*Lx_mes;
%x_EF_max = x_mes_max+struct_parametres_maillage_EF.dx_plus*Lx_mes;
%y_EF_min = y_mes_min-struct_parametres_maillage_EF.dy_moins*Ly_mes;
%y_EF_max = y_mes_max+struct_parametres_maillage_EF.dy_plus*Ly_mes;
%z_EF_min = z_mes_min-struct_parametres_maillage_EF.dz_moins*Lz_mes;
%z_EF_max = z_mes_max+struct_parametres_maillage_EF.dz_plus*Lz_mes;
x_EF_min = x_mes_min;
x_EF_max = x_mes_max;
y_EF_min = y_mes_min;
y_EF_max = y_mes_max;
z_EF_min = z_mes_min;
z_EF_max = z_mes_max;

%%% Creation du maillage de phases
nb_elem_pha = struct_parametres_maillage_EF.nb_elem_pha_x*struct_parametres_maillage_EF.nb_elem_pha_y*struct_parametres_maillage_EF.nb_elem_pha_z;

% determination du maillage pha
nb_elem_pha_i = struct_parametres_maillage_EF.nb_elem_pha_x;
nb_elem_pha_j = struct_parametres_maillage_EF.nb_elem_pha_y;
nb_elem_pha_k = struct_parametres_maillage_EF.nb_elem_pha_z;

nb_noeuds_pha_i = nb_elem_pha_i*step_noeuds_pha_elem_i+1;
nb_noeuds_pha_j = nb_elem_pha_j*step_noeuds_pha_elem_j+1;
nb_noeuds_pha_k = nb_elem_pha_k*step_noeuds_pha_elem_k+1;

dx_pha = (x_EF_max-x_EF_min)/nb_elem_pha_i;
dy_pha = (y_EF_max-y_EF_min)/nb_elem_pha_j;
dz_pha = (z_EF_max-z_EF_min)/nb_elem_pha_k;

nb_noeuds_pha = nb_noeuds_pha_i*nb_noeuds_pha_j*nb_noeuds_pha_k;
if (elem_pha_ref.n_elem == 1 )
 nb_noeuds_pha = nb_noeuds_pha_i*nb_noeuds_pha_j*nb_noeuds_pha_k;
 vec_x_support_DDL_pha = nan(1,nb_elem_pha);
 vec_y_support_DDL_pha = nan(1,nb_elem_pha);
 vec_z_support_DDL_pha = nan(1,nb_elem_pha);
end

liste_elem_pha = cell(1,nb_elem_pha);

vec_x_noeuds_pha = nan(1,nb_noeuds_pha);
vec_y_noeuds_pha = nan(1,nb_noeuds_pha);
vec_z_noeuds_pha = nan(1,nb_noeuds_pha);

mat_n_pha = zeros(nb_elem_pha_i,nb_elem_pha_j,nb_elem_pha_k);

vec_n_noeuds_pha_ref = zeros(1,length(vec_ordre_pha_ref));

n_elem_pha = 0;
for i = 1:nb_elem_pha_i
 for j = 1:nb_elem_pha_j
  for k = 1:nb_elem_pha_k
   n_elem_pha = n_elem_pha+1;
   
   vec_i_local = step_noeuds_pha_elem_i*(i-1)*nb_noeuds_pha_j*nb_noeuds_pha_k+vec_i_pha_ref*nb_noeuds_pha_j*nb_noeuds_pha_k;
   vec_j_local = step_noeuds_pha_elem_j*(j-1)*nb_noeuds_pha_k+vec_j_pha_ref*nb_noeuds_pha_k;
   vec_k_local = step_noeuds_pha_elem_k*(k-1)+vec_k_pha_ref;
   vec_n_noeud_local = vec_i_local+vec_j_local+vec_k_local+1;
   vec_n_noeuds_pha_ref(vec_ordre_pha_ref) = vec_n_noeud_local(vec_ordre_pha_local);

   vec_x_noeuds_pha(vec_n_noeud_local) = x_EF_min+(i-1)*dx_pha+vec_i_pha_ref*dx_pha/step_noeuds_pha_elem_i;
   vec_y_noeuds_pha(vec_n_noeud_local) = y_EF_min+(j-1)*dy_pha+vec_j_pha_ref*dy_pha/step_noeuds_pha_elem_j;
   vec_z_noeuds_pha(vec_n_noeud_local) = z_EF_min+(k-1)*dz_pha+vec_k_pha_ref*dz_pha/step_noeuds_pha_elem_k;
       
   if (elem_pha_ref.n_elem == 1 )
    vec_x_support_DDL_pha(n_elem_pha) = (min(vec_x_noeuds_pha(vec_n_noeud_local))+max(vec_x_noeuds_pha(vec_n_noeud_local)))/2;
    vec_y_support_DDL_pha(n_elem_pha) = (min(vec_y_noeuds_pha(vec_n_noeud_local))+max(vec_y_noeuds_pha(vec_n_noeud_local)))/2;
    vec_z_support_DDL_pha(n_elem_pha) = (min(vec_z_noeuds_pha(vec_n_noeud_local))+max(vec_z_noeuds_pha(vec_n_noeud_local)))/2;
   end

   if (elem_pha_ref.n_elem ~= 1 )
    elem = struct('n_elem_ref',elem_pha_ref.n_elem,'nb_noeuds',length(vec_n_noeuds_pha_ref),'vec_n_noeuds',vec_n_noeuds_pha_ref,'vec_ijk',[i j k],'type_elem',elem_pha_ref.type_elem);
   elseif (elem_pha_ref.n_elem == 1 )
    elem = struct('n_elem_ref',elem_pha_ref.n_elem,'nb_noeuds',1,'vec_n_noeuds',n_elem_pha,'vec_ijk',[i j k],'type_elem',elem_pha_ref.type_elem);
    elem.nb_noeuds_maillage = length(vec_n_noeuds_pha_ref);
    elem.vec_n_noeuds_maillage = vec_n_noeuds_pha_ref;
   end
   liste_elem_pha{n_elem_pha} = elem; 
   mat_n_pha(i,j,k) = n_elem_pha;
  end
 end
end
mat_pos_maillage_pha = [vec_x_noeuds_pha;vec_y_noeuds_pha;vec_z_noeuds_pha]';
if elem_pha_ref.n_elem ~= 1
 mat_pos_pha = mat_pos_maillage_pha;
else
 mat_pos_pha = [vec_x_support_DDL_pha;vec_y_support_DDL_pha;vec_z_support_DDL_pha]';
end

%%% Creation du maillage de contraintes
nb_elem_sig = struct_parametres_maillage_EF.nb_elem_sig_x*struct_parametres_maillage_EF.nb_elem_sig_y*struct_parametres_maillage_EF.nb_elem_sig_z;

% determination du maillage sig
nb_elem_sig_i = struct_parametres_maillage_EF.nb_elem_sig_x;
nb_elem_sig_j = struct_parametres_maillage_EF.nb_elem_sig_y;
nb_elem_sig_k = struct_parametres_maillage_EF.nb_elem_sig_z;

nb_noeuds_sig_i = nb_elem_sig_i*step_noeuds_sig_elem_i+1;
nb_noeuds_sig_j = nb_elem_sig_j*step_noeuds_sig_elem_j+1;
nb_noeuds_sig_k = nb_elem_sig_k*step_noeuds_sig_elem_k+1;

dx_sig = (x_EF_max-x_EF_min)/nb_elem_sig_i;
dy_sig = (y_EF_max-y_EF_min)/nb_elem_sig_j;
dz_sig = (z_EF_max-z_EF_min)/nb_elem_sig_k;

nb_noeuds_sig = nb_noeuds_sig_i*nb_noeuds_sig_j*nb_noeuds_sig_k;
if (elem_sig_ref.n_elem == 1 )
 nb_noeuds_sig = nb_noeuds_sig_i*nb_noeuds_sig_j*nb_noeuds_sig_k;
 vec_x_support_DDL_sig = nan(1,nb_elem_sig);
 vec_y_support_DDL_sig = nan(1,nb_elem_sig);
 vec_z_support_DDL_sig = nan(1,nb_elem_sig);
end

liste_elem_sig = cell(1,nb_elem_sig);

vec_x_noeuds_sig = nan(1,nb_noeuds_sig);
vec_y_noeuds_sig = nan(1,nb_noeuds_sig);
vec_z_noeuds_sig = nan(1,nb_noeuds_sig);

mat_n_sig = zeros(nb_elem_sig_i,nb_elem_sig_j,nb_elem_sig_k);

vec_n_noeuds_sig_ref = zeros(1,length(vec_ordre_sig_ref));

n_elem_sig = 0;
for i = 1:nb_elem_sig_i
 for j = 1:nb_elem_sig_j
  for k = 1:nb_elem_sig_k
%   n_elem_sig = n_elem_sig+1;

   vec_i_local = step_noeuds_sig_elem_i*(i-1)*nb_noeuds_sig_j*nb_noeuds_sig_k+vec_i_sig_ref*nb_noeuds_sig_j*nb_noeuds_sig_k;
   vec_j_local = step_noeuds_sig_elem_j*(j-1)*nb_noeuds_sig_k+vec_j_sig_ref*nb_noeuds_sig_k;
   vec_k_local = step_noeuds_sig_elem_k*(k-1)+vec_k_sig_ref;
   vec_n_noeud_local = vec_i_local+vec_j_local+vec_k_local+1;
   vec_n_noeuds_sig_ref(vec_ordre_sig_ref) = vec_n_noeud_local(vec_ordre_sig_local);

   vec_x_noeuds_sig(vec_n_noeud_local) = x_EF_min+(i-1)*dx_sig+vec_i_sig_ref*dx_sig/step_noeuds_sig_elem_i;
   vec_y_noeuds_sig(vec_n_noeud_local) = y_EF_min+(j-1)*dy_sig+vec_j_sig_ref*dy_sig/step_noeuds_sig_elem_j;
   vec_z_noeuds_sig(vec_n_noeud_local) = z_EF_min+(k-1)*dz_sig+vec_k_sig_ref*dz_sig/step_noeuds_sig_elem_k;

%    if (elem_sig_ref.n_elem == 1 )
%     vec_x_support_DDL_sig(n_elem_sig) = (min(vec_x_noeuds_sig(vec_n_noeud_local))+max(vec_x_noeuds_sig(vec_n_noeud_local)))/2;
%     vec_y_support_DDL_sig(n_elem_sig) = (min(vec_y_noeuds_sig(vec_n_noeud_local))+max(vec_y_noeuds_sig(vec_n_noeud_local)))/2;
%     vec_z_support_DDL_sig(n_elem_sig) = (min(vec_z_noeuds_sig(vec_n_noeud_local))+max(vec_z_noeuds_sig(vec_n_noeud_local)))/2;
%    end

% determination des phases aux points de Gauss de la matrice de raideur "K"
% Phases : matrice K
   vec_x_G_sig_K = x_EF_min+(i-1)*dx_sig+(vec_x_G_ref_sig_K/2+0.5)*dx_sig; % coord des ref gauss sig dans elem sig [-1 1]->[0 1]
   vec_y_G_sig_K = y_EF_min+(j-1)*dy_sig+(vec_y_G_ref_sig_K/2+0.5)*dy_sig;
   vec_z_G_sig_K = z_EF_min+(k-1)*dz_sig+(vec_z_G_ref_sig_K/2+0.5)*dz_sig;
   vec_i_pha_G_K = zeros(1,length(vec_x_G_sig_K));
   vec_j_pha_G_K = zeros(1,length(vec_x_G_sig_K));
   vec_k_pha_G_K = zeros(1,length(vec_x_G_sig_K));
   vec_n_pha_G_K = zeros(1,length(vec_x_G_sig_K));
   mat_pos_pha_G_K = zeros(3,length(vec_x_G_sig_K));
   for n_G = 1:length(vec_x_G_sig_K) 
    x_G_local = vec_x_G_sig_K(n_G);
    y_G_local = vec_y_G_sig_K(n_G);
    z_G_local = vec_z_G_sig_K(n_G);
    i_pha_local = min([1+floor((x_G_local-x_EF_min)/dx_pha) nb_elem_pha_i]); % indice de l'elem pha 
    j_pha_local = min([1+floor((y_G_local-y_EF_min)/dy_pha) nb_elem_pha_j]);
    k_pha_local = min([1+floor((z_G_local-z_EF_min)/dz_pha) nb_elem_pha_k]);
    n_elem_pha_K = mat_n_pha(i_pha_local,j_pha_local,k_pha_local);
    vec_i_pha_G_K(n_G) = i_pha_local;
    vec_j_pha_G_K(n_G) = j_pha_local;
    vec_k_pha_G_K(n_G) = k_pha_local;
    vec_n_pha_G_K(n_G) = n_elem_pha_K;
%    ksi_G_local = ((x_G_local-x_EF_min)/(x_EF_max-x_EF_min)-0.5)*2; % [0 1]->[-1 1]
%    eta_G_local = ((y_G_local-y_EF_min)/(x_EF_max-x_EF_min)-0.5)*2;
%    phi_G_local = ((z_G_local-z_EF_min)/(x_EF_max-x_EF_min)-0.5)*2;
    ksi_G_local = ((((x_G_local-x_EF_min)-(i_pha_local-1)*dx_pha)/dx_pha)-0.5)*2; % [0 1]->[-1 1]
    eta_G_local = ((((y_G_local-y_EF_min)-(j_pha_local-1)*dy_pha)/dy_pha)-0.5)*2;
    phi_G_local = ((((z_G_local-z_EF_min)-(k_pha_local-1)*dz_pha)/dz_pha)-0.5)*2;
    mat_pos_pha_G_K(1,n_G) = ksi_G_local;
    mat_pos_pha_G_K(2,n_G) = eta_G_local;
    mat_pos_pha_G_K(3,n_G) = phi_G_local;
   end

% Phases : matrice M
   vec_x_G_sig_M = x_EF_min+(i-1)*dx_sig+(vec_x_G_ref_sig_M/2+0.5)*dx_sig; % coord des ref gauss sig dans elem sig [-1 1]->[0 1]
   vec_y_G_sig_M = y_EF_min+(j-1)*dy_sig+(vec_y_G_ref_sig_M/2+0.5)*dy_sig;
   vec_z_G_sig_M = z_EF_min+(k-1)*dz_sig+(vec_z_G_ref_sig_M/2+0.5)*dz_sig;
   vec_n_pha_G_M = zeros(1,length(vec_x_G_sig_M));
   mat_pos_pha_G_M = zeros(3,length(vec_x_G_sig_M));
   for n_G = 1:length(vec_x_G_sig_M) 
    x_G_local = vec_x_G_sig_M(n_G);
    y_G_local = vec_y_G_sig_M(n_G);
    z_G_local = vec_z_G_sig_M(n_G);
    i_pha_local = min([1+floor((x_G_local-x_EF_min)/dx_pha) nb_elem_pha_i]); % indice de l'elem pha 
    j_pha_local = min([1+floor((y_G_local-y_EF_min)/dy_pha) nb_elem_pha_j]);
    k_pha_local = min([1+floor((z_G_local-z_EF_min)/dz_pha) nb_elem_pha_k]);
    n_elem_pha_M = mat_n_pha(i_pha_local,j_pha_local,k_pha_local);
    vec_n_pha_G_M(n_G) = n_elem_pha_M;
%    ksi_G_local = ((x_G_local-x_EF_min)/(x_EF_max-x_EF_min)-0.5)*2; % [0 1]->[-1 1]
%    eta_G_local = ((y_G_local-y_EF_min)/(x_EF_max-x_EF_min)-0.5)*2;
%    phi_G_local = ((z_G_local-z_EF_min)/(x_EF_max-x_EF_min)-0.5)*2;
    ksi_G_local = ((((x_G_local-x_EF_min)-(i_pha_local-1)*dx_pha)/dx_pha)-0.5)*2; % [0 1]->[-1 1]
    eta_G_local = ((((y_G_local-y_EF_min)-(j_pha_local-1)*dy_pha)/dy_pha)-0.5)*2;
    phi_G_local = ((((z_G_local-z_EF_min)-(k_pha_local-1)*dz_pha)/dz_pha)-0.5)*2;
    mat_pos_pha_G_M(1,n_G) = ksi_G_local;
    mat_pos_pha_G_M(2,n_G) = eta_G_local;
    mat_pos_pha_G_M(3,n_G) = phi_G_local;
   end

   elem = struct('n_elem_ref',elem_sig_ref.n_elem,'nb_noeuds',length(vec_n_noeuds_sig_ref),'vec_n_noeuds',vec_n_noeuds_sig_ref,'vec_ijk',[i j k],'type_elem',elem_sig_ref.type_elem,'mat_pos_pha_G_K',mat_pos_pha_G_K,'vec_n_pha_G_K',vec_n_pha_G_K,'mat_pos_pha_G_M',mat_pos_pha_G_M,'vec_n_pha_G_M',vec_n_pha_G_M);

   if (elem_sig_ref.n_elem == 1 )
    elem.nb_noeuds_maillage = length(vec_n_noeuds_sig_ref);
    elem.vec_n_noeuds_maillage = vec_n_noeuds_sig_ref;
   end

% selection des elements de contrainte qui contiennent "suffisamment" de points de mesure
   x_min_elem_sig = min(vec_x_noeuds_sig(elem.vec_n_noeuds));
   x_max_elem_sig = max(vec_x_noeuds_sig(elem.vec_n_noeuds));
   y_min_elem_sig = min(vec_y_noeuds_sig(elem.vec_n_noeuds));
   y_max_elem_sig = max(vec_y_noeuds_sig(elem.vec_n_noeuds));
   z_min_elem_sig = min(vec_z_noeuds_sig(elem.vec_n_noeuds));
   z_max_elem_sig = max(vec_z_noeuds_sig(elem.vec_n_noeuds));
   i_min_elem_sig_mes = i_mes_min+ceil((x_min_elem_sig-x_mes_min-tolerance_position)/dx_mes);
   j_min_elem_sig_mes = j_mes_min+ceil((y_min_elem_sig-y_mes_min-tolerance_position)/dy_mes);
   k_min_elem_sig_mes = k_mes_min+ceil((z_min_elem_sig-z_mes_min-tolerance_position)/dz_mes);
   i_max_elem_sig_mes = i_mes_min+floor((x_max_elem_sig-x_mes_min+tolerance_position)/dx_mes);
   j_max_elem_sig_mes = j_mes_min+floor((y_max_elem_sig-y_mes_min+tolerance_position)/dy_mes);
   k_max_elem_sig_mes = k_mes_min+floor((z_max_elem_sig-z_mes_min+tolerance_position)/dz_mes);
   
   ni_noeuds_sig_max = length(unique(elem_sig_ref.pos_noeuds(1,:)));
   nj_noeuds_sig_max = length(unique(elem_sig_ref.pos_noeuds(2,:)));
   nk_noeuds_sig_max = length(unique(elem_sig_ref.pos_noeuds(3,:)));

% on vérifie que l'on a bien au moins un point de mesure par DDL de contrainte sinon, on ne prend pas en compte l'element de contrainte
   vec_x_mes_elem = reshape(squeeze(mat_X_mes_3D(i_min_elem_sig_mes:i_max_elem_sig_mes,j_min_elem_sig_mes:j_max_elem_sig_mes,k_min_elem_sig_mes:k_max_elem_sig_mes,1)),1,[]);
   vec_y_mes_elem = reshape(squeeze(mat_X_mes_3D(i_min_elem_sig_mes:i_max_elem_sig_mes,j_min_elem_sig_mes:j_max_elem_sig_mes,k_min_elem_sig_mes:k_max_elem_sig_mes,2)),1,[]);
   vec_z_mes_elem = reshape(squeeze(mat_X_mes_3D(i_min_elem_sig_mes:i_max_elem_sig_mes,j_min_elem_sig_mes:j_max_elem_sig_mes,k_min_elem_sig_mes:k_max_elem_sig_mes,3)),1,[]);
   vec_i_mes_elem = reshape(squeeze(mat_i_mes_3D(i_min_elem_sig_mes:i_max_elem_sig_mes,j_min_elem_sig_mes:j_max_elem_sig_mes,k_min_elem_sig_mes:k_max_elem_sig_mes,1)),1,[]);
   vec_j_mes_elem = reshape(squeeze(mat_j_mes_3D(i_min_elem_sig_mes:i_max_elem_sig_mes,j_min_elem_sig_mes:j_max_elem_sig_mes,k_min_elem_sig_mes:k_max_elem_sig_mes,1)),1,[]);
   vec_k_mes_elem = reshape(squeeze(mat_k_mes_3D(i_min_elem_sig_mes:i_max_elem_sig_mes,j_min_elem_sig_mes:j_max_elem_sig_mes,k_min_elem_sig_mes:k_max_elem_sig_mes,1)),1,[]);
   vec_no_nan_mes_elem = find ( ~isnan(vec_x_mes_elem) & ~isnan(vec_y_mes_elem) & ~isnan(vec_z_mes_elem) );
   if ( (length(vec_no_nan_mes_elem) >= elem.nb_noeuds) && (length(unique(vec_x_mes_elem(vec_no_nan_mes_elem))) >= ni_noeuds_sig_max) && (length(unique(vec_y_mes_elem(vec_no_nan_mes_elem))) >= nj_noeuds_sig_max) && (length(unique(vec_z_mes_elem(vec_no_nan_mes_elem))) >= nk_noeuds_sig_max) )
    n_elem_sig = n_elem_sig+1;
    if (elem_sig_ref.n_elem == 1 )
     vec_x_support_DDL_sig(n_elem_sig) = (min(vec_x_noeuds_sig(vec_n_noeud_local))+max(vec_x_noeuds_sig(vec_n_noeud_local)))/2;
     vec_y_support_DDL_sig(n_elem_sig) = (min(vec_y_noeuds_sig(vec_n_noeud_local))+max(vec_y_noeuds_sig(vec_n_noeud_local)))/2;
     vec_z_support_DDL_sig(n_elem_sig) = (min(vec_z_noeuds_sig(vec_n_noeud_local))+max(vec_z_noeuds_sig(vec_n_noeud_local)))/2;
    end

% on determines les coordonnees locales des noeuds de déplacement dans le repere local de l'element de contrainte
    vec_ksi_mes_elem = (max(elem_sig_ref.pos_noeuds(1,:))-min(elem_sig_ref.pos_noeuds(1,:)))*(vec_x_mes_elem-x_min_elem_sig)/(x_max_elem_sig-x_min_elem_sig)+min(elem_sig_ref.pos_noeuds(1,:));
    vec_eta_mes_elem = (max(elem_sig_ref.pos_noeuds(2,:))-min(elem_sig_ref.pos_noeuds(2,:)))*(vec_y_mes_elem-y_min_elem_sig)/(y_max_elem_sig-y_min_elem_sig)+min(elem_sig_ref.pos_noeuds(2,:));
    vec_zeta_mes_elem = (max(elem_sig_ref.pos_noeuds(3,:))-min(elem_sig_ref.pos_noeuds(3,:)))*(vec_z_mes_elem-z_min_elem_sig)/(z_max_elem_sig-z_min_elem_sig)+min(elem_sig_ref.pos_noeuds(3,:));
    elem.vec_i_mes = vec_i_mes_elem(vec_no_nan_mes_elem);
    elem.vec_j_mes = vec_j_mes_elem(vec_no_nan_mes_elem);
    elem.vec_k_mes = vec_k_mes_elem(vec_no_nan_mes_elem);
    elem.vec_ksi_mes = vec_ksi_mes_elem(vec_no_nan_mes_elem);
    elem.vec_eta_mes = vec_eta_mes_elem(vec_no_nan_mes_elem);
    elem.vec_zeta_mes = vec_zeta_mes_elem(vec_no_nan_mes_elem);

    liste_elem_sig{n_elem_sig} = elem; 
    mat_n_sig(i,j,k) = n_elem_sig;
   else
   end

%    liste_elem_sig{n_elem_sig} = elem; 
%    mat_n_sig(i,j,k) = n_elem_sig;

%   figure;plot3(vec_x_G_sig_K,vec_y_G_sig_K,vec_z_G_sig_K,'xk');grid;hold on;plot3(vec_x_noeuds_sig(vec_n_noeud_local),vec_y_noeuds_sig(vec_n_noeud_local),vec_z_noeuds_sig(vec_n_noeud_local),'or');plot3(x_EF_min+(vec_i_pha_G_K-1)*dx_pha+(mat_pos_pha_G_K(1,:)+1)/2*dx_pha,y_EF_min+(vec_j_pha_G_K-1)*dy_pha+(mat_pos_pha_G_K(2,:)+1)/2*dy_pha,z_EF_min+(vec_k_pha_G_K-1)*dz_pha+(mat_pos_pha_G_K(3,:)+1)/2*dz_pha,'db');plot3(vec_x_noeuds_sig(elem.vec_n_noeuds),vec_y_noeuds_sig(elem.vec_n_noeuds),vec_z_noeuds_sig(elem.vec_n_noeuds),'sm');

  end
 end
end
mat_pos_maillage_sig = [vec_x_noeuds_sig;vec_y_noeuds_sig;vec_z_noeuds_sig]';
if elem_sig_ref.n_elem ~= 1
 mat_pos_sig = mat_pos_maillage_sig;
else
 mat_pos_sig = [vec_x_support_DDL_sig;vec_y_support_DDL_sig;vec_z_support_DDL_sig]';
end
liste_elem_sig_SAV = liste_elem_sig;
nb_elem_sig = 0;
for n_elem_sig = 1:length(liste_elem_sig_SAV)
 if ( ~isempty(liste_elem_sig{n_elem_sig}) )
  nb_elem_sig = nb_elem_sig+1;
 end
end
liste_elem_sig = cell(1,nb_elem_sig);
n_elem_sig = 0;
for n_elem_sig_SAV = 1:length(liste_elem_sig_SAV)
 if ( ~isempty(liste_elem_sig_SAV{n_elem_sig_SAV}) )
  n_elem_sig = n_elem_sig+1;
  liste_elem_sig{n_elem_sig} = liste_elem_sig_SAV{n_elem_sig_SAV};
 end
end
