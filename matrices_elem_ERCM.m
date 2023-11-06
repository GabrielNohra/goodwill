% function [Ke,Me,mat_pha] = matrices_elem_ERCM (liste_elem_pha,elem_sig,mat_pos_sig,liste_elem_ref,lump,nb_DDL_par_noeud,champ_proprietes)
function [Ke,Me,d_K_d_p] = matrices_elem_ERCM (liste_elem_ref,liste_elem_pha,struct_param_masse_raideur,elem_sig,mat_pos_sig,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local,nb_DDL_par_noeud,struct_param_comportement)

n_integration_K = struct_param_masse_raideur.n_integration_K;
n_integration_M = struct_param_masse_raideur.n_integration_M;
test_lump = struct_param_masse_raideur.test_lump;

%n_elem_ref = elem_sig.n_elem_ref;

% mat_pha = [];

%type_elem = elem_sig.type_elem;
%mat_pos_noeud_elem = [liste_elem_ref{type_elem+1}.vec_x_noeuds;liste_elem_ref{type_elem+1}.vec_y_noeuds;liste_elem_ref{type_elem+1}.vec_z_noeuds];
% initialisation matrice raideur élementaire
L_mat = nb_DDL_par_noeud*elem_sig.nb_noeuds; % taille matrice raideur
nb_param_pha = max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local>0));
Ke = zeros(L_mat,L_mat);
Me = zeros(L_mat,L_mat);
d_K_d_p = zeros(nb_param_pha,L_mat,length(struct_param_comportement.vec_numeros_parametres_a_identifier));

mat_pos_noeuds = mat_pos_sig(elem_sig.vec_n_noeuds,:);
  
% Raideur
w = liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.poids_Gauss;
p_G = liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.pos_Gauss;
%d_Nf_global = reshape(squeeze(liste_elem_ref{elem_sig.n_elem_ref}.d_f_Nf(p_G(:,1),p_G(:,2),p_G(:,3))),size(p_G,1),size(mat_pos_noeuds,1),size(mat_pos_noeuds,2));
d_Nf_global = liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.d_Nf_G;

%d_Nf => (nb_points_Gauss,nb_noeuds_FF,nb_dim)

for k = 1:size(p_G,1)
%  x = p_G(k,1);
%  y = p_G(k,2);
%  z = p_G(k,3);
% % calcul des gradients dans la configuration de reference
%  d_Nf = squeeze(liste_elem_ref{elem_sig.n_elem_ref}.d_f_Nf(x,y,z))';
 d_Nf = reshape(squeeze(d_Nf_global(k,:,:)),size(mat_pos_noeuds,1),size(mat_pos_noeuds,2))';
% calcul du jacobien entre la configuration reelle et la configuration de reference
 Jaco = d_Nf*mat_pos_noeuds; % matrice 3x3. Produit du gradient fonction de forme associé et coordonnées des noeuds
 J = det(Jaco); % déterminant
% matrice du gradient des fonctions de formes dans la configuration reelle
% permettant de calculer le tenseur de deformation en notation "Standard" : (exx,eyy,ezz,2*exy,2*exz,2*eyz)
 [Be] = calcul_Be(d_Nf,Jaco,nb_DDL_par_noeud);
% Matrice proprietes
 n_pha = elem_sig.vec_n_pha_G_K(k);
 elem_pha = liste_elem_pha{n_pha};
 vec_f_formes_pha = squeeze(liste_elem_ref{liste_elem_pha{n_pha}.n_elem_ref}.f_Nf(elem_sig.mat_pos_pha_G_K(1,k),elem_sig.mat_pos_pha_G_K(2,k),elem_sig.mat_pos_pha_G_K(3,k)));
 A = struct_param_comportement.f_C(struct_param_comportement.mat_param(struct_param_comportement.vec_numeros_parametres_comportement,elem_pha.vec_n_noeuds)*vec_f_formes_pha);
% %  if ( strcmp(struct_param_comportement.type_comportement,'elastique_lineaire_isotrope') == 1 )
% %   lambda_G = struct_param_comportement.mat_param(1,elem_pha.vec_n_noeuds)*vec_f_formes_pha;
% %   mu_G = struct_param_comportement.mat_param(2,elem_pha.vec_n_noeuds)*vec_f_formes_pha;
% %   a = real(lambda_G+2*mu_G);
% %   b = real(lambda_G);
% %   c = real(mu_G);
% %   A = [a b b 0 0 0; b a b 0 0 0; b b a 0 0 0; 0 0 0 c 0 0; 0 0 0 0 c 0; 0 0 0 0 0 c];
% %  elseif ( strcmp(struct_param_comportement.type_comportement,'elastique_lineaire_isotrope_quasi_incompressible') == 1 )
% %  elseif ( strcmp(struct_param_comportement.type_comportement,'elastique_lineaire_isotrope_incompressible') == 1 )
% %  elseif ( strcmp(struct_param_comportement.type_comportement,'visco_elastique_lineaire_isotrope') == 1 )
% %   lambda_G = struct_param_comportement.mat_param(1,elem_pha.vec_n_noeuds)*vec_f_formes_pha;
% %   mu_G = struct_param_comportement.mat_param(2,elem_pha.vec_n_noeuds)*vec_f_formes_pha;   
% %   a = lambda_G+2*mu_G;
% %   b = lambda_G;
% %   c = mu_G;
% %   A = [a b b 0 0 0; b a b 0 0 0; b b a 0 0 0; 0 0 0 c 0 0; 0 0 0 0 c 0; 0 0 0 0 0 c];
% %  elseif ( strcmp(struct_param_comportement.type_comportement,'visco_elastique_lineaire_isotrope_quasi_incompressible') == 1 )
% %  elseif ( strcmp(struct_param_comportement.type_comportement,'visco_elastique_lineaire_isotrope_incompressible') == 1 )
% %  end
%    mat_pha = [mat_pha;lambda_G mu_G];
% matrice pour le calcul des contraintes
%    [Ce] = A*Be;
% Calcul raideur elementaire sur l element de reference
 Ke=Ke+(w(k)*J*Be'*A*Be);
% (i,:)
% Calcul de la "contribution materielle" sur le noeud de phase "elem_pha.vec_n_noeuds"
% => vec_f_formes_pha
 d_A_dp = struct_param_comportement.d_f_C_d_p(struct_param_comportement.mat_param(struct_param_comportement.vec_numeros_parametres_comportement,elem_pha.vec_n_noeuds)*vec_f_formes_pha);
 for l = 1:length(struct_param_comportement.vec_numeros_parametres_a_identifier)
  d_K_d_p(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(elem_pha.vec_n_noeuds),:,l) = d_K_d_p(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(elem_pha.vec_n_noeuds),:,l)+(w(k)*J*sum(Be.*(squeeze(d_A_dp(:,:,l))*Be),1)); % remarque on change le "'*" en ".*"
 end
end


if nargout == 2
% Masse
 w = liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_M}.poids_Gauss;
 p_G = liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_M}.pos_Gauss;
% Nf_global = reshape(squeeze(liste_elem_ref{elem_sig.n_elem_ref}.f_Nf(p_G(:,1),p_G(:,2),p_G(:,3))),size(p_G,1),size(mat_pos_noeuds,1));
% d_Nf_global = reshape(squeeze(liste_elem_ref{elem_sig.n_elem_ref}.d_f_Nf(p_G(:,1),p_G(:,2),p_G(:,3))),size(p_G,1),size(mat_pos_noeuds,1),size(mat_pos_noeuds,2));
 Nf_global = liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_M}.Nf_G;
 d_Nf_global = liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_M}.d_Nf_G;
 v_J = zeros(1,size(p_G,1));
 vec_rho_G = zeros(1,size(p_G,1));
 for k = 1:size(p_G,1)
%   x = p_G(k,1);
%   y = p_G(k,2);
%   z = p_G(k,3);
% % calcul des gradients dans la configuration de reference
%   Nf = squeeze(liste_elem_ref{elem_sig.n_elem_ref}.f_Nf(x,y,z))';
%   d_Nf = squeeze(liste_elem_ref{elem_sig.n_elem_ref}.d_f_Nf(x,y,z))';
  Nf = reshape(squeeze(Nf_global(k,:)),size(mat_pos_noeuds,1),1)';
  d_Nf = reshape(squeeze(d_Nf_global(k,:,:)),size(mat_pos_noeuds,1),size(mat_pos_noeuds,2))';
%  calcul du jacobien entre la configuration reelle et la configuration de reference
  Jaco = d_Nf*mat_pos_noeuds; % matrice 3x3. Produit du gradient fonction de forme associé et coordonnées des noeuds
  J = det(Jaco); % déterminant
  v_J(k) = J;
 % Formulation matrice des fonctions de formes
  mat_Nf = zeros(3,L_mat);
  mat_Nf(1,1:nb_DDL_par_noeud:end) = Nf;
  mat_Nf(2,2:nb_DDL_par_noeud:end) = Nf;
  mat_Nf(3,3:nb_DDL_par_noeud:end) = Nf;
% Matrice proprietes
  n_pha = elem_sig.vec_n_pha_G_M(k);
  elem_pha = liste_elem_pha{n_pha};
  vec_f_formes_pha = squeeze(liste_elem_ref{liste_elem_pha{n_pha}.n_elem_ref}.f_Nf(elem_sig.mat_pos_pha_G_M(1,k),elem_sig.mat_pos_pha_G_M(2,k),elem_sig.mat_pos_pha_G_M(3,k)));
% %  rho_G = champ_proprietes.rho(elem_pha.vec_n_noeuds)*vec_f_formes;
%  rho_G = struct_param_comportement.mat_param(3,elem_pha.vec_n_noeuds)*vec_f_formes_pha;
  rho_G = struct_param_comportement.mat_param(struct_param_comportement.vec_numeros_parametres_masse,elem_pha.vec_n_noeuds)*vec_f_formes_pha;
  vec_rho_G(k) = rho_G;
 % Calcul de la matrice de masse élémentaire
  Me = Me+(w(k)*J*rho_G*(mat_Nf')*mat_Nf);
 end
 
 if ( test_lump ) % lumping - Masse
%  masse = sum(w)*mean(rho_G)*mean(v_J);
  masse = sum(w)*mean(vec_rho_G)*mean(v_J);
  Me_sum = sum(diag(Me));
  Me = (masse/Me_sum)*diag(diag(Me));
 end
end

end

