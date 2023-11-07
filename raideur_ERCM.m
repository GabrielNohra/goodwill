%function [K,T,M] = raideur_ERCM(liste_elem_pha,liste_elem_sig,liste_elem_ref,n_integration_K,n_integration_M,nb_noeuds_sig,mat_pos_sig,lump,nb_DDL_par_noeud,champ_proprietes,champ_proprietes_elastiques)
function [K,T,M,D,d_K_d_p] = raideur_ERCM(liste_elem_pha,liste_elem_sig,liste_elem_ref,struct_param_masse_raideur,vec_correspondance_n_noeud_sig_global_n_noeud_sig_local,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local,mat_pos_sig,nb_DDL_par_noeud,struct_param_comportement_a_identifier,struct_param_comportement_normalisation)

K = [];
T = [];
M = [];
D = [];

% mat_pha_global = [];

nb_noeuds_sig_local = max(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local);
nb_noeuds_pha_local = max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local>0));
nb_param_pha = length(struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier);

if nargout >= 3
 size_K = nb_DDL_par_noeud*nb_noeuds_sig_local; % nb total de ddl (dim*nb_noeuds)
 d_K_d_p = zeros(nb_noeuds_pha_local,nb_DDL_par_noeud*nb_noeuds_sig_local,nb_param_pha);
 
%  K_eq = sparse(size_K,size_K);
 n_elem_sig = 1;
 nb_noeuds_par_element_sig = length(liste_elem_sig{n_elem_sig}.vec_n_noeuds); % numérotation noeud global
 nb_tempo = (nb_noeuds_par_element_sig*nb_DDL_par_noeud)^2;
 nb_data = nb_tempo*length(liste_elem_sig);
 vec_i_K = nan(1,nb_data);
 vec_j_K = nan(1,nb_data);
 vec_s_K = nan(1,nb_data);
 vec_i_T = nan(1,nb_data);
 vec_j_T = nan(1,nb_data);
 vec_s_T = nan(1,nb_data);
 vec_i_M = nan(1,nb_data);
 vec_j_M = nan(1,nb_data);
 vec_s_M = nan(1,nb_data);
 vec_i_D = nan(1,nb_data);
 vec_j_D = nan(1,nb_data);
 vec_s_D = nan(1,nb_data);

 compteur = 0;
 delta_indice_affich = 0.1;
 indice_affich = 0;
 for n_elem_sig=1:length(liste_elem_sig) % boucle sur 1 elt matériaux
  elem_sig = liste_elem_sig{n_elem_sig};  
%   [Ke,Me,mat_pha] = matrices_elem_ERCM(liste_elem_pha,elem_sig,mat_pos_sig,liste_elem_ref,lump,nb_DDL_par_noeud,champ_proprietes);
%   mat_pha_global = [mat_pha_global ; mat_pha];
%  [Ke,Me] = matrices_elem_ERCM (liste_elem_ref,liste_elem_pha,struct_param_masse_raideur,elem_sig,mat_pos_sig,nb_DDL_par_noeud,struct_param_comportement_a_identifier);
  [Ke,Me,De,d_Ke_d_p] = matrices_elem_ERCM (liste_elem_ref,liste_elem_pha,struct_param_masse_raideur,elem_sig,mat_pos_sig,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local,nb_DDL_par_noeud,struct_param_comportement_a_identifier);
  [Te] = matrices_elem_ERCM (liste_elem_ref,liste_elem_pha,struct_param_masse_raideur,elem_sig,mat_pos_sig,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local,nb_DDL_par_noeud,struct_param_comportement_normalisation);
 % Assemblage
  [vec_i_local_K,vec_j_local_K,vec_s_local_K] = raideur_assemblage(Ke,n_elem_sig,liste_elem_sig,nb_DDL_par_noeud,vec_correspondance_n_noeud_sig_global_n_noeud_sig_local);
%  vec_correspondance_n_noeud_sig_global_n_noeud_sig_local
  vec_i_K(compteur*nb_tempo+(1:nb_tempo)) = vec_i_local_K;
  vec_j_K(compteur*nb_tempo+(1:nb_tempo)) = vec_j_local_K;
  vec_s_K(compteur*nb_tempo+(1:nb_tempo)) = vec_s_local_K;
  [vec_i_local_M,vec_j_local_M,vec_s_local_M] = raideur_assemblage(Me,n_elem_sig,liste_elem_sig,nb_DDL_par_noeud,vec_correspondance_n_noeud_sig_global_n_noeud_sig_local);
  vec_i_M(compteur*nb_tempo+(1:nb_tempo)) = vec_i_local_M;
  vec_j_M(compteur*nb_tempo+(1:nb_tempo)) = vec_j_local_M;
  vec_s_M(compteur*nb_tempo+(1:nb_tempo)) = vec_s_local_M;
  [vec_i_local_D,vec_j_local_D,vec_s_local_D] = raideur_assemblage(De,n_elem_sig,liste_elem_sig,nb_DDL_par_noeud,vec_correspondance_n_noeud_sig_global_n_noeud_sig_local);
  vec_i_D(compteur*nb_tempo+(1:nb_tempo)) = vec_i_local_D;
  vec_j_D(compteur*nb_tempo+(1:nb_tempo)) = vec_j_local_D;
  vec_s_D(compteur*nb_tempo+(1:nb_tempo)) = vec_s_local_D;
  [vec_i_local_T,vec_j_local_T,vec_s_local_T] = raideur_assemblage(Te,n_elem_sig,liste_elem_sig,nb_DDL_par_noeud,vec_correspondance_n_noeud_sig_global_n_noeud_sig_local);
  vec_i_T(compteur*nb_tempo+(1:nb_tempo)) = vec_i_local_T;
  vec_j_T(compteur*nb_tempo+(1:nb_tempo)) = vec_j_local_T;
  vec_s_T(compteur*nb_tempo+(1:nb_tempo)) = vec_s_local_T; 
% assemblage d_K_d_p
  vec_n_noeuds_local = vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds); % local
  vec_n_DDL_local = zeros(1,nb_DDL_par_noeud*elem_sig.nb_noeuds);
  for ii = 1:nb_DDL_par_noeud
   vec_n_DDL_local(ii:nb_DDL_par_noeud:end) = nb_DDL_par_noeud*(vec_n_noeuds_local-1)+ii;
  end
  d_K_d_p(:,vec_n_DDL_local,:) = d_Ke_d_p;
  compteur = compteur+1;
  if (n_elem_sig/length(liste_elem_sig)) > (indice_affich + delta_indice_affich)
   indice_affich = indice_affich + delta_indice_affich;
%    disp(['                matrices elementaires : ' num2str(100*indice_affich) '%']);
  end
 end
vec_i_K = vec_i_K(~isnan(vec_i_K));
vec_j_K = vec_j_K(~isnan(vec_j_K));
vec_s_K = vec_s_K(~isnan(vec_s_K));
vec_i_T = vec_i_T(~isnan(vec_i_T));
vec_j_T = vec_j_T(~isnan(vec_j_T));
vec_s_T = vec_s_T(~isnan(vec_s_T));
vec_i_M = vec_i_M(~isnan(vec_i_M));
vec_j_M = vec_j_M(~isnan(vec_j_M));
vec_s_M = vec_s_M(~isnan(vec_s_M));
vec_i_D = vec_i_D(~isnan(vec_i_D));
vec_j_D = vec_j_D(~isnan(vec_j_D));
vec_s_D = vec_s_D(~isnan(vec_s_D));
K = sparse(vec_i_K,vec_j_K,vec_s_K,size_K,size_K);
T = sparse(vec_i_T,vec_j_T,vec_s_T,size_K,size_K);
M = sparse(vec_i_M,vec_j_M,vec_s_M,size_K,size_K);
D = sparse(vec_i_D,vec_j_D,vec_s_D,size_K,size_K);

% figure;plot(real(mat_pha_global(:,1)));
% figure;plot(imag(mat_pha_global(:,1)));
% figure;plot(real(mat_pha_global(:,2)));
% figure;plot(imag(mat_pha_global(:,2)));

elseif nargout == 1
 disp('A COMPLETER');
end


