function [N,D] = projection_mes(mat_pos_mes,mat_pos_sig,liste_elem_sig,mat_n_sig,liste_elem_ref,nb_DDL_par_noeud)

vec_x_mes = mat_pos_mes(1,:);
vec_y_mes = mat_pos_mes(2,:);
vec_z_mes = mat_pos_mes(3,:);

vec_x_min_sig = zeros(1,length(liste_elem_sig));
vec_x_max_sig = zeros(1,length(liste_elem_sig));
vec_y_min_sig = zeros(1,length(liste_elem_sig));
vec_y_max_sig = zeros(1,length(liste_elem_sig));
vec_z_min_sig = zeros(1,length(liste_elem_sig));
vec_z_max_sig = zeros(1,length(liste_elem_sig));
for n_elem_sig = 1:length(liste_elem_sig)
 vec_x_min_sig(n_elem_sig) = min(mat_pos_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,1));
 vec_x_max_sig(n_elem_sig) = max(mat_pos_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,1));
 vec_y_min_sig(n_elem_sig) = min(mat_pos_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,2));
 vec_y_max_sig(n_elem_sig) = max(mat_pos_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,2));
 vec_z_min_sig(n_elem_sig) = min(mat_pos_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,3));
 vec_z_max_sig(n_elem_sig) = max(mat_pos_sig(liste_elem_sig{n_elem_sig}.vec_n_noeuds,3));
end
vec_x_sig_extr = unique([unique(vec_x_min_sig) unique(vec_x_max_sig)]);
vec_y_sig_extr = unique([unique(vec_y_min_sig) unique(vec_y_max_sig)]);
vec_z_sig_extr = unique([unique(vec_z_min_sig) unique(vec_z_max_sig)]);

nb_mes = size(mat_pos_mes,2);

% element sig
nb_sig = size(mat_pos_sig,1);
val_nx_sig = (interp1(vec_x_sig_extr,0:(length(vec_x_sig_extr)-1),vec_x_mes,'linear'));
for i = 1:size(mat_n_sig,1)
 if ( i == 1 )
  ii = find((val_nx_sig>=(i-1)) & (val_nx_sig<i));
 else
  ii = find((val_nx_sig>(i-1)) & (val_nx_sig<=i));
 end
 val_nx_sig(ii) = i;
end

val_ny_sig = (interp1(vec_y_sig_extr,0:(length(vec_y_sig_extr)-1),vec_y_mes,'linear'));
for i = 1:size(mat_n_sig,2)
 if ( i == 1 )
  ii = find((val_ny_sig>=(i-1)) & (val_ny_sig<i));
 else
  ii = find((val_ny_sig>(i-1)) & (val_ny_sig<=i));
 end
 val_ny_sig(ii) = i;
end

val_nz_sig = (interp1(vec_z_sig_extr,0:(length(vec_z_sig_extr)-1),vec_z_mes,'linear'));
for i = 1:size(mat_n_sig,3)
 if ( i == 1 )
  ii = find((val_nz_sig>=(i-1)) & (val_nz_sig<i));
 else
  ii = find((val_nz_sig>(i-1)) & (val_nz_sig<=i));
 end
 val_nz_sig(ii) = i;
end

ni_N = nb_mes*nb_DDL_par_noeud;
nj_N = nb_sig*nb_DDL_par_noeud;

Nf = squeeze(liste_elem_ref{liste_elem_sig{1}.n_elem_ref}.f_Nf(0,0,0));

vec_i_N = nan(1,length(Nf)*ni_N);
vec_j_N = nan(1,length(Nf)*ni_N);
vec_s_N = nan(1,length(Nf)*ni_N);

i_prec = 0;
for n_mes = 1:nb_mes
 i_sig = val_nx_sig(n_mes);
 j_sig = val_ny_sig(n_mes);
 k_sig = val_nz_sig(n_mes);
 n_sig = mat_n_sig(i_sig,j_sig,k_sig);
 x_ref = 2*((vec_x_mes(n_mes)-min(mat_pos_sig(liste_elem_sig{n_sig}.vec_n_noeuds,1)))/(max(mat_pos_sig(liste_elem_sig{n_sig}.vec_n_noeuds,1))-min(mat_pos_sig(liste_elem_sig{n_sig}.vec_n_noeuds,1))))-1;
 y_ref = 2*((vec_y_mes(n_mes)-min(mat_pos_sig(liste_elem_sig{n_sig}.vec_n_noeuds,2)))/(max(mat_pos_sig(liste_elem_sig{n_sig}.vec_n_noeuds,2))-min(mat_pos_sig(liste_elem_sig{n_sig}.vec_n_noeuds,2))))-1;
 z_ref = 2*((vec_z_mes(n_mes)-min(mat_pos_sig(liste_elem_sig{n_sig}.vec_n_noeuds,3)))/(max(mat_pos_sig(liste_elem_sig{n_sig}.vec_n_noeuds,3))-min(mat_pos_sig(liste_elem_sig{n_sig}.vec_n_noeuds,3))))-1;
 Nf = squeeze(liste_elem_ref{liste_elem_sig{n_sig}.n_elem_ref}.f_Nf(x_ref,y_ref,z_ref));
 for n_DDL = 1:nb_DDL_par_noeud
  vec_i_N(i_prec+(1:length(Nf))) = nb_DDL_par_noeud*(n_mes-1)+n_DDL;
  vec_j_N(i_prec+(1:length(Nf))) = nb_DDL_par_noeud*(liste_elem_sig{n_sig}.vec_n_noeuds-1)+n_DDL;
  vec_s_N(i_prec+(1:length(Nf))) = Nf;
  i_prec = i_prec+length(Nf);
 end
end
vec_i_N = vec_i_N(~isnan(vec_i_N));
vec_j_N = vec_j_N(~isnan(vec_j_N));
vec_s_N = vec_s_N(~isnan(vec_s_N));

N = sparse(vec_i_N,vec_j_N,vec_s_N,ni_N,nj_N);

if nargout == 2
 D = N'*N;
end
