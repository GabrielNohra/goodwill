function [mat_U_filtr] = filtrage_deplacements_IRM(mat_X_mes_3D,mat_U_mes_3D,mat_pos_filtr,struct_grille_mes,struct_filtrage_deplacement)

if ( strcmp(struct_filtrage_deplacement.type,'Gaussien') == 1 )
 Sigma_x = struct_filtrage_deplacement.Sigma_x;
 R_x = struct_filtrage_deplacement.R_x;
 Sigma_y = struct_filtrage_deplacement.Sigma_y;
 R_y = struct_filtrage_deplacement.R_y;
 Sigma_z = struct_filtrage_deplacement.Sigma_z;
 R_z = struct_filtrage_deplacement.R_z;
% recherche du point de la grille 3D le plus proche de la mesure
 vec_i_pos_filtr = min([max([round((mat_pos_filtr(1,:)-struct_grille_mes.x_min)/struct_grille_mes.dx)+1;ones(1,size(mat_pos_filtr,2))],[],2);size(mat_X_mes_3D,1)+zeros(1,size(mat_pos_filtr,2))],[],2);
 vec_j_pos_filtr = min([max([round((mat_pos_filtr(2,:)-struct_grille_mes.y_min)/struct_grille_mes.dy)+1;ones(1,size(mat_pos_filtr,2))],[],2);size(mat_X_mes_3D,2)+zeros(1,size(mat_pos_filtr,2))],[],2);
 vec_k_pos_filtr = min([max([round((mat_pos_filtr(3,:)-struct_grille_mes.z_min)/struct_grille_mes.dz)+1;ones(1,size(mat_pos_filtr,2))],[],2);size(mat_X_mes_3D,3)+zeros(1,size(mat_pos_filtr,2))],[],2);
 vec_i_min_pos_filtr = max([round(vec_i_pos_filtr-Rx);ones(1,size(mat_pos_filtr,2))],[],2);
 vec_i_max_pos_filtr = min([round(vec_i_pos_filtr+Rx);size(mat_X_mes_3D,1)+zeros(1,size(mat_pos_filtr,2))],[],2);
 vec_j_min_pos_filtr = max([round(vec_j_pos_filtr-Ry);ones(1,size(mat_pos_filtr,2))],[],2);
 vec_j_max_pos_filtr = min([round(vec_j_pos_filtr+Ry);size(mat_X_mes_3D,2)+zeros(1,size(mat_pos_filtr,2))],[],2);
 vec_k_min_pos_filtr = max([round(vec_k_pos_filtr-Rz);ones(1,size(mat_pos_filtr,2))],[],2);
 vec_k_max_pos_filtr = min([round(vec_k_pos_filtr+Rz);size(mat_X_mes_3D,3)+zeros(1,size(mat_pos_filtr,2))],[],2);
end

