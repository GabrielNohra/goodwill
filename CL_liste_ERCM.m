function [liste_DL_bloque] = CL_liste_ERCM(liste_CL_data,mat_pos_sig,mat_pos_mes,mat_U_mes,parametres_filtrage_dep,nb_DDL_par_noeud)

nb_ddl_tot = size(mat_pos_sig,1)*nb_DDL_par_noeud;

vec_test_DL_bloque = ones(1,nb_ddl_tot);

liste_DL_bloque = cell(1,length(liste_CL_data));

for k = 1:length(liste_CL_data)
 CL_data = liste_CL_data{k};
 val_DL_bloque = CL_data.val_DL_bloque;
 DL_bloque = CL_data.DL_bloque;
 type_entite = CL_data.type_entite;
 caracteristique_entite = CL_data.caracteristique_entite;
 if ( strcmp(type_entite,'plan') == 1 )
  vec_n = caracteristique_entite.vec_n;
  vec_point = caracteristique_entite.point;
  tolerance = caracteristique_entite.tolerance;
  a = vec_n(1);
  b = vec_n(2);
  c = vec_n(3);
  d = -dot(vec_n,vec_point);
  vec_distance = abs(a*mat_pos_sig(:,1)+b*mat_pos_sig(:,2)+c*mat_pos_sig(:,3)+d)/norm(vec_n);
  vec_n_noeud = find ( vec_distance <= tolerance);
  if ( (strcmp(val_DL_bloque,'Um') == 1) || (strcmp(val_DL_bloque,'Um_filtre') == 1) )
   vec_distance_mes = abs(a*mat_pos_mes(1,:)'+b*mat_pos_mes(2,:)'+c*mat_pos_mes(3,:)'+d)/norm(vec_n);
   vec_n_noeud_mes = find ( vec_distance_mes <= tolerance);
  end
 elseif ( strcmp(type_entite,'droite') == 1 )
  vec_d = caracteristique_entite.vec_d;
  if ( ~iscolumn(vec_d) )
   vec_d = vec_d.';
  end
  vec_point = caracteristique_entite.point;
  tolerance = caracteristique_entite.tolerance;
  mat_AM = [(mat_pos_sig(:,1)-vec_point(1))';(mat_pos_sig(:,2)-vec_point(2))';(mat_pos_sig(:,3)-vec_point(3))'];
  mat_u = vec_d*ones(1,size(mat_pos_sig,1));
  mat_AM_vectoriel_u = cross(mat_AM,mat_u);
  vec_distance = sqrt(mat_AM_vectoriel_u(1,:).^2+mat_AM_vectoriel_u(2,:).^2+mat_AM_vectoriel_u(3,:).^2)./sqrt(mat_u(1,:).^2+mat_u(2,:).^2+mat_u(3,:).^2);
  [vec_n_noeud] = find ( vec_distance <= tolerance)';
%   if ( isempty(vec_n_noeud) ) % risque, mais assure au moins un point
%    find (vec_distance == min(vec_distance));
%   end
  if ( (strcmp(val_DL_bloque,'Um') == 1) || (strcmp(val_DL_bloque,'Um_filtre') == 1) )
   mat_AM_mes = [(mat_pos_mes(1,:)'-vec_point(1))';(mat_pos_mes(2,:)'-vec_point(2))';(mat_pos_mes(3,:)'-vec_point(3))'];
   mat_u_mes = vec_d*ones(1,size(mat_pos_mes,2));
   mat_AM_vectoriel_u_mes = cross(mat_AM_mes,mat_u_mes);
   vec_distance_mes = sqrt(mat_AM_vectoriel_u_mes(1,:).^2+mat_AM_vectoriel_u_mes(2,:).^2+mat_AM_vectoriel_u_mes(3,:).^2)./sqrt(mat_AM_vectoriel_u_mes(1,:).^2+mat_AM_vectoriel_u_mes(2,:).^2+mat_AM_vectoriel_u_mes(3,:).^2);
   [vec_n_noeud_mes] = find ( vec_distance_mes <= tolerance)';
  end
 elseif ( strcmp(type_entite,'point') == 1 )
  vec_point = caracteristique_entite.point;
  tolerance = caracteristique_entite.tolerance;
  vec_distance = sqrt((mat_pos_sig(:,1)-vec_point(1)).^2+(mat_pos_sig(:,2)-vec_point(2)).^2+(mat_pos_sig(:,3)-vec_point(3)).^2);
  [vec_n_noeud] = find ( vec_distance <= tolerance); % bonne methode
%  [vec_n_noeud] = find ( vec_distance == min(vec_distance )); % mauvais, ajuster la tolerance ou revoir le maillage
  if ( (strcmp(val_DL_bloque,'Um') == 1) || (strcmp(val_DL_bloque,'Um_filtre') == 1) )
   vec_distance_mes = sqrt((mat_pos_mes(1,:)'-vec_point(1)).^2+(mat_pos_mes(2,:)'-vec_point(2)).^2+(mat_pos_mes(3,:)'-vec_point(3)).^2);
   [vec_n_noeud_mes] = find ( vec_distance_mes <= tolerance); % bonne methode
  end
 end

% determination des numeros de DL bloques
 if ( ~isempty(vec_n_noeud) )
  if ( strcmp(DL_bloque(2),'x') == 1 )
   vec_n_DL_bloque = (vec_n_noeud-1)*nb_DDL_par_noeud+1;
   n_DL_imp = 1;
  elseif ( strcmp(DL_bloque(2),'y') == 1 )
   vec_n_DL_bloque = (vec_n_noeud-1)*nb_DDL_par_noeud+2;
   n_DL_imp = 2;
  elseif ( strcmp(DL_bloque(2),'z') == 1 )
   vec_n_DL_bloque = (vec_n_noeud-1)*nb_DDL_par_noeud+3;
   n_DL_imp = 3;
  end
  if ( ~isempty(vec_n_DL_bloque) )
   [ii] = find ( vec_test_DL_bloque(vec_n_DL_bloque) == 1 );
   if ( ~isempty(ii) )
    vec_n_DL_bloque = vec_n_DL_bloque(ii);
    vec_n_noeud = vec_n_noeud(ii);
    vec_test_DL_bloque(vec_n_DL_bloque) = 0;
   else
    vec_n_noeud = [];
    vec_n_DL_bloque = [];
   end
  else
   vec_n_noeud = [];
   n_DL_imp = [];
  end
 else
  vec_n_DL_bloque = [];
  n_DL_imp = [];
 end

 if ( strcmp(val_DL_bloque,'mesure') == 1 )
%  
%   if ( ~isempty(vec_n_noeud) )
%    if ( strcmp(DL_bloque(2),'x') == 1 )
%     indice_DDL = 1;
%    elseif ( strcmp(DL_bloque(2),'y') == 1 )
%     indice_DDL = 2;
%    elseif ( strcmp(DL_bloque(2),'z') == 1 )
%     indice_DDL = 3;
%    end
%    vec_n_DL_bloque = (vec_n_noeud-1)*nb_DDL_par_noeud+indice_DDL;
%   else
%    vec_n_DL_bloque = [];
%   end
%   if ( ~isempty(vec_n_DL_bloque) )
%    [ii] = find ( vec_test_DL_bloque(vec_n_DL_bloque) == 1 );
%    if ( ~isempty(ii) )
%     vec_n_DL_bloque = vec_n_DL_bloque(ii);
%     vec_test_DL_bloque(vec_n_DL_bloque) = 0;
%    else
%     vec_n_DL_bloque = [];
%    end
%    vec_n_noeud = round((vec_n_DL_bloque-indice_DDL)/nb_DDL_par_noeud)+1;
%    vec_val_DL_bloque = mat_U_mes(indice_DDL,vec_n_noeud);
%   end
%  
 elseif ( strcmp(val_DL_bloque,'Um') == 1 )
  if ( ~isempty(n_DL_imp) )
   vec_X_bloque = mat_pos_mes(1,vec_n_noeud_mes);
   vec_Y_bloque = mat_pos_mes(2,vec_n_noeud_mes);
   vec_Z_bloque = mat_pos_mes(3,vec_n_noeud_mes);
   vec_U_bloque = mat_U_mes(n_DL_imp,vec_n_noeud_mes);
   dX = max(vec_X_bloque)-min(vec_X_bloque);
   dY = max(vec_Y_bloque)-min(vec_Y_bloque);
   dZ = max(vec_Z_bloque)-min(vec_Z_bloque);
   if ( dX < min([dY dZ]) ) % fit dans plan (Y,Z)
    F_r = scatteredInterpolant(vec_Y_bloque',vec_Z_bloque',real(vec_U_bloque'),'linear','linear');
    F_i = scatteredInterpolant(vec_Y_bloque',vec_Z_bloque',imag(vec_U_bloque'),'linear','linear');
    vec_val_DL_bloque = F_r(mat_pos_sig(vec_n_noeud,2)',mat_pos_sig(vec_n_noeud,3)')+1i*F_i(mat_pos_sig(vec_n_noeud,2)',mat_pos_sig(vec_n_noeud,3)');
   elseif ( dY < min([dZ dX]) ) % fit dans plan (X,Z)
    F_r = scatteredInterpolant(vec_X_bloque',vec_Z_bloque',real(vec_U_bloque'),'linear','linear');
    F_i = scatteredInterpolant(vec_X_bloque',vec_Z_bloque',imag(vec_U_bloque'),'linear','linear');
    vec_val_DL_bloque = F_r(mat_pos_sig(vec_n_noeud,1)',mat_pos_sig(vec_n_noeud,3)')+1i*F_i(mat_pos_sig(vec_n_noeud,1)',mat_pos_sig(vec_n_noeud,3)');
   else % fit dans plan (X,Y)
    F_r = scatteredInterpolant(vec_X_bloque',vec_Y_bloque',real(vec_U_bloque'),'linear','linear');
    F_i = scatteredInterpolant(vec_X_bloque',vec_Y_bloque',imag(vec_U_bloque'),'linear','linear');
    vec_val_DL_bloque = F_r(mat_pos_sig(vec_n_noeud,1)',mat_pos_sig(vec_n_noeud,2)')+1i*F_i(mat_pos_sig(vec_n_noeud,1)',mat_pos_sig(vec_n_noeud,2)');
   end
  end
 elseif ( strcmp(val_DL_bloque,'Um_filtre') == 1 )
  if ( ~isempty(n_DL_imp) )
   vec_X_bloque = mat_pos_mes(1,vec_n_noeud_mes);
   vec_Y_bloque = mat_pos_mes(2,vec_n_noeud_mes);
   vec_Z_bloque = mat_pos_mes(3,vec_n_noeud_mes);
   vec_U_bloque = mat_U_mes(n_DL_imp,vec_n_noeud_mes);
   dX = max(vec_X_bloque)-min(vec_X_bloque);
   dY = max(vec_Y_bloque)-min(vec_Y_bloque);
   dZ = max(vec_Z_bloque)-min(vec_Z_bloque);
   if ( dX < min([dY dZ]) ) % fit dans plan (Y,Z)
    coef_r = fit_polynomial_2D(vec_Y_bloque,vec_Z_bloque,real(vec_U_bloque),parametres_filtrage_dep.degre,parametres_filtrage_dep.degre,[],[],[]);
    coef_i = fit_polynomial_2D(vec_Y_bloque,vec_Z_bloque,imag(vec_U_bloque),parametres_filtrage_dep.degre,parametres_filtrage_dep.degre,[],[],[]);
    vec_val_DL_bloque = valeur_polynome(coef_r,mat_pos_sig(vec_n_noeud,2)',mat_pos_sig(vec_n_noeud,3)')+1i*valeur_polynome(coef_i,mat_pos_sig(vec_n_noeud,2)',mat_pos_sig(vec_n_noeud,3)');
   elseif ( dY < min([dZ dX]) ) % fit dans plan (X,Z)
    coef_r = fit_polynomial_2D(vec_X_bloque,vec_Z_bloque,real(vec_U_bloque),parametres_filtrage_dep.degre,parametres_filtrage_dep.degre,[],[],[]);
    coef_i = fit_polynomial_2D(vec_X_bloque,vec_Z_bloque,imag(vec_U_bloque),parametres_filtrage_dep.degre,parametres_filtrage_dep.degre,[],[],[]);
    vec_val_DL_bloque = valeur_polynome(coef_r,mat_pos_sig(vec_n_noeud,1)',mat_pos_sig(vec_n_noeud,3)')+1i*valeur_polynome(coef_i,mat_pos_sig(vec_n_noeud,1)',mat_pos_sig(vec_n_noeud,3)');
   else % fit dans plan (X,Y)
    coef_r = fit_polynomial_2D(vec_X_bloque,vec_Y_bloque,real(vec_U_bloque),parametres_filtrage_dep.degre,parametres_filtrage_dep.degre,[],[],[]);
    coef_i = fit_polynomial_2D(vec_X_bloque,vec_Y_bloque,imag(vec_U_bloque),parametres_filtrage_dep.degre,parametres_filtrage_dep.degre,[],[],[]);
    vec_val_DL_bloque = valeur_polynome(coef_r,mat_pos_sig(vec_n_noeud,1)',mat_pos_sig(vec_n_noeud,2)')+1i*valeur_polynome(coef_i,mat_pos_sig(vec_n_noeud,1)',mat_pos_sig(vec_n_noeud,2)');
   end
  end
 elseif ( isa(val_DL_bloque,'double') ) % on a une valeur 
%   if ( ~isempty(vec_n_noeud) )
%    if ( strcmp(DL_bloque(2),'x') == 1 )
%     vec_n_DL_bloque = (vec_n_noeud-1)*nb_DDL_par_noeud+1;
%    elseif ( strcmp(DL_bloque(2),'y') == 1 )
%     vec_n_DL_bloque = (vec_n_noeud-1)*nb_DDL_par_noeud+2;
%    elseif ( strcmp(DL_bloque(2),'z') == 1 )
%     vec_n_DL_bloque = (vec_n_noeud-1)*nb_DDL_par_noeud+3;
%    end
%   else
%    vec_n_DL_bloque = [];
%   end;
%   if ( ~isempty(vec_n_DL_bloque) )
%    [ii] = find ( vec_test_DL_bloque(vec_n_DL_bloque) == 1 );
%    if ( ~isempty(ii) )
%     vec_n_DL_bloque = vec_n_DL_bloque(ii);
%     vec_n_noeud = vec_n_noeud(ii);
%     vec_test_DL_bloque(vec_n_DL_bloque) = 0;
%    else
%     vec_n_DL_bloque = [];
%     vec_n_noeud = [];
%    end
%   end
  vec_val_DL_bloque = val_DL_bloque+zeros(1,length(vec_n_DL_bloque));
 else
     disp(['Mauvaise définition des CL']);
 end

 liste_DL_bloque{k} = struct('type',CL_data.DL_bloque(1),'vec_n_DL_bloque',vec_n_DL_bloque','vec_val_DL_bloque',vec_val_DL_bloque,'vec_n_noeud',vec_n_noeud');
 if ( (strcmp(val_DL_bloque,'Um') == 1) || (strcmp(val_DL_bloque,'Um_filtre') == 1) )
  liste_DL_bloque{k}.vec_n_noeud_mes = vec_n_noeud_mes;
 end
end
