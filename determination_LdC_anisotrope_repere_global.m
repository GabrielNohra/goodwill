clear all;
close all;

% repere global (x,y,z)
% repere d'anisotropie (1,2,3)
% P : matrice de passage x_i = Pij*e_j
% P = R_x*R_y*R_z

% notation Voigt : (s_xx;s_yy;s_zz;s_yz;s_xz;s_xy) et (e_xx;e_yy;e_zz;2*e_yz;2*e_xz;2*e_xy) 
% notation Standard : (s_xx;s_yy;s_zz;s_xy;s_xz;s_yz) et (e_xx;e_yy;e_zz;2*e_xy;2*e_xz;2*e_yz) 
%type_notation = 'Voigt';
type_notation = 'Standard';

%type_comportement = 'isotrope';
%type_comportement = 'isotrope_incompressible';
%type_comportement = 'cubique';
%type_comportement = 'isotrope_transverse';
type_comportement = 'orthotrope';

theta_x = sym('theta_x','real');
theta_y = sym('theta_y','real');
theta_z = sym('theta_z','real');

e_xx = sym('e_xx','real');
e_xy = sym('e_xy','real');
e_xz = sym('e_xz','real');
e_yy = sym('e_yy','real');
e_yz = sym('e_yz','real');
e_zz = sym('e_zz','real');

R_x = [1 0 0;0 cos(theta_x) sin(theta_x);0 -sin(theta_x) cos(theta_x)];
R_y = [cos(theta_y) 0 sin(theta_y);0 1 0;-sin(theta_y) 0 cos(theta_y)];
R_z = [cos(theta_z) sin(theta_z) 0;-sin(theta_z) cos(theta_z) 0; 0 0 1];
P = R_z*R_y*R_x;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% s_xx = sym('s_xx','real');
% s_xy = sym('s_xy','real');
% s_xz = sym('s_xz','real');
% s_yy = sym('s_yy','real');
% s_yz = sym('s_yz','real');
% s_zz = sym('s_zz','real');

% for i = 1:6
%  for j = i:6
%   eval(['C_' int2str(i) int2str(j) ' = sym(''C_' int2str(i) int2str(j) ''',''real'');']);
%  end
% end

%%%%%%%%%%%%%%%%%%%%%%%%
% parametre de masse
rho = sym('rho','real');

% comportement isotrope
if ( strcmp(type_comportement,'isotrope') == 1 )
 lambda = sym('lambda','real');
 mu = sym('mu','real');
 liste_parametres = {lambda,mu};
 C_11 = lambda+2*mu;
 C_12 = lambda;
 C_13 = lambda;
 C_14 = 0;
 C_15 = 0;
 C_16 = 0;
 C_22 = lambda+2*mu;
 C_23 = lambda;
 C_24 = 0;
 C_25 = 0;
 C_26 = 0;
 C_33 = lambda+2*mu;
 C_34 = 0;
 C_35 = 0;
 C_36 = 0;
 C_44 = mu;
 C_45 = 0;
 C_46 = 0;
 C_55 = mu;
 C_56 = 0;
 C_66 = mu;
elseif ( strcmp(type_comportement,'isotrope_incompressible') == 1 )
 mu = sym('mu','real');
 liste_parametres = {mu};
 C_11 = 2*mu;
 C_12 = 0;
 C_13 = 0;
 C_14 = 0;
 C_15 = 0;
 C_16 = 0;
 C_22 = 2*mu;
 C_23 = 0;
 C_24 = 0;
 C_25 = 0;
 C_26 = 0;
 C_33 = 2*mu;
 C_34 = 0;
 C_35 = 0;
 C_36 = 0;
 C_44 = mu;
 C_45 = 0;
 C_46 = 0;
 C_55 = mu;
 C_56 = 0;
 C_66 = mu;
elseif ( strcmp(type_comportement,'cubique') == 1 )
 lambda = sym('lambda','real');
 mu = sym('mu','real');
 G = sym('G','real');
 liste_parametres = {lambda,mu,G,theta_x,theta_y,theta_z};
 C_11 = lambda+2*mu;
 C_12 = lambda;
 C_13 = lambda;
 C_14 = 0;
 C_15 = 0;
 C_16 = 0;
 C_22 = lambda+2*mu;
 C_23 = lambda;
 C_24 = 0;
 C_25 = 0;
 C_26 = 0;
 C_33 = lambda+2*mu;
 C_34 = 0;
 C_35 = 0;
 C_36 = 0;
 C_44 = G;
 C_45 = 0;
 C_46 = 0;
 C_55 = G;
 C_56 = 0;
 C_66 = G;
elseif ( strcmp(type_comportement,'isotrope_transverse') == 1 )
 C_11 = sym('C_11','real');
 C_33 = sym('C_33','real');
 C_44 = sym('C_44','real');
 C_12 = sym('C_12','real');
 C_13 = sym('C_13','real');
 liste_parametres = {C_11,C_33,C_44,C_12,C_13,theta_x,theta_y,theta_z};
 C_14 = 0;
 C_15 = 0;
 C_16 = 0;
 C_22 = C_11;
 C_23 = C_13;
 C_24 = 0;
 C_25 = 0;
 C_26 = 0;
 C_34 = 0;
 C_35 = 0;
 C_36 = 0;
 C_45 = 0;
 C_46 = 0;
 C_55 = C_44;
 C_56 = 0;
 C_66 = (C_11-C_12)/2;
elseif ( strcmp(type_comportement,'orthotrope') == 1 ) % https://en.wikipedia.org/wiki/Transverse_isotropy
 C_11 = sym('C_11','real');
 C_22 = sym('C_22','real');
 C_33 = sym('C_33','real');
 C_44 = sym('C_44','real');
 C_55 = sym('C_55','real');
 C_66 = sym('C_66','real');
 C_12 = sym('C_12','real');
 C_13 = sym('C_13','real');
 C_23 = sym('C_23','real');
 liste_parametres = {C_11,C_22,C_33,C_44,C_55,C_66,C_12,C_13,C_23,theta_x,theta_y,theta_z};
 C_14 = 0;
 C_15 = 0;
 C_16 = 0;
 C_24 = 0;
 C_25 = 0;
 C_26 = 0;
 C_34 = 0;
 C_35 = 0;
 C_36 = 0;
 C_45 = 0;
 C_46 = 0;
 C_56 = 0;
end

%C = zeros(6);
for i = 1:6
 for j = i:6
  eval(['C(' int2str(i) ',' int2str(j) ') = C_' int2str(i) int2str(j) ';']);
  eval(['C(' int2str(j) ',' int2str(i) ') = C_' int2str(i) int2str(j) ';']);
 end
end
% % verification de la symetrie de C
% simplify(C-C')


% % matrice de contrainte dans la base des (x_i)
% S_xyz = [s_xx s_xy s_xz ; s_xy s_yy s_yz ; s_xz s_yz s_zz];

% matrice de déformation dans la base des (x_i)
E_xyz = [e_xx e_xy e_xz ; e_xy e_yy e_yz ; e_xz e_yz e_zz];

% matrice de déformation dans la base des (e_i)
E_123 = P'*E_xyz*P;

% matrice de contrainte dans la base des (e_i)
if ( strcmp(type_notation,'Voigt') == 1 )
 vec_E_123 = [E_123(1,1) ; E_123(2,2) ; E_123(3,3) ; 2*E_123(2,3) ; 2*E_123(1,3) ; 2*E_123(1,2)];
elseif ( strcmp(type_notation,'Standard') == 1 )
 vec_E_123 = [E_123(1,1) ; E_123(2,2) ; E_123(3,3) ; 2*E_123(1,2) ; 2*E_123(1,3) ; 2*E_123(2,3)];
end

vec_S_123 = C*vec_E_123;

if ( strcmp(type_notation,'Voigt') == 1 )
 S_123 = [vec_S_123(1) vec_S_123(6) vec_S_123(5) ; vec_S_123(6) vec_S_123(2) vec_S_123(4) ; vec_S_123(5) vec_S_123(4) vec_S_123(3)];
elseif ( strcmp(type_notation,'Standard') == 1 )
 S_123 = [vec_S_123(1) vec_S_123(4) vec_S_123(5) ; vec_S_123(4) vec_S_123(2) vec_S_123(6) ; vec_S_123(5) vec_S_123(6) vec_S_123(3)];
end

% matrice de contrainte dans la base des (x_i)
S_xyz = P*S_123*P';
for i = 1:3
 for j = 1:3
  S_xyz(i,j) = simplify(expand(S_xyz(i,j)));
 end
end

s_xx = S_xyz(1,1);
s_xy = S_xyz(1,2);
s_xz = S_xyz(1,3);
s_yy = S_xyz(2,2);
s_yz = S_xyz(2,3);
s_zz = S_xyz(3,3);

if ( strcmp(type_notation,'Voigt') == 1 )
 Cxyz_11 = simplify(diff(s_xx,e_xx));
 Cxyz_12 = simplify(diff(s_xx,e_yy));
 Cxyz_13 = simplify(diff(s_xx,e_zz));
 Cxyz_14 = simplify(diff(s_xx,e_yz))/2;
 Cxyz_15 = simplify(diff(s_xx,e_xz))/2;
 Cxyz_16 = simplify(diff(s_xx,e_xy))/2;
 Cxyz_21 = simplify(diff(s_yy,e_xx));
 Cxyz_22 = simplify(diff(s_yy,e_yy));
 Cxyz_23 = simplify(diff(s_yy,e_zz));
 Cxyz_24 = simplify(diff(s_yy,e_yz))/2;
 Cxyz_25 = simplify(diff(s_yy,e_xz))/2;
 Cxyz_26 = simplify(diff(s_yy,e_xy))/2;
 Cxyz_31 = simplify(diff(s_zz,e_xx));
 Cxyz_32 = simplify(diff(s_zz,e_yy));
 Cxyz_33 = simplify(diff(s_zz,e_zz));
 Cxyz_34 = simplify(diff(s_zz,e_yz))/2;
 Cxyz_35 = simplify(diff(s_zz,e_xz))/2;
 Cxyz_36 = simplify(diff(s_zz,e_xy))/2;
 Cxyz_41 = simplify(diff(s_yz,e_xx));
 Cxyz_42 = simplify(diff(s_yz,e_yy));
 Cxyz_43 = simplify(diff(s_yz,e_zz));
 Cxyz_44 = simplify(diff(s_yz,e_yz))/2;
 Cxyz_45 = simplify(diff(s_yz,e_xz))/2;
 Cxyz_46 = simplify(diff(s_yz,e_xy))/2;
 Cxyz_51 = simplify(diff(s_xz,e_xx));
 Cxyz_52 = simplify(diff(s_xz,e_yy));
 Cxyz_53 = simplify(diff(s_xz,e_zz));
 Cxyz_54 = simplify(diff(s_xz,e_yz))/2;
 Cxyz_55 = simplify(diff(s_xz,e_xz))/2;
 Cxyz_56 = simplify(diff(s_xz,e_xy))/2;
 Cxyz_61 = simplify(diff(s_xy,e_xx));
 Cxyz_62 = simplify(diff(s_xy,e_yy));
 Cxyz_63 = simplify(diff(s_xy,e_zz));
 Cxyz_64 = simplify(diff(s_xy,e_yz))/2;
 Cxyz_65 = simplify(diff(s_xy,e_xz))/2;
 Cxyz_66 = simplify(diff(s_xy,e_xy))/2;
elseif ( strcmp(type_notation,'Standard') == 1 )
 Cxyz_11 = simplify(diff(s_xx,e_xx));
 Cxyz_12 = simplify(diff(s_xx,e_yy));
 Cxyz_13 = simplify(diff(s_xx,e_zz));
 Cxyz_14 = simplify(diff(s_xx,e_xy))/2;
 Cxyz_15 = simplify(diff(s_xx,e_xz))/2;
 Cxyz_16 = simplify(diff(s_xx,e_yz))/2;
 Cxyz_21 = simplify(diff(s_yy,e_xx));
 Cxyz_22 = simplify(diff(s_yy,e_yy));
 Cxyz_23 = simplify(diff(s_yy,e_zz));
 Cxyz_24 = simplify(diff(s_yy,e_xy))/2;
 Cxyz_25 = simplify(diff(s_yy,e_xz))/2;
 Cxyz_26 = simplify(diff(s_yy,e_yz))/2;
 Cxyz_31 = simplify(diff(s_zz,e_xx));
 Cxyz_32 = simplify(diff(s_zz,e_yy));
 Cxyz_33 = simplify(diff(s_zz,e_zz));
 Cxyz_34 = simplify(diff(s_zz,e_xy))/2;
 Cxyz_35 = simplify(diff(s_zz,e_xz))/2;
 Cxyz_36 = simplify(diff(s_zz,e_yz))/2;
 Cxyz_41 = simplify(diff(s_xy,e_xx));
 Cxyz_42 = simplify(diff(s_xy,e_yy));
 Cxyz_43 = simplify(diff(s_xy,e_zz));
 Cxyz_44 = simplify(diff(s_xy,e_xy))/2;
 Cxyz_45 = simplify(diff(s_xy,e_xz))/2;
 Cxyz_46 = simplify(diff(s_xy,e_yz))/2;
 Cxyz_51 = simplify(diff(s_xz,e_xx));
 Cxyz_52 = simplify(diff(s_xz,e_yy));
 Cxyz_53 = simplify(diff(s_xz,e_zz));
 Cxyz_54 = simplify(diff(s_xz,e_xy))/2;
 Cxyz_55 = simplify(diff(s_xz,e_xz))/2;
 Cxyz_56 = simplify(diff(s_xz,e_yz))/2;
 Cxyz_61 = simplify(diff(s_yz,e_xx));
 Cxyz_62 = simplify(diff(s_yz,e_yy));
 Cxyz_63 = simplify(diff(s_yz,e_zz));
 Cxyz_64 = simplify(diff(s_yz,e_xy))/2;
 Cxyz_65 = simplify(diff(s_yz,e_xz))/2;
 Cxyz_66 = simplify(diff(s_yz,e_yz))/2;
end

for i = 1:6
% for j = i:6
 for j = 1:6
  eval(['Cxyz(' int2str(i) ',' int2str(j) ') = Cxyz_' int2str(i) int2str(j) ';']);
%  eval(['Cxyz(' int2str(j) ',' int2str(i) ') = Cxyz_' int2str(i) int2str(j) ';']);
 end
end

% verification que Cxyz est bien symetrique
mat_test_symetrie_Cxyz = true(size(Cxyz));
for i = 1:6
 for j = i:6
  delta_Cxyz = simplify(Cxyz(i,j)-Cxyz(j,i));
  if ( strcmp(char(delta_Cxyz),'0') ~= 1 )
   mat_test_symetrie_Cxyz(i,j) = false;
  end
 end
end
figure;imagesc(mat_test_symetrie_Cxyz);colorbar;title('test de la symetrie de Cxyz');xlabel('i');ylabel('j');
if ( prod(mat_test_symetrie_Cxyz(:)) == 0 )
 disp('Cxyz NON SYMETRIQUE => ARRET');
 return;
end

% affichage de Cxyz(i,j)
i = 1;
j = 2;
disp(['Cxyz(' int2str(i) ',' int2str(j) ') = ' char(Cxyz(i,j))]);
disp(' ');

if ( strcmp(type_notation,'Voigt') == 1 )
 vec_S_xyz = [S_xyz(1,1) ; S_xyz(2,2) ; S_xyz(3,3) ;   S_xyz(2,3) ;   S_xyz(1,3) ;   S_xyz(1,2)];
 vec_E_xyz = [E_xyz(1,1) ; E_xyz(2,2) ; E_xyz(3,3) ; 2*E_xyz(2,3) ; 2*E_xyz(1,3) ; 2*E_xyz(1,2)];
elseif ( strcmp(type_notation,'Standard') == 1 )
 vec_S_xyz = [S_xyz(1,1) ; S_xyz(2,2) ; S_xyz(3,3) ;   S_xyz(1,2) ;   S_xyz(1,3) ;   S_xyz(2,3)];
 vec_E_xyz = [E_xyz(1,1) ; E_xyz(2,2) ; E_xyz(3,3) ; 2*E_xyz(1,2) ; 2*E_xyz(1,3) ; 2*E_xyz(2,3)];
end
vec_erreur_xyz = simplify(vec_S_xyz-Cxyz*vec_E_xyz);
disp('erreur "S_xyz-C_xyz*E_xyz" = ');
vec_erreur_xyz
disp(' ');

vec_test_erreur_xyz = true(1,length(vec_erreur_xyz));
for i = 1:length(vec_erreur_xyz)
 if ( strcmp(char(vec_erreur_xyz(i)),'0') ~=1 )
  vec_test_erreur_xyz(i) = false;
 end
end
if ( prod(vec_test_erreur_xyz) == 0 )
 disp('erreur "S_xyz-C_xyz*E_xyz" NON NULLE => ARRET');
 return;
end

% k = 4;
% simplify(coeffs(vec_erreur_xyz(k),[e_xx,e_yy,e_zz,e_yz,e_xz,e_xy]))

liste_d_Cxyz_d_p = cell(1,length(liste_parametres));
for n_param = 1:length(liste_parametres)
 for i = 1:6
  for j = i:6
   d_Cxyz_d_p(i,j) = diff(Cxyz(i,j),liste_parametres{n_param});
   d_Cxyz_d_p(j,i) = d_Cxyz_d_p(i,j);
  end
 end
 liste_d_Cxyz_d_p{n_param} = d_Cxyz_d_p;
end

% sortie des resultats

liste_parametres_C = cell(1,length(liste_parametres));
for k = 1:length(liste_parametres)
 liste_parametres_C{k} = char(liste_parametres{k});
end

liste_parametres_M = cell(1,1);
liste_parametres_M{1} = char(rho);

% chaine_f_C = ['f_C = @('];
% for k = 1:length(liste_parametres)
%  chaine_f_C = [chaine_f_C char(liste_parametres{k})];
%  if ( k < length(liste_parametres) )
%   chaine_f_C = [chaine_f_C ','];
%  else
%   chaine_f_C = [chaine_f_C ') '];
%  end
% end
chaine_f_C = ['f_C = @(vec_param) ['];
for k = 1:size(Cxyz,1)
 for l = 1:size(Cxyz,2)
  chaine_Cxyz_kl = char(Cxyz(k,l));
  chaine_Cxyz_kl = strrep(chaine_Cxyz_kl,' ','');
  chaine_f_C = [chaine_f_C chaine_Cxyz_kl ' '];
 end
 if ( k < size(Cxyz,1) )
  chaine_f_C = [chaine_f_C ';'];
 end
end
chaine_f_C = [chaine_f_C '];'];
for k = 1:length(liste_parametres_C)
 chaine_f_C = strrep(chaine_f_C,liste_parametres_C{k},['vec_param(' int2str(k) ')']);
end

% chaine_d_f_C_d_p = ['d_f_C_d_p = @('];
% for k = 1:length(liste_parametres)
%  chaine_d_f_C_d_p = [chaine_d_f_C_d_p char(liste_parametres{k})];
%  if ( k < length(liste_parametres) )
%   chaine_d_f_C_d_p = [chaine_d_f_C_d_p ','];
%  else
%   chaine_d_f_C_d_p = [chaine_d_f_C_d_p ') '];
%  end
% end
chaine_d_f_C_d_p = ['d_f_C_d_p = @(vec_param) reshape(['];
for i = 1:length(liste_d_Cxyz_d_p)
 d_Cxyz_d_p = liste_d_Cxyz_d_p{i};
 for k = 1:size(d_Cxyz_d_p,1)
  for l = 1:size(d_Cxyz_d_p,2)
   chaine_d_Cxyz_d_p_kl = char(d_Cxyz_d_p(k,l));
   chaine_d_Cxyz_d_p_kl = strrep(chaine_d_Cxyz_d_p_kl,' ','');
%   chaine_d_f_C_d_p = [chaine_d_f_C_d_p chaine_d_Cxyz_d_p_kl ' '];
   chaine_d_f_C_d_p = [chaine_d_f_C_d_p chaine_d_Cxyz_d_p_kl];
   if ( l < size(d_Cxyz_d_p,2) )
    chaine_d_f_C_d_p = [chaine_d_f_C_d_p ';'];
   end
  end
  if ( k < size(d_Cxyz_d_p,1) )
   chaine_d_f_C_d_p = [chaine_d_f_C_d_p ';'];
  end
 end
 if ( i < length(liste_d_Cxyz_d_p) )
  chaine_d_f_C_d_p = [chaine_d_f_C_d_p ';'];
 end
end
chaine_d_f_C_d_p = [chaine_d_f_C_d_p '],' int2str(size(Cxyz,1)) ',' int2str(size(Cxyz,2)) ',' int2str(length(liste_d_Cxyz_d_p)) ');'];
for k = 1:length(liste_parametres_C)
 chaine_d_f_C_d_p = strrep(chaine_d_f_C_d_p,liste_parametres_C{k},['vec_param(' int2str(k) ')']);
end

% eval(chaine_d_f_C_d_p);
% toto = d_f_C_d_p(1,1);
% for i = 1:length(liste_d_Cxyz_d_p)
%  figure;imagesc(squeeze(toto(:,:,i)));colorbar;
% end

liste_parametres_C
chaine_f_C
chaine_d_f_C_d_p

fid = fopen('infos_LdC_a_recopier.don','w+');
fprintf(fid,'liste_parametres_comportement = {');
for k = 1:length(liste_parametres_C)
 fprintf(fid,['''' liste_parametres_C{k} '''']);
 if ( k < length(liste_parametres_C) )
  fprintf(fid,',');
 end
end
fprintf(fid,'};\n');
fprintf(fid,'liste_parametres_masse = {');
for k = 1:length(liste_parametres_M)
 fprintf(fid,['''' liste_parametres_M{k} '''']);
 if ( k < length(liste_parametres_M) )
  fprintf(fid,',');
 end
end
fprintf(fid,'};\n');
fprintf(fid,'%s\n',chaine_f_C);
fprintf(fid,'%s\n',chaine_d_f_C_d_p);
fprintf(fid,'liste_LdC{n_comportement}.type_comportement = type_comportement;\n');
fprintf(fid,'liste_LdC{n_comportement}.type_notation = type_notation;\n');
fprintf(fid,'liste_LdC{n_comportement}.liste_parametres_comportement = liste_parametres_comportement;\n');
fprintf(fid,'liste_LdC{n_comportement}.liste_parametres_masse = liste_parametres_masse;\n');
fprintf(fid,'liste_LdC{n_comportement}.f_C = f_C;\n');
fprintf(fid,'liste_LdC{n_comportement}.d_f_C_d_p = d_f_C_d_p;\n');


fclose(fid);

% diff(s_xx,theta_x)

