function liste_elem_ref = creation_elem_ref

nb_dim = 3;
x = sym('x','real');
y = sym('y','real');
z = sym('z','real');

%%%%% D�finition des elements de reference
liste_elem_ref = {};
n_elem_ref = 0;

%%%%% D�finition des elements de reference

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% BRIQUE A 8 NOEUDS CONSTANTE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_elem_ref = n_elem_ref+1;
type_elem_ref = 'HEX1'; % constant 1 noeuds
vec_x_noeuds = [0];
vec_y_noeuds = [0];
vec_z_noeuds = [0];
pos_noeuds = [vec_x_noeuds;vec_y_noeuds;vec_z_noeuds];
nb_noeuds = length(vec_x_noeuds);
liste_expression_Nf = cell(1,nb_noeuds);
nb_noeuds = length(vec_x_noeuds);
liste_expression_Nf = cell(1,nb_noeuds);
liste_expression_Nf{1} = '1';
liste_expression_d_Nf = cell(nb_dim,nb_noeuds);
for i = 1:nb_noeuds
 liste_expression_d_Nf{1,i} = char(diff(liste_expression_Nf{i},x));
 liste_expression_d_Nf{2,i} = char(diff(liste_expression_Nf{i},y));
 liste_expression_d_Nf{3,i} = char(diff(liste_expression_Nf{i},z));
end
for i = 1:nb_noeuds
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'*','.*');
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'/','./');
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'^','.^');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'*','.*');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'/','./');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'^','.^');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'*','.*');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'/','./');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'^','.^');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'*','.*');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'/','./');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'^','.^');
 if ( (isempty(strfind(liste_expression_d_Nf{2,i},'x'))) && (isempty(strfind(liste_expression_d_Nf{2,i},'y'))) )
  liste_expression_d_Nf{1,i} = ['(' liste_expression_d_Nf{1,i} ').*ones(size(x))'];
  liste_expression_d_Nf{2,i} = ['(' liste_expression_d_Nf{2,i} ').*ones(size(x))'];
  liste_expression_d_Nf{3,i} = ['(' liste_expression_d_Nf{3,i} ').*ones(size(x))'];
 end
end
chaine_Nf = '[';
for i = 1:nb_noeuds
 chaine_Nf = [chaine_Nf liste_expression_Nf{i}];
 if ( i == nb_noeuds )
  chaine_Nf = [chaine_Nf ']'];
 else
  chaine_Nf = [chaine_Nf ','];
 end
end
if ( strcmp(chaine_Nf,'[1]') == 1 )
 chaine_Nf = 'ones(size(x))';
elseif ( strcmp(chaine_Nf,'[0]') == 1 )
 chaine_Nf = 'zeros(size(x))';
end
%eval(['f_Nf = @(x,y) (reshape(' chaine_Nf ',size(x,1),size(x,2),nb_noeuds));']);
eval(['f_Nf = @(x,y,z) (reshape(' chaine_Nf ',size(x,1),size(x,2),' int2str(nb_noeuds) '));']);
chaine_d_Nf = '[';
for k = 1:nb_dim
 for i = 1:nb_noeuds
  chaine_d_Nf = [chaine_d_Nf liste_expression_d_Nf{k,i}];
  if ( (i == nb_noeuds) && (k == nb_dim) )
   chaine_d_Nf = [chaine_d_Nf ']'];
  else
   chaine_d_Nf = [chaine_d_Nf ','];
  end
 end
end
eval(['d_f_Nf = @(x,y,z) (reshape(' chaine_d_Nf ',size(x,1),size(x,2),' int2str(nb_noeuds) ',' int2str(nb_dim) '));']);
% liste_expression_Nf{i} = inline(liste_expression_Nf{i},'x','y');
% liste_expression_d_Nf{1,i} = inline(liste_expression_d_Nf{1,i},'x','y');
% liste_expression_d_Nf{2,i} = inline(liste_expression_d_Nf{2,i},'x','y');
clear i;
clear liste_expression_Nf liste_expression_d_Nf chaine_Nf chaine_d_Nf;
% int�gration
% 1 points de Gauss
nb_Gauss = 1;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{1} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 6 points de Gauss
nb_Gauss = 6;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{2} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 8 points de Gauss
nb_Gauss = 8;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{3} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 9 points de Gauss
nb_Gauss = 9;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{4} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 26 points de Gauss
nb_Gauss = 26;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{5} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 27 points de Gauss
nb_Gauss = 27;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{6} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 64 points de Gauss
nb_Gauss = 64;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{7} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
%aretes = [];
%faces = [];
noeuds_aretes(1:4,:) = [1 2; 2 3; 3 4; 1 4];
noeuds_aretes(5:8,:) = [5 6; 6 7; 7 8; 5 8];
noeuds_aretes(9:12,:) = [1 5; 2 6; 3 7; 4 8];
lignes_faces(1,:) = [1 2 3 -4]; % attention num�rotation et sens
lignes_faces(2,:) = [5 6 7 -8]; % attention num�rotation et sens
lignes_faces(3,:) = [1 10 -5 -9]; % attention num�rotation et sens
lignes_faces(4,:) = [2 11 -6 -10]; % attention num�rotation et sens
lignes_faces(5,:) = [3 12 -7 -11]; % attention num�rotation et sens
lignes_faces(6,:) = [4 12 -8 -9]; % attention num�rotation et sens
noeuds_faces(1,:) = [1 2 3 4];
noeuds_faces(2,:) = [5 6 7 8];
noeuds_faces(3,:) = [1 2 5 6];
noeuds_faces(4,:) = [2 3 6 7];
noeuds_faces(5,:) = [3 4 7 8];
noeuds_faces(6,:) = [1 4 5 8];
struct_elem_ref = struct('n_elem',n_elem_ref,'type_elem',type_elem_ref,'nb_noeuds',nb_noeuds,'pos_noeuds',pos_noeuds,'n_noeuds_aretes',noeuds_aretes,'n_lignes_faces',lignes_faces,'n_noeuds_faces',noeuds_faces);
struct_elem_ref.f_Nf = f_Nf;
struct_elem_ref.d_f_Nf = d_f_Nf;
struct_elem_ref.liste_parametres_integration = liste_parametres_integration;
liste_elem_ref{n_elem_ref} = struct_elem_ref;
clear noeuds_aretes;
clear lignes_faces;
clear noeuds_faces;
clear liste_parametres_integration;
clear pos_Gauss;
clear vec_poids_Gauss;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% BRIQUE A 8 NOEUDS LINEAIRE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_elem_ref = n_elem_ref+1;
type_elem_ref = 'HEX8'; % lin�aire 8 noeuds
vec_x_noeuds = [-1  1  1  -1 -1  1 1 -1];
vec_y_noeuds = [-1 -1  1   1 -1 -1 1  1];
vec_z_noeuds = [-1 -1 -1  -1  1  1 1  1];
pos_noeuds = [vec_x_noeuds;vec_y_noeuds;vec_z_noeuds];
nb_noeuds = length(vec_x_noeuds);
liste_expression_Nf = cell(1,nb_noeuds);
nb_noeuds = length(vec_x_noeuds);
liste_expression_Nf = cell(1,nb_noeuds);
liste_expression_Nf{1} = '0.125*(1-x)*(1-y)*(1-z)';
liste_expression_Nf{2} = '0.125*(1+x)*(1-y)*(1-z)';
liste_expression_Nf{3} = '0.125*(1+x)*(1+y)*(1-z)';
liste_expression_Nf{4} = '0.125*(1-x)*(1+y)*(1-z)';
liste_expression_Nf{5} = '0.125*(1-x)*(1-y)*(1+z)';
liste_expression_Nf{6} = '0.125*(1+x)*(1-y)*(1+z)';
liste_expression_Nf{7} = '0.125*(1+x)*(1+y)*(1+z)';
liste_expression_Nf{8} = '0.125*(1-x)*(1+y)*(1+z)';
liste_expression_d_Nf = cell(nb_dim,nb_noeuds);
for i = 1:nb_noeuds
 liste_expression_d_Nf{1,i} = char(diff(liste_expression_Nf{i},x));
 liste_expression_d_Nf{2,i} = char(diff(liste_expression_Nf{i},y));
 liste_expression_d_Nf{3,i} = char(diff(liste_expression_Nf{i},z));
end
for i = 1:nb_noeuds
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'*','.*');
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'/','./');
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'^','.^');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'*','.*');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'/','./');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'^','.^');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'*','.*');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'/','./');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'^','.^');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'*','.*');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'/','./');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'^','.^');
 if ( (isempty(strfind(liste_expression_d_Nf{2,i},'x'))) && (isempty(strfind(liste_expression_d_Nf{2,i},'y'))) )
  liste_expression_d_Nf{1,i} = ['(' liste_expression_d_Nf{1,i} ').*ones(size(x))'];
  liste_expression_d_Nf{2,i} = ['(' liste_expression_d_Nf{2,i} ').*ones(size(x))'];
  liste_expression_d_Nf{3,i} = ['(' liste_expression_d_Nf{3,i} ').*ones(size(x))'];
 end
end
chaine_Nf = '[';
for i = 1:nb_noeuds
 chaine_Nf = [chaine_Nf liste_expression_Nf{i}];
 if ( i == nb_noeuds )
  chaine_Nf = [chaine_Nf ']'];
 else
  chaine_Nf = [chaine_Nf ','];
 end
end
%eval(['f_Nf = @(x,y) (reshape(' chaine_Nf ',size(x,1),size(x,2),nb_noeuds));']);
eval(['f_Nf = @(x,y,z) (reshape(' chaine_Nf ',size(x,1),size(x,2),' int2str(nb_noeuds) '));']);
chaine_d_Nf = '[';
for k = 1:nb_dim
 for i = 1:nb_noeuds
  chaine_d_Nf = [chaine_d_Nf liste_expression_d_Nf{k,i}];
  if ( (i == nb_noeuds) && (k == nb_dim) )
   chaine_d_Nf = [chaine_d_Nf ']'];
  else
   chaine_d_Nf = [chaine_d_Nf ','];
  end
 end
end
eval(['d_f_Nf = @(x,y,z) (reshape(' chaine_d_Nf ',size(x,1),size(x,2),' int2str(nb_noeuds) ',' int2str(nb_dim) '));']);
% liste_expression_Nf{i} = inline(liste_expression_Nf{i},'x','y');
% liste_expression_d_Nf{1,i} = inline(liste_expression_d_Nf{1,i},'x','y');
% liste_expression_d_Nf{2,i} = inline(liste_expression_d_Nf{2,i},'x','y');
clear i;
clear liste_expression_Nf liste_expression_d_Nf chaine_Nf chaine_d_Nf;
% int�gration
% 1 points de Gauss
nb_Gauss = 1;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{1} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 6 points de Gauss
nb_Gauss = 6;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{2} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 8 points de Gauss
nb_Gauss = 8;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{3} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 9 points de Gauss
nb_Gauss = 9;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{4} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 26 points de Gauss
nb_Gauss = 26;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{5} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 27 points de Gauss
nb_Gauss = 27;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{6} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 64 points de Gauss
nb_Gauss = 64;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{7} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
%aretes = [];
%faces = [];
noeuds_aretes(1:4,:) = [1 2; 2 3; 3 4; 1 4];
noeuds_aretes(5:8,:) = [5 6; 6 7; 7 8; 5 8];
noeuds_aretes(9:12,:) = [1 5; 2 6; 3 7; 4 8];
lignes_faces(1,:) = [1 2 3 -4]; % attention num�rotation et sens 
lignes_faces(2,:) = [5 6 7 -8]; % attention num�rotation et sens
lignes_faces(3,:) = [1 10 -5 -9]; % attention num�rotation et sens
lignes_faces(4,:) = [2 11 -6 -10]; % attention num�rotation et sens
lignes_faces(5,:) = [3 12 -7 -11]; % attention num�rotation et sens
lignes_faces(6,:) = [4 12 -8 -9]; % attention num�rotation et sens
noeuds_faces(1,:) = [1 2 3 4];
noeuds_faces(2,:) = [5 6 7 8];
noeuds_faces(3,:) = [1 2 5 6];
noeuds_faces(4,:) = [2 3 6 7];
noeuds_faces(5,:) = [3 4 7 8];
noeuds_faces(6,:) = [1 4 5 8];
struct_elem_ref = struct('n_elem',n_elem_ref,'type_elem',type_elem_ref,'nb_noeuds',nb_noeuds,'pos_noeuds',pos_noeuds,'n_noeuds_aretes',noeuds_aretes,'n_lignes_faces',lignes_faces,'n_noeuds_faces',noeuds_faces);
struct_elem_ref.f_Nf = f_Nf;
struct_elem_ref.d_f_Nf = d_f_Nf;
struct_elem_ref.liste_parametres_integration = liste_parametres_integration;
liste_elem_ref{n_elem_ref} = struct_elem_ref;
clear noeuds_aretes;
clear lignes_faces;
clear noeuds_faces;
clear liste_parametres_integration;
clear pos_Gauss;
clear vec_poids_Gauss;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% BRIQUE A 20 NOEUDS QUADRATIQUE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_elem_ref = n_elem_ref+1;
type_elem_ref = 'HEX20'; % quadratique 20 noeuds
vec_x_noeuds = [-1  1  1  -1 -1  1 1 -1  0  1  0 -1 -1  1 1 -1  0 1  0 -1];
vec_y_noeuds = [-1 -1  1   1 -1 -1 1  1 -1  0  1  0 -1 -1 1  1 -1 0  1  0];
vec_z_noeuds = [-1 -1 -1  -1  1  1 1  1 -1 -1 -1 -1  0  0 0  0  1 1  1 1];
pos_noeuds = [vec_x_noeuds;vec_y_noeuds;vec_z_noeuds];
nb_noeuds = length(vec_x_noeuds);
liste_expression_Nf = cell(1,nb_noeuds);
nb_noeuds = length(vec_x_noeuds);
liste_expression_Nf = cell(1,nb_noeuds);
liste_expression_Nf{1} = '0.125*(1-x)*(1-y)*(1-z)*(-2-x-y-z)';
liste_expression_Nf{2} = '0.125*(1+x)*(1-y)*(1-z)*(-2+x-y-z)';
liste_expression_Nf{3} = '0.125*(1+x)*(1+y)*(1-z)*(-2+x+y-z)';
liste_expression_Nf{4} = '0.125*(1-x)*(1+y)*(1-z)*(-2-x+y-z)';
liste_expression_Nf{5} = '0.125*(1-x)*(1-y)*(1+z)*(-2-x-y+z)';
liste_expression_Nf{6} = '0.125*(1+x)*(1-y)*(1+z)*(-2+x-y+z)';
liste_expression_Nf{7} = '0.125*(1+x)*(1+y)*(1+z)*(-2+x+y+z)';
liste_expression_Nf{8} = '0.125*(1-x)*(1+y)*(1+z)*(-2-x+y+z)';
liste_expression_Nf{9} = '0.25*(1-x^2)*(1-y)*(1-z)';
liste_expression_Nf{10} = '0.25*(1-y^2)*(1+x)*(1-z)';
liste_expression_Nf{11} = '0.25*(1-x^2)*(1+y)*(1-z)';
liste_expression_Nf{12} = '0.25*(1-y^2)*(1-x)*(1-z)';
liste_expression_Nf{13} = '0.25*(1-z^2)*(1-x)*(1-y)';
liste_expression_Nf{14} = '0.25*(1-z^2)*(1+x)*(1-y)';
liste_expression_Nf{15} = '0.25*(1-z^2)*(1+x)*(1+y)';
liste_expression_Nf{16} = '0.25*(1-z^2)*(1-x)*(1+y)';
liste_expression_Nf{17} = '0.25*(1-x^2)*(1-y)*(1+z)';
liste_expression_Nf{18} = '0.25*(1-y^2)*(1+x)*(1+z)';
liste_expression_Nf{19} = '0.25*(1-x^2)*(1+y)*(1+z)';
liste_expression_Nf{20} = '0.25*(1-y^2)*(1-x)*(1+z)';
liste_expression_d_Nf = cell(nb_dim,nb_noeuds);
for i = 1:nb_noeuds
 liste_expression_d_Nf{1,i} = char(diff(liste_expression_Nf{i},x));
 liste_expression_d_Nf{2,i} = char(diff(liste_expression_Nf{i},y));
 liste_expression_d_Nf{3,i} = char(diff(liste_expression_Nf{i},z));
end
for i = 1:nb_noeuds
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'*','.*');
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'/','./');
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'^','.^');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'*','.*');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'/','./');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'^','.^');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'*','.*');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'/','./');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'^','.^');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'*','.*');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'/','./');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'^','.^');
 if ( (isempty(strfind(liste_expression_d_Nf{2,i},'x'))) && (isempty(strfind(liste_expression_d_Nf{2,i},'y'))) )
  liste_expression_d_Nf{1,i} = ['(' liste_expression_d_Nf{1,i} ').*ones(size(x))'];
  liste_expression_d_Nf{2,i} = ['(' liste_expression_d_Nf{2,i} ').*ones(size(x))'];
  liste_expression_d_Nf{3,i} = ['(' liste_expression_d_Nf{3,i} ').*ones(size(x))'];
 end
end
chaine_Nf = '[';
for i = 1:nb_noeuds
 chaine_Nf = [chaine_Nf liste_expression_Nf{i}];
 if ( i == nb_noeuds )
  chaine_Nf = [chaine_Nf ']'];
 else
  chaine_Nf = [chaine_Nf ','];
 end
end
%eval(['f_Nf = @(x,y) (reshape(' chaine_Nf ',size(x,1),size(x,2),nb_noeuds));']);
eval(['f_Nf = @(x,y,z) (reshape(' chaine_Nf ',size(x,1),size(x,2),' int2str(nb_noeuds) '));']);
chaine_d_Nf = '[';
for k = 1:nb_dim
 for i = 1:nb_noeuds
  chaine_d_Nf = [chaine_d_Nf liste_expression_d_Nf{k,i}];
  if ( (i == nb_noeuds) && (k == nb_dim) )
   chaine_d_Nf = [chaine_d_Nf ']'];
  else
   chaine_d_Nf = [chaine_d_Nf ','];
  end
 end
end
eval(['d_f_Nf = @(x,y,z) (reshape(' chaine_d_Nf ',size(x,1),size(x,2),' int2str(nb_noeuds) ',' int2str(nb_dim) '));']);
% liste_expression_Nf{i} = inline(liste_expression_Nf{i},'x','y');
% liste_expression_d_Nf{1,i} = inline(liste_expression_d_Nf{1,i},'x','y');
% liste_expression_d_Nf{2,i} = inline(liste_expression_d_Nf{2,i},'x','y');
clear i;
clear liste_expression_Nf liste_expression_d_Nf chaine_Nf chaine_d_Nf;
% int�gration
% 1 points de Gauss
nb_Gauss = 1;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{1} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 6 points de Gauss
nb_Gauss = 6;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{2} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 8 points de Gauss
nb_Gauss = 8;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{3} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 9 points de Gauss
nb_Gauss = 9;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{4} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 26 points de Gauss
nb_Gauss = 26;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{5} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 27 points de Gauss
nb_Gauss = 27;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{6} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 64 points de Gauss
nb_Gauss = 64;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{7} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
noeuds_aretes(1:4,:) = [1 9 2 ; 2 10 3 ; 3 11 4 ; 1 12 4];
noeuds_aretes(5:8,:) = [5 17 6 ; 6 18 7 ; 7 19 8; 5 20 8];
noeuds_aretes(9:12,:) = [1 13 5 ; 2 14 6 ; 3 15 7; 4 16 8];
lignes_faces(1,:) = [1 2 3 -4]; % attention num�rotation et sens
lignes_faces(2,:) = [5 6 7 -8]; % attention num�rotation et sens
lignes_faces(3,:) = [1 10 -5 -9]; % attention num�rotation et sens
lignes_faces(4,:) = [2 11 -6 -10]; % attention num�rotation et sens
lignes_faces(5,:) = [3 12 -7 -11]; % attention num�rotation et sens
lignes_faces(6,:) = [4 -9 -8 12]; % attention num�rotation et sens
noeuds_faces(1,:) = [1 2 3 4 9 10 11 12];
noeuds_faces(2,:) = [5 6 7 8 17 18 19 20];
noeuds_faces(3,:) = [1 2 5 6 9 13 14 17];
noeuds_faces(4,:) = [2 3 6 7 10 14 15 18];
noeuds_faces(5,:) = [3 4 7 8 11 15 16 19];
noeuds_faces(6,:) = [1 4 5 8 12 13 16 20];
struct_elem_ref = struct('n_elem',n_elem_ref,'type_elem',type_elem_ref,'nb_noeuds',nb_noeuds,'pos_noeuds',pos_noeuds,'n_noeuds_aretes',noeuds_aretes,'n_lignes_faces',lignes_faces,'n_noeuds_faces',noeuds_faces);
struct_elem_ref.f_Nf = f_Nf;
struct_elem_ref.d_f_Nf = d_f_Nf;
struct_elem_ref.liste_parametres_integration = liste_parametres_integration;
liste_elem_ref{n_elem_ref} = struct_elem_ref;
clear noeuds_aretes;
clear lignes_faces;
clear noeuds_faces;
clear liste_parametres_integration;
clear pos_Gauss;
clear vec_poids_Gauss;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% BRIQUE A 27 NOEUDS QUADRATIQUE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_elem_ref = n_elem_ref+1;
type_elem_ref = 'HEX27'; % quadratique 27 noeuds
%vec_x_noeuds = [0 1  1  0 0 1  1 0 0.5  1  0.5  0 0.5  1 0.5 0 0 1  1 0]; % num�rotation MAJ
%vec_y_noeuds = [0 0 1  1  0 0 1 1  0 0.5  1  0.5  0 0.5 1 0.5  0 0 1 1];
%vec_z_noeuds = [0 0 0 0 1  1  1 1  0 0 0 0 1  1 1 1  0.5  0.5  0.5 0.5];
vec_x_noeuds = [-1  1  1  -1 -1  1 1 -1  0  1  0 -1 -1  1 1 -1  0 1 0 -1  0  0 1 0 -1 0 0]; % num�rotation MAJ
vec_y_noeuds = [-1 -1  1   1 -1 -1 1  1 -1  0  1  0 -1 -1 1  1 -1 0 1  0  0 -1 0 1  0 0 0];
vec_z_noeuds = [-1 -1 -1  -1  1  1 1  1 -1 -1 -1 -1  0  0 0  0  1 1 1 1 -1  0 0 0  0 1 0];
pos_noeuds = [vec_x_noeuds;vec_y_noeuds;vec_z_noeuds];
nb_noeuds = length(vec_x_noeuds);
liste_expression_Nf = cell(1,nb_noeuds);
liste_expression_Nf{1} = '0.125*x*(x-1)*y*(y-1)*z*(z-1)';
liste_expression_Nf{2} = '0.125*x*(x+1)*y*(y-1)*z*(z-1)';
liste_expression_Nf{3} = '0.125*x*(x+1)*y*(y+1)*z*(z-1)';
liste_expression_Nf{4} = '0.125*x*(x-1)*y*(y+1)*z*(z-1)';
liste_expression_Nf{5} = '0.125*x*(x-1)*y*(y-1)*z*(z+1)';
liste_expression_Nf{6} = '0.125*x*(x+1)*y*(y-1)*z*(z+1)';
liste_expression_Nf{7} = '0.125*x*(x+1)*y*(y+1)*z*(z+1)';
liste_expression_Nf{8} = '0.125*x*(x-1)*y*(y+1)*z*(z+1)';
liste_expression_Nf{9} = '0.25*(1-x^2)*y*(y-1)*z*(z-1)';
liste_expression_Nf{10} = '0.25*x*(x+1)*(1-y^2)*z*(z-1)';
liste_expression_Nf{11} = '0.25*(1-x^2)*y*(y+1)*z*(z-1)';
liste_expression_Nf{12} = '0.25*x*(x-1)*(1-y^2)*z*(z-1)';
liste_expression_Nf{13} = '0.25*x*(x-1)*y*(y-1)*(1-z^2)';
liste_expression_Nf{14} = '0.25*x*(x+1)*y*(y-1)*(1-z^2)';
liste_expression_Nf{15} = '0.25*x*(x+1)*y*(y+1)*(1-z^2)';
liste_expression_Nf{16} = '0.25*x*(x-1)*y*(y+1)*(1-z^2)';
liste_expression_Nf{17} = '0.25*(1-x^2)*y*(y-1)*z*(z+1)';
liste_expression_Nf{18} = '0.25*x*(x+1)*(1-y^2)*z*(z+1)';
liste_expression_Nf{19} = '0.25*(1-x^2)*y*(y+1)*z*(z+1)';
liste_expression_Nf{20} = '0.25*x*(x-1)*(1-y^2)*z*(z+1)';
liste_expression_Nf{21} = '0.5*(1-x^2)*(1-y^2)*z*(z-1)';
liste_expression_Nf{22} = '0.5*(1-x^2)*y*(y-1)*(1-z^2)';
liste_expression_Nf{23} = '0.5*x*(x+1)*(1-y^2)*(1-z^2)';
liste_expression_Nf{24} = '0.5*(1-x^2)*y*(y+1)*(1-z^2)';
liste_expression_Nf{25} = '0.5*x*(x-1)*(1-y^2)*(1-z^2)';
liste_expression_Nf{26} = '0.5*(1-x^2)*(1-y^2)*z*(z+1)';
liste_expression_Nf{27} = '(1-x^2)*(1-y^2)*(1-z^2)';
liste_expression_d_Nf = cell(nb_dim,nb_noeuds);
for i = 1:nb_noeuds
 liste_expression_d_Nf{1,i} = char(diff(liste_expression_Nf{i},x));
 liste_expression_d_Nf{2,i} = char(diff(liste_expression_Nf{i},y));
 liste_expression_d_Nf{3,i} = char(diff(liste_expression_Nf{i},z));
end
for i = 1:nb_noeuds
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'*','.*');
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'/','./');
 liste_expression_Nf{i} = strrep(liste_expression_Nf{i},'^','.^');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'*','.*');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'/','./');
 liste_expression_d_Nf{1,i} = strrep(liste_expression_d_Nf{1,i},'^','.^');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'*','.*');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'/','./');
 liste_expression_d_Nf{2,i} = strrep(liste_expression_d_Nf{2,i},'^','.^');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'*','.*');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'/','./');
 liste_expression_d_Nf{3,i} = strrep(liste_expression_d_Nf{3,i},'^','.^');
 if ( (isempty(strfind(liste_expression_d_Nf{2,i},'x'))) && (isempty(strfind(liste_expression_d_Nf{2,i},'y'))) )
  liste_expression_d_Nf{1,i} = ['(' liste_expression_d_Nf{1,i} ').*ones(size(x))'];
  liste_expression_d_Nf{2,i} = ['(' liste_expression_d_Nf{2,i} ').*ones(size(x))'];
  liste_expression_d_Nf{3,i} = ['(' liste_expression_d_Nf{3,i} ').*ones(size(x))'];
 end
end
chaine_Nf = '[';
for i = 1:nb_noeuds
 chaine_Nf = [chaine_Nf liste_expression_Nf{i}];
 if ( i == nb_noeuds )
  chaine_Nf = [chaine_Nf ']'];
 else
  chaine_Nf = [chaine_Nf ','];
 end
end
%eval(['f_Nf = @(x,y) (reshape(' chaine_Nf ',size(x,1),size(x,2),nb_noeuds));']);
eval(['f_Nf = @(x,y,z) (reshape(' chaine_Nf ',size(x,1),size(x,2),' int2str(nb_noeuds) '));']);
chaine_d_Nf = '[';
for k = 1:nb_dim
 for i = 1:nb_noeuds
  chaine_d_Nf = [chaine_d_Nf liste_expression_d_Nf{k,i}];
  if ( (i == nb_noeuds) && (k == nb_dim) )
   chaine_d_Nf = [chaine_d_Nf ']'];
  else
   chaine_d_Nf = [chaine_d_Nf ','];
  end
 end
end
eval(['d_f_Nf = @(x,y,z) (reshape(' chaine_d_Nf ',size(x,1),size(x,2),' int2str(nb_noeuds) ',' int2str(nb_dim) '));']);
% liste_expression_Nf{i} = inline(liste_expression_Nf{i},'x','y');
% liste_expression_d_Nf{1,i} = inline(liste_expression_d_Nf{1,i},'x','y');
% liste_expression_d_Nf{2,i} = inline(liste_expression_d_Nf{2,i},'x','y');
clear i;
clear liste_expression_Nf liste_expression_d_Nf chaine_Nf chaine_d_Nf;
% int�gration
% 1 points de Gauss
nb_Gauss = 1;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{1} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 6 points de Gauss
nb_Gauss = 6;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{2} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 8 points de Gauss
nb_Gauss = 8;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{3} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 9 points de Gauss
nb_Gauss = 9;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{4} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 26 points de Gauss
nb_Gauss = 26;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{5} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 27 points de Gauss
nb_Gauss = 27;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{6} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
% 64 points de Gauss
nb_Gauss = 64;
[pos_Gauss,vec_poids_Gauss] = data_gauss (nb_Gauss);
liste_parametres_integration{7} = struct('nb_Gauss',nb_Gauss,'pos_Gauss',pos_Gauss,'poids_Gauss',vec_poids_Gauss);
aretes = [];
faces = [];
% aretes(1:8,:) = [1 9 ; 9 2 ; 2 10 ; 10 3 ; 3 11 ; 11 4; 4 12 ; 12 1];
% aretes(9:16,:) = [5 17 ; 17 6 ; 6 18 ; 18 7 ; 7 19 ; 19 8 ; 8 20 ; 20 5];
% aretes(17:24,:) = [13 22 ; 22 14 ; 14 23; 23 15 ; 15 24 ; 24 16 ; 16 25 ; 25 13];
% aretes(25:32,:) = [1 13 ; 13 5 ; 2 14 ; 14 6 ; 3 15 ; 15 7 ; 4 16 ; 16 8];
% faces(1,:) = [1 2 3 4 5 6 7 8]; % attention num�rotation et sens 
% faces(2,:) = [17 18 19 20 21 22 23 24];
% faces(3,:) = [17 18 9 10 -20 -19 -2 -1];
% faces(4,:) = [19 20 11 12 -22 -21 -4 -3];
% faces(5,:) = [21 22 13 14 -24 -23 -6 -5];
% faces(6,:) = [23 24 15 16 -18 -17 -8 -7];
noeuds_aretes(1:4,:) = [1 9 2 ; 2 10 3 ; 3 11 4 ; 1 12 4];
noeuds_aretes(5:8,:) = [5 17 6 ; 6 18 7 ; 7 19 8; 5 20 8];
noeuds_aretes(9:12,:) = [1 13 5 ; 2 14 6 ; 3 15 7; 4 16 8];
lignes_faces(1,:) = [1 2 3 -4]; % attention num�rotation et sens
lignes_faces(2,:) = [5 6 7 -8]; % attention num�rotation et sens
lignes_faces(3,:) = [1 10 -5 -9]; % attention num�rotation et sens
lignes_faces(4,:) = [2 11 -6 -10]; % attention num�rotation et sens
lignes_faces(5,:) = [3 12 -7 -11]; % attention num�rotation et sens
lignes_faces(6,:) = [4 -9 -8 12]; % attention num�rotation et sens
noeuds_faces(1,:) = [1 2 3 4 9 10 11 12 21];
noeuds_faces(2,:) = [5 6 7 8 17 18 19 20 26];
noeuds_faces(3,:) = [1 2 5 6 9 13 14 17 22];
noeuds_faces(4,:) = [2 3 6 7 10 14 15 18 23];
noeuds_faces(5,:) = [3 4 7 8 11 15 16 19 24];
noeuds_faces(6,:) = [1 4 5 8 12 13 16 20 25];
struct_elem_ref = struct('n_elem',n_elem_ref,'type_elem',type_elem_ref,'nb_noeuds',nb_noeuds,'pos_noeuds',pos_noeuds,'n_noeuds_aretes',noeuds_aretes,'n_lignes_faces',lignes_faces,'n_noeuds_faces',noeuds_faces);
struct_elem_ref.f_Nf = f_Nf;
struct_elem_ref.d_f_Nf = d_f_Nf;
struct_elem_ref.liste_parametres_integration = liste_parametres_integration;
liste_elem_ref{n_elem_ref} = struct_elem_ref;
clear noeuds_aretes;
clear lignes_faces;
clear noeuds_faces;
clear liste_parametres_integration;
clear pos_Gauss;
clear vec_poids_Gauss;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% CALCUL DES FONCTIONS DE FORMES AUX POINTS DE GAUSS POUR TOUS LES
%%%% PARAMETRES D'INTEGRATION ET POUR TOUS LES ELEMENTS DE REFERENCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n_elem_ref = 1:length(liste_elem_ref)
 struct_elem_ref = liste_elem_ref{n_elem_ref};
 nb_noeuds_elem_ref = struct_elem_ref.nb_noeuds;
 nb_dim_elem_ref = size(struct_elem_ref.pos_noeuds,1);
 liste_parametres_integration = struct_elem_ref.liste_parametres_integration;
 for n_parametre_integration = 1:length(liste_parametres_integration)
  struct_param_integration = struct_elem_ref.liste_parametres_integration{n_parametre_integration};
  w = struct_param_integration.poids_Gauss;
  p_G = struct_param_integration.pos_Gauss;
  Nf_global = reshape(squeeze(struct_elem_ref.f_Nf(p_G(:,1),p_G(:,2),p_G(:,3))),size(p_G,1),nb_noeuds_elem_ref);
  d_Nf_global = reshape(squeeze(struct_elem_ref.d_f_Nf(p_G(:,1),p_G(:,2),p_G(:,3))),size(p_G,1),nb_noeuds_elem_ref,nb_dim_elem_ref);
  struct_param_integration.Nf_G = Nf_global;
  struct_param_integration.d_Nf_G = d_Nf_global;
  liste_parametres_integration{n_parametre_integration} = struct_param_integration;
 end
 struct_elem_ref.liste_parametres_integration = liste_parametres_integration;
 liste_elem_ref{n_elem_ref} = struct_elem_ref;
end


