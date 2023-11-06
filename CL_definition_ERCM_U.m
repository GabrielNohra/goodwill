function [liste_CL_data_U] = CL_definition_ERCM_U(mat_pos_sig,tolerance_position)

x_min = min(mat_pos_sig(:,1));
y_min = min(mat_pos_sig(:,2));
z_min = min(mat_pos_sig(:,3));

x_max = max(mat_pos_sig(:,1));
y_max = max(mat_pos_sig(:,2));
z_max = max(mat_pos_sig(:,3));

liste_CL_data_U = {};

% %%%%%%%%%%%%%%%%%%%%%%%%
% % 0) test resolution CL : U = 0 sur 3 faces et U = 1 sur 1 face 
% %%%%%%%%%%%%%%%%%%%%%%%%
% vec_type_CL_local = ['U' 'U' 'U' 'U'];
% vec_valeur_CL_local = [0 0 0 1]; % m ou Pa
% liste_CL_data_U{1} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{2} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{3} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{4} = struct('DL_bloque',[vec_type_CL_local(4) 'z'],'val_DL_bloque',vec_valeur_CL_local(4),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_max z_max],'tolerance',tolerance_position)); % plan : vecteur_normal + point

% %%%%%%%%%%%%%%%%%%%%%%%%
% % 1) CL : U = 0 sur 3 faces
% %%%%%%%%%%%%%%%%%%%%%%%%
% vec_type_CL_local = ['U' 'U' 'U'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{1} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{2} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{3} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% %liste_CL_data_U{4} = struct('DL_bloque',[vec_type_CL_local(4) 'y'],'val_DL_bloque',vec_valeur_CL_local(4),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_max z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point

% %%%%%%%%%%%%%%%%%%%%%%%%
% % 2) CL : U = 0 sur toutes les faces
% %%%%%%%%%%%%%%%%%%%%%%%%
% % CL face "bas : n = -z"
% vec_type_CL_local = ['U' 'U' 'U'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{1} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{2} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{3} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% 
% % CL face "haut : n = z"
% vec_type_CL_local = ['U' 'U' 'U'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{4} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_max],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{5} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_max],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{6} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_max],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% 
% % CL face "gauche : n = -x"
% vec_type_CL_local = ['U' 'U' 'U'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{7} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{8} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{9} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% 
% % CL face "droite : n = x"
% vec_type_CL_local = ['U' 'U' 'U'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{10} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_max y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{11} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_max y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{12} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_max y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% 
% % CL face "arriere : n =-y"
% vec_type_CL_local = ['U' 'U' 'U'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{13} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{14} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{15} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% 
% % CL face "avant : n = y"
% vec_type_CL_local = ['U' 'U' 'U'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{16} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_max z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{17} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_max z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{18} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_max z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point

% %%%%%%%%%%%%%%%%%%%%%%%%
% % 3) CL mixtes : U = 0 sur 3 faces et F = 0 ailleurs
% %%%%%%%%%%%%%%%%%%%%%%%%
% % CL face "bas : n = -z"
% vec_type_CL_local = ['F' 'F' 'U'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{1} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{2} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{3} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% 
% % CL face "haut : n = z"
% %vec_type_CL_local = ['F' 'F' 'F'];
% %vec_valeur_CL_local = [0 0 0]; % m ou Pa
% vec_type_CL_local = ['U' 'F' 'F'];
% vec_valeur_CL_local = [(0.001*(z_max-z_min)) 0 0]; % m ou Pa
% liste_CL_data_U{4} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_max],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{5} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_max],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{6} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_max],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% 
% % CL face "gauche : n = -x"
% vec_type_CL_local = ['U' 'F' 'F'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{7} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{8} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{9} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% 
% % CL face "droite : n = x"
% vec_type_CL_local = ['F' 'F' 'F'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{10} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_max y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{11} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_max y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{12} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_max y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% 
% % CL face "arriere : n =-y"
% vec_type_CL_local = ['F' 'U' 'F'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{13} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{14} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{15} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% 
% % CL face "avant : n = y"
% vec_type_CL_local = ['F' 'F' 'F'];
% vec_valeur_CL_local = [0 0 0]; % m ou Pa
% liste_CL_data_U{16} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',vec_valeur_CL_local(1),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_max z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{17} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',vec_valeur_CL_local(2),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_max z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% liste_CL_data_U{18} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',vec_valeur_CL_local(3),'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_max z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
% 

%%%%%%%%%%%%%%%%%%%%%%%%
% 4) CL : U = 'Um_filtre' sur toutes les faces
%%%%%%%%%%%%%%%%%%%%%%%%
% CL face "bas : n = -z"
vec_type_CL_local = ['U' 'U' 'U'];
liste_valeur_CL_local = {'Um_filtre','Um_filtre','Um_filtre'}; % m ou Pa
liste_CL_data_U{1} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',liste_valeur_CL_local{1},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{2} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',liste_valeur_CL_local{2},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{3} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',liste_valeur_CL_local{3},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point

% CL face "haut : n = z"
vec_type_CL_local = ['U' 'U' 'U'];
liste_valeur_CL_local = {'Um_filtre','Um_filtre','Um_filtre'}; % m ou Pa
liste_CL_data_U{4} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',liste_valeur_CL_local{1},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_max],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{5} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',liste_valeur_CL_local{2},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_max],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{6} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',liste_valeur_CL_local{3},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 0 1],'point',[x_min y_min z_max],'tolerance',tolerance_position)); % plan : vecteur_normal + point

% CL face "gauche : n = -x"
vec_type_CL_local = ['U' 'U' 'U'];
liste_valeur_CL_local = {'Um_filtre','Um_filtre','Um_filtre'}; % m ou Pa
liste_CL_data_U{7} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',liste_valeur_CL_local{1},'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{8} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',liste_valeur_CL_local{2},'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{9} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',liste_valeur_CL_local{3},'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point

% CL face "droite : n = x"
vec_type_CL_local = ['U' 'U' 'U'];
liste_valeur_CL_local = {'Um_filtre','Um_filtre','Um_filtre'}; % m ou Pa
liste_CL_data_U{10} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',liste_valeur_CL_local{1},'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_max y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{11} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',liste_valeur_CL_local{2},'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_max y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{12} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',liste_valeur_CL_local{3},'type_entite','plan','caracteristique_entite',struct('vec_n',[1 0 0],'point',[x_max y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point

% CL face "arriere : n =-y"
vec_type_CL_local = ['U' 'U' 'U'];
liste_valeur_CL_local = {'Um_filtre','Um_filtre','Um_filtre'}; % m ou Pa
liste_CL_data_U{13} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',liste_valeur_CL_local{1},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{14} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',liste_valeur_CL_local{2},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{15} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',liste_valeur_CL_local{3},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_min z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point

% CL face "avant : n = y"
vec_type_CL_local = ['U' 'U' 'U'];
liste_valeur_CL_local = {'Um_filtre','Um_filtre','Um_filtre'}; % m ou Pa
liste_CL_data_U{16} = struct('DL_bloque',[vec_type_CL_local(1) 'x'],'val_DL_bloque',liste_valeur_CL_local{1},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_max z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{17} = struct('DL_bloque',[vec_type_CL_local(2) 'y'],'val_DL_bloque',liste_valeur_CL_local{2},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_max z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point
liste_CL_data_U{18} = struct('DL_bloque',[vec_type_CL_local(3) 'z'],'val_DL_bloque',liste_valeur_CL_local{3},'type_entite','plan','caracteristique_entite',struct('vec_n',[0 1 0],'point',[x_min y_max z_min],'tolerance',tolerance_position)); % plan : vecteur_normal + point

