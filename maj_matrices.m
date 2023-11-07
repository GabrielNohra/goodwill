function [Ks,Fs,U_impose,vec_n_DDL_conserves,vec_n_DDL_supprimes] = maj_matrices(K,F,liste_DL_bloque)

%%%%%%%% determination des numeros des DDL supprimes et des DDL conserves
vec_n_DDL_conserves = 1:size(K,1);
vec_n_DDL_supprimes = [];
for k = 1:length(liste_DL_bloque)
 if ( ~isempty(liste_DL_bloque{k}.vec_n_DL_bloque) && ((strcmp(liste_DL_bloque{k}.type,'U') == 1) || (strcmp(liste_DL_bloque{k}.type,'W') == 1)) )
  vec_n_DDL_supprimes = [vec_n_DDL_supprimes liste_DL_bloque{k}.vec_n_DL_bloque];  
 end
end
vec_test = ones(1,size(K,1));
vec_test(vec_n_DDL_supprimes) = 0;
[ii] = find ( vec_test == 1 );
vec_n_DDL_conserves = vec_n_DDL_conserves(ii);
clear ii;
clear vec_test;

%%%%%%%%%
nb_DDL = length(vec_n_DDL_conserves)+length(vec_n_DDL_supprimes);
%Fs = zeros(nb_DDL,1);
Fs = F;
U_impose = zeros(length(vec_n_DDL_supprimes),1);
n_prec = 1;
for k = 1:length(liste_DL_bloque)
% A MODIFIER : PROGRAMMER L'INTEGRATION SUR LES ELEMENTS DE SURFACE
% CALCUL DE "Fs" incorrect si les efforts appliqués ne sont pas nuls ...
 if ( ~isempty(liste_DL_bloque{k}.vec_n_DL_bloque) && (strcmp(liste_DL_bloque{k}.type,'F') == 1) )
  Fs(liste_DL_bloque{k}.vec_n_DL_bloque) = liste_DL_bloque{k}.vec_val_DL_bloque;
 elseif ( ~isempty(liste_DL_bloque{k}.vec_n_DL_bloque) && ((strcmp(liste_DL_bloque{k}.type,'U') == 1) || (strcmp(liste_DL_bloque{k}.type,'W') == 1)) )
  Fs = Fs-K(:,liste_DL_bloque{k}.vec_n_DL_bloque)*liste_DL_bloque{k}.vec_val_DL_bloque.';
  U_impose(n_prec:n_prec+length(liste_DL_bloque{k}.vec_val_DL_bloque)-1) = liste_DL_bloque{k}.vec_val_DL_bloque;
  n_prec = n_prec+length(liste_DL_bloque{k}.vec_val_DL_bloque);
 end
end

%%%%%%%% suppression des lignes et des colonnes associees aux DDL_supprimes
if ( ~isempty(vec_n_DDL_conserves) )
% matrice
 Ks = K(:,vec_n_DDL_conserves);
 Ks = Ks(vec_n_DDL_conserves,:);
% second membre
 Fs = Fs(vec_n_DDL_conserves,1);
% normalisation du systeme : matrice Ks et vecteur Fs
%  vec_normalisation_Ks = ones(size(Ks,1),1);
%  vec_normalisation_Ks = zeros(size(Ks,1),1)+max(abs(vec_s));
 vec_normalisation_Ks = max(abs(Ks),[],2);
 [vec_i,vec_j,vec_s] = find(Ks);
 vec_normalisation_s = vec_normalisation_Ks(vec_i);
 vec_s = vec_s./vec_normalisation_s;
% creation de la matrice sparse Ks normee 
 Ks = sparse(vec_i,vec_j,vec_s,size(Ks,1),size(Ks,2));
 Fs = Fs./vec_normalisation_Ks;
else
 Ks = [];
 Fs = [];
end
