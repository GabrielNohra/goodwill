function [vec_i,vec_j,vec_s] = raideur_assemblage (Ke_l,n_elem_sig,liste_struct_elem_sig,nb_DDL_par_noeud,vec_correspondance_n_noeud_sig_global_n_noeud_sig_local)

%vec_n_noeuds = liste_struct_elem_sig{n_elem_sig}.vec_n_noeuds; % numérotation noeud global
vec_n_noeuds = vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(liste_struct_elem_sig{n_elem_sig}.vec_n_noeuds); % numérotation noeud dans la subzone


vec_i = zeros(length(vec_n_noeuds)*length(vec_n_noeuds)*nb_DDL_par_noeud*nb_DDL_par_noeud,1);
vec_j = zeros(length(vec_n_noeuds)*length(vec_n_noeuds)*nb_DDL_par_noeud*nb_DDL_par_noeud,1);
vec_s = zeros(length(vec_n_noeuds)*length(vec_n_noeuds)*nb_DDL_par_noeud*nb_DDL_par_noeud,1);
n_prec = 1;
for n_l = 1:length(vec_n_noeuds) % local
    n_g = vec_n_noeuds(n_l); % global
    for ii = 1:nb_DDL_par_noeud
        val_i = nb_DDL_par_noeud*(n_g-1)+ii;
        for jj = 1:nb_DDL_par_noeud
            val_j = nb_DDL_par_noeud*(vec_n_noeuds-1)+jj;
            vec_i(n_prec:n_prec+length(vec_n_noeuds)-1) = zeros(1,length(vec_n_noeuds))+val_i;
            vec_j(n_prec:n_prec+length(vec_n_noeuds)-1) = val_j;
            vec_s(n_prec:n_prec+length(vec_n_noeuds)-1) = Ke_l(nb_DDL_par_noeud*(n_l-1)+ii,jj:nb_DDL_par_noeud:end);
            n_prec = n_prec+length(vec_n_noeuds);
        end
    end
end

end
