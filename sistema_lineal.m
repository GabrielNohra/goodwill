%% Nomenclature

% red_mat: reduced matrix which contains all the coeficients of the linear
% system, grouped by stress elements.

%% Code

close all;

mat_coef_mat = zeros(2*length(liste_elem_sig_sub_zone)*27,2*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local));
mat_red = zeros(2*length(liste_elem_sig_sub_zone), 2*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local));

[vec_x_G, vec_y_G, vec_z_G, vec_tr_epsilon_U_G, vec_tr_sigma_G, vec_deviateur_epsilon_U_deviateur_epsilon_conjugue_U_G, vec_deviateur_sigma_deviateur_epsilon_conjugue_U_G] = deal(zeros(1,length(liste_elem_sig_sub_zone)*27));

vec_second_membre_mat = zeros(2*length(liste_elem_sig_sub_zone)*27,1);
vec_n_elem_pha_systeme = zeros(length(liste_elem_sig_sub_zone)*27,1);

n_G_local = 0;

for n_sig = 1:length(liste_elem_sig_sub_zone)
    
    elem_sig = liste_elem_sig_sub_zone{n_sig};
    elem_ref_sig = liste_elem_ref{elem_sig.n_elem_ref};
    vec_U_local = nan(3*length(elem_sig.vec_n_noeuds),1);
    vec_U_local(1:3:end) = Ux(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_U_local(2:3:end) = Uy(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_U_local(3:3:end) = Uz(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_conjugue_U_local = conj(vec_U_local);
    vec_W_local = nan(3*length(elem_sig.vec_n_noeuds),1);
    vec_W_local(1:3:end) = Wx(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_W_local(2:3:end) = Wy(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_W_local(3:3:end) = Wz(vec_correspondance_n_noeud_sig_global_n_noeud_sig_local(elem_sig.vec_n_noeuds)).';
    vec_poids_Gauss = liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.poids_Gauss;
    
    
    %
    %
    %    mat_d_f_Nf_sig = (elem_ref_sig.d_f_Nf(elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(:,1),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(:,2),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(:,3)));
    %
    %    n_pha = vec_n_pha_G_K(nn_pha_local);
    %    elem_pha = liste_elem_pha{n_pha};
    %    vec_n_noeuds_pha = elem_pha.vec_n_noeuds;
    %    elem_ref_pha = liste_elem_ref{elem_pha.n_elem_ref};
    %    vec_f_Nf_pha = zeros(size(elem_sig.mat_pos_pha_G_K,2),1,);
    %    vec_n_pha_G_K = unique(elem_sig.vec_n_pha_G_K);
    %    for nn_pha_local = 1:length(vec_n_pha_G_K)
    %        n_pha = vec_n_pha_G_K(nn_pha_local);
    %        elem_pha = liste_elem_pha{n_pha};
    %        vec_n_noeuds_pha = elem_pha.vec_n_noeuds;
    %        elem_ref_pha = liste_elem_ref{elem_pha.n_elem_ref};
    %        nn_noeud_sig_pha_courante = find ( elem_sig.vec_n_pha_G_K == vec_n_pha_G_K(nn_pha_local) );
    %        vec_f_Nf_pha(nn_noeud_sig_pha_courante,1,:) = elem_ref_pha.f_Nf(elem_sig.mat_pos_pha_G_K(1,nn_noeud_sig_pha_courante),elem_sig.mat_pos_pha_G_K(2,nn_noeud_sig_pha_courante),elem_sig.mat_pos_pha_G_K(3,nn_noeud_sig_pha_courante));
    %    end
    %
    
    for n_G = 1:size(elem_sig.mat_pos_pha_G_K,2)
        
        n_pha = elem_sig.vec_n_pha_G_K(n_G);
        vec_pos_G_ref = elem_sig.mat_pos_pha_G_K(:,n_G);
        elem_pha = liste_elem_pha{n_pha};
        vec_n_noeuds_pha = elem_pha.vec_n_noeuds;
        elem_ref_pha = liste_elem_ref{elem_pha.n_elem_ref};
        vec_f_Nf_pha = squeeze(elem_ref_pha.f_Nf(vec_pos_G_ref(1),vec_pos_G_ref(2),vec_pos_G_ref(3)))';
        % % calcul de la deformation au point de Gauss de l'element de contrainte
        mat_d_f_Nf_sig = squeeze(elem_ref_sig.d_f_Nf(elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,1),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,2),elem_ref_sig.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,3)))';
        Jaco = mat_d_f_Nf_sig*mat_pos_maillage_sig(elem_sig.vec_n_noeuds,:); % matrice 3x3. Produit du gradient fonction de forme associ� et coordonn�es des noeuds
        J = det(Jaco); % d�terminant
        % % matrice du gradient des fonctions de formes dans la configuration reelle
        % % permettant de calculer le tenseur de deformation en notation de "Voigt modifiee" : (exx,eyy,ezz,2*exy,2*exz,2*eyz)
        [Be] = calcul_Be(mat_d_f_Nf_sig,Jaco,nb_DDL_par_noeud);
        vec_epsilon_U_Voigt = Be*vec_U_local;
        vec_epsilon_conjugue_U_Voigt = Be*vec_conjugue_U_local;
        vec_epsilon_W_Voigt = Be*vec_W_local;
        % % calcul des fonctions de forme sur les phases au point de Gauss de l'element de contrainte
        lambda_G = champ_proprietes.lambda(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
        mu_G = champ_proprietes.mu(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
        data_LDC = struct('lambda',lambda_G,'mu',mu_G);
        A_visco = LDC(data_LDC);
        lambda_elastique_G = champ_proprietes_elastiques.lambda(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
        mu_elastique_G = champ_proprietes_elastiques.mu(elem_pha.vec_n_noeuds)*vec_f_Nf_pha';
        data_LDC_elastique = struct('lambda',lambda_elastique_G,'mu',mu_elastique_G);
        A_elastique = LDC(data_LDC_elastique);
        % % calcul de sigma : sigma = C:epsilon[u] + P:epsilon[w]
        vec_sigma_Voigt = A_visco*vec_epsilon_U_Voigt+A_elastique*vec_epsilon_W_Voigt;
        % % calcul des quantites necessaires a la mise a jour des proprietes des phases
        deviateur_epsilon_U = calcul_deviateur(vec_epsilon_U_Voigt./[1 1 1 2 2 2]');
        deviateur_epsilon_conjugue_U = calcul_deviateur(vec_epsilon_conjugue_U_Voigt./[1 1 1 2 2 2]');
        deviateur_sigma = calcul_deviateur(vec_sigma_Voigt);
        tr_epsilon_U = vec_epsilon_U_Voigt(1)+vec_epsilon_U_Voigt(2)+vec_epsilon_U_Voigt(3);
        tr_sigma = vec_sigma_Voigt(1)+vec_sigma_Voigt(2)+vec_sigma_Voigt(3);
        deviateur_epsilon_U_deviateur_epsilon_conjugue_U = sum(sum(deviateur_epsilon_U.*deviateur_epsilon_conjugue_U));
        deviateur_sigma_deviateur_epsilon_conjugue_U = sum(sum(deviateur_sigma.*deviateur_epsilon_conjugue_U));
        
        % remplissage des vecteurs pour tester
        
%         n_G_local = n_G_local+1;
%         vec_x_G(n_G_local) = (min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,1))+(max(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,1))-min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,1)))*(liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,1)+1)/2);
%         vec_y_G(n_G_local) = (min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,2))+(max(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,2))-min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,2)))*(liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,2)+1)/2);
%         vec_z_G(n_G_local) = (min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,3))+(max(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,3))-min(mat_pos_maillage_sig(elem_sig.vec_n_noeuds,3)))*(liste_elem_ref{elem_sig.n_elem_ref}.liste_parametres_integration{n_integration_K}.pos_Gauss(n_G,3)+1)/2);
%         vec_tr_epsilon_U_G(n_G_local) = tr_epsilon_U;
%         vec_tr_sigma_G(n_G_local) = tr_sigma;
%         vec_deviateur_epsilon_U_deviateur_epsilon_conjugue_U_G(n_G_local) = deviateur_epsilon_U_deviateur_epsilon_conjugue_U;
%         vec_deviateur_sigma_deviateur_epsilon_conjugue_U_G(n_G_local) = deviateur_sigma_deviateur_epsilon_conjugue_U; % remplissage du systeme lineaire :
        % remplissage du systeme lineaire :
        
        % % methode (i) ecriture des equations en (lambda,mu)
        
        % % equation sur lambda
        
        mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*3*tr_epsilon_U*vec_f_Nf_pha; % lambda
        mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*2*tr_epsilon_U*vec_f_Nf_pha; % mu
        
        vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = vec_poids_Gauss(n_G)*J*tr_sigma;
        
        vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_pha;
        
        % % equation sur mu
        
        mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2,max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local)+vec_correspondance_n_noeud_pha_global_n_noeud_pha_local(vec_n_noeuds_pha)) = vec_poids_Gauss(n_G)*J*2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U*vec_f_Nf_pha; % mu
        
        vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = vec_poids_Gauss(n_G)*J*deviateur_sigma_deviateur_epsilon_conjugue_U;
        
        vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_pha;
        
        %  % methode (ii) ecriture des equations en (B,G)
        %  % equation sur B
        %     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1,vec_n_noeuds_pha) = vec_poids_Gauss(n_G)*J*3*tr_epsilon_U*vec_f_Nf_pha; % B
        %     vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = vec_poids_Gauss(n_G)*J*tr_sigma;
        %     vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+1) = n_pha;
        %  % equation sur G
        %     mat_coef_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2,size(mat_pos_pha,1)+vec_n_noeuds_pha) = vec_poids_Gauss(n_G)*J*2*deviateur_epsilon_U_deviateur_epsilon_conjugue_U*vec_f_Nf_pha; % G
        %     vec_second_membre_mat(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = vec_poids_Gauss(n_G)*J*deviateur_sigma_deviateur_epsilon_conjugue_U;
        %     vec_n_elem_pha_systeme(2*elem_sig_ref.liste_parametres_integration{n_integration_K}.nb_Gauss*(n_sig-1)+2*(n_G-1)+2) = n_pha;
    end
    
    % by stress elements
    
 	 mat_red(2*n_sig-1,:) = sum(mat_coef_mat( 2*(n_sig-1)*27+1:2:2*n_sig*27-1,: ));
 	 mat_red(2*n_sig,:) = sum(mat_coef_mat( 2*(n_sig-1)*27+2:2:2*n_sig*27,: ));
%      
	 vsm_red(2*n_sig-1,:) = sum(vec_second_membre_mat( ((n_sig-1)*27 + n_sig):2:(n_sig*(2*27-1)+n_sig-1),: ));
	 vsm_red(2*n_sig,:) = sum(vec_second_membre_mat( (n_sig-1)*27 + n_sig+1:2:2*27,: ));
    
    
end

% by phase elements

M_p = zeros(2*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local));
v_p = zeros(2*max(vec_correspondance_n_noeud_pha_global_n_noeud_pha_local),1);

aux = unique(vec_n_elem_pha_systeme);

for i=1:size(mat_coef_mat,2)/2
    A_aux = mat_coef_mat(vec_n_elem_pha_systeme(:) == aux(i),:);
    v_aux = vec_second_membre_mat(vec_n_elem_pha_systeme(:) == aux(i));
    M_p(2*(i-1)+1,:) = sum(A_aux(1:2:end,:));
    M_p(2*i,:) = sum(A_aux(2:2:end,:));
    v_p(2*(i-1)+1) = sum(v_aux(1:2:end));
    v_p(2*i) = sum(v_aux(2:2:end));
end


M_ps = (M_p.'*M_p);
v_ps = (M_p.'*v_p);
vmax = max(abs(M_ps),[],2);
%vec_max_square = ones(size(mat_coef_mat_square,1),1);
M_ps = M_ps./(vmax*ones(1,size(M_ps,2)));
v_ps = v_ps./vmax;
vss = M_ps\v_ps;

vss = sign_change(vss,struct_param_comportement_a_identifier.vec_param_initialisation,0.6);




%% --- Mise a jour des proprietes
    
    liste_proprietes_iterations{n_iter_LDC+1} = nan(1,60);
    
    for i=1:size(vss,1)/2
        liste_proprietes_iterations{n_iter_LDC+1}(1,aux(i)) = vss(2*(i-1)+1);
        liste_proprietes_iterations{n_iter_LDC+1}(2,aux(i)) = vss(2*i);
    end
    
     %% --- Test de la convergence

     dev = [liste_proprietes_iterations{n_iter_LDC+1}(1,:) liste_proprietes_iterations{n_iter_LDC+1}(2,:)] - [liste_proprietes_iterations{n_iter_LDC}(1,:) liste_proprietes_iterations{n_iter_LDC+1}(2,:)];
     
     if ( norm(dev) < tolerance_LDC*norm( [liste_proprietes_iterations{n_iter_LDC}(1,:) liste_proprietes_iterations{n_iter_LDC}(2,:)] ) )
         test_convergence_LDC = true;
     end


%%

% vec_max = max(abs(mat_coef_mat),[],2);
% %vec_max = ones(size(mat_coef_mat,1),1);
% mat_coef_mat = mat_coef_mat./(vec_max*ones(1,size(mat_coef_mat,2)));
% vec_second_membre_mat = vec_second_membre_mat./vec_max;
% %vec_sol = (mat_coef_mat.'*mat_coef_mat)\(mat_coef_mat.'*vec_second_membre_mat.');
% vec_sol = mat_coef_mat\vec_second_membre_mat;
% figure;hold on;plot(real(vec_second_membre_mat),'r');plot(real(mat_coef_mat*vec_sol),'b');grid;

figure;imagesc(abs(mat_red));colorbar;
figure;imagesc(abs(mat_coef_mat));colorbar;

options = optimoptions('lsqlin','Algorithm','trust-region-reflective','Display','iter','MaxIterations',1000);

% first approach:

% system of linear equations (magnitude of complex arrays)

M_mat_red = abs(mat_red);
M_vsm_red = abs(vsm_red);
lb_mag = zeros(size(mat_red,2),1);
ub_mag = Inf*ones(size(mat_red,2),1);
x_sol_M = lsqlin(M_mat_red,M_vsm_red,[],[],[],[],lb_mag,ub_mag);

% system of linear equations (phase of complex arrays)

P_mat_red = angle(mat_red);
P_vsm_red = angle(vsm_red);
lb_pha = zeros(size(mat_red,2),1);
ub_pha = pi()*ones(size(mat_red,2),1)/2;
x_sol_P = lsqlin(P_mat_red,P_vsm_red,[],[],[],[],lb_pha,ub_pha);

[a,b] = pol2cart(x_sol_P,x_sol_M);
x_sol_1 = a + 1i*b;

% second approach:

lb = zeros(size(mat_red,2),1);
ub = Inf*ones(size(mat_red,2),1);

% system of linear equations (real part of arrays)

R_mat_red = abs(mat_red);
R_vsm_red = abs(vsm_red);
x_sol_R = lsqlin(R_mat_red,R_vsm_red,[],[],[],[],lb,ub);

% system of linear equations (imaginary part of arrays)

I_mat_red = angle(mat_red);
I_vsm_red = angle(vsm_red);
x_sol_I = lsqlin(I_mat_red,I_vsm_red,[],[],[],[],lb,ub);

% x_sol = x_solR + 1i* x_solI

x_sol_2 = x_sol_R + 1i*x_sol_I;

% second approach (modified):

lb = zeros(size(M_p,2),1);
ub = Inf*ones(size(M_p,2),1);

% system of linear equations (real part of arrays)

R_mat_red = real(M_p);
R_vsm_red = real(v_p);
x_sol_R = lsqlin(R_mat_red,R_vsm_red,[],[],[],[],lb,[]);

% system of linear equations (imaginary part of arrays)

I_mat_red = imag(M_p);
I_vsm_red = imag(v_p);
x_sol_I = lsqlin(I_mat_red,I_vsm_red,[],[],[],[],lb,[]);

% x_sol = x_solR + 1i* x_solI

x_sol_2 = x_sol_R + 1i*x_sol_I;


% --------

% third approach

mat_coef_mat_square = (mat_coef_mat.'*mat_coef_mat);
vec_second_membre_mat_square = (mat_coef_mat.'*vec_second_membre_mat);
vec_max_square = max(abs(mat_coef_mat_square),[],2);
%vec_max_square = ones(size(mat_coef_mat_square,1),1);
mat_coef_mat_square = mat_coef_mat_square./(vec_max_square*ones(1,size(mat_coef_mat_square,2)));
vec_second_membre_mat_square = vec_second_membre_mat_square./vec_max_square;
vec_sol_square = mat_coef_mat_square\vec_second_membre_mat_square;


if nnz(sum(real(vec_sol_square)<0))~=0
    vec_sol_square(real(vec_sol_square)<0) = vec_sol_square(real(vec_sol_square)<0) + 2*real(vec_sol_square(real(vec_sol_square)<0));
end

if nnz(sum(imag(vec_sol_square)<0))~=0
    vec_sol_square(imag(vec_sol_square)<0) = conj(vec_sol_square(imag(vec_sol_square)<0));
end

% --------

mat2 = (mat_red.'*mat_red);
vsm2 = (mat_red.'*vsm_red);
mat2_max = max(abs(mat2),[],2);
%vec_max_square = ones(size(mat_coef_mat_square,1),1);
mat2 = mat2./(mat2_max*ones(1,size(mat2,2)));
vsm2 = vsm2./mat2_max;
vec2 = mat2\vsm2;


if nnz(sum(real(vec2)<0))~=0
    vec2(real(vec2)<0) = vec2(real(vec2)<0) - 2*real(vec2(real(vec2)<0));
end

if nnz(sum(imag(vec2)<0))~=0
    vec2(imag(vec2)<0) = conj(vec2(imag(vec2)<0));
end

% 
% figure;subplot(2,1,1);hold on;plot(real(vec_second_membre_mat_square),'r');plot(real(mat_coef_mat_square*vec_sol_square),'b');grid;hold off;
% subplot(2,1,2);hold on;plot(real(vsm2),'r');plot(real(mat2*vec2),'b');grid;hold off;
% 
% figure;subplot(2,1,1);hold on;plot(real(vec_second_membre_mat_square-mat_coef_mat_square*vec_sol_square),'r');plot(imag(vec_second_membre_mat_square-mat_coef_mat_square*vec_sol_square),'b');grid;
% subplot(2,1,2);hold on;plot(real(vsm2-mat2*vec2),'r');plot(imag(vsm2-mat2*vec2),'b');grid;

figure;subplot(2,1,1);hold on;plot(real(vec_sol_square),'r');plot(imag(vec_sol_square),'b');grid;
subplot(2,1,2);hold on;plot(real(vec2),'r');plot(imag(vec2),'b');grid;



