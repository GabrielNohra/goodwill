n_kappa=1;

U_global = lista4(:,n_kappa);

% deplacements mesures

                vec_Ux_mes_sub_zone = mat_U_mes(1,vec_n_noeuds_mes_sub_zone);
                vec_Uy_mes_sub_zone = mat_U_mes(2,vec_n_noeuds_mes_sub_zone);
                vec_Uz_mes_sub_zone = mat_U_mes(3,vec_n_noeuds_mes_sub_zone);
                vec_i_mes_sub_zone = vec_i_mes(vec_n_noeuds_mes_sub_zone);
                vec_j_mes_sub_zone = vec_j_mes(vec_n_noeuds_mes_sub_zone);
                vec_k_mes_sub_zone = vec_k_mes(vec_n_noeuds_mes_sub_zone);
                vec_np_mes_sub_zone_global = vec_i_mes_sub_zone+ni_mes*((vec_j_mes_sub_zone-1)+nj_mes*(vec_k_mes_sub_zone-1));
                vec_i_mes_sub_zone = vec_i_mes_sub_zone-min(vec_i_mes_sub_zone)+1;
                vec_j_mes_sub_zone = vec_j_mes_sub_zone-min(vec_j_mes_sub_zone)+1;
                vec_k_mes_sub_zone = vec_k_mes_sub_zone-min(vec_k_mes_sub_zone)+1;
                ni_mes_sub_zone = max(vec_i_mes_sub_zone);
                nj_mes_sub_zone = max(vec_j_mes_sub_zone);
                nk_mes_sub_zone = max(vec_k_mes_sub_zone);
                vec_np_mes_sub_zone_local = vec_i_mes_sub_zone+ni_mes_sub_zone*((vec_j_mes_sub_zone-1)+nj_mes_sub_zone*(vec_k_mes_sub_zone-1));
                mat_Ux_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
                mat_Ux_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
                mat_Uy_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
                mat_Uy_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
                mat_Uz_real_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
                mat_Uz_imag_mes_sub_zone_affichage = nan(ni_mes_sub_zone,nj_mes_sub_zone,nk_mes_sub_zone);
                mat_Ux_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Ux_mes_sub_zone);
                mat_Ux_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Ux_mes_sub_zone);
                mat_Uy_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Uy_mes_sub_zone);
                mat_Uy_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Uy_mes_sub_zone);
                mat_Uz_real_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = real(vec_Uz_mes_sub_zone);
                mat_Uz_imag_mes_sub_zone_affichage(vec_np_mes_sub_zone_local) = imag(vec_Uz_mes_sub_zone);
                vec_x_grille_mes_sub_zone = struct_grille_mes.x_min+struct_grille_mes.dx*((min(vec_i_mes(vec_np_mes_sub_zone_global)):max(vec_i_mes(vec_np_mes_sub_zone_global)))-1);
                vec_y_grille_mes_sub_zone = struct_grille_mes.y_min+struct_grille_mes.dy*((min(vec_j_mes(vec_np_mes_sub_zone_global)):max(vec_j_mes(vec_np_mes_sub_zone_global)))-1);
                vec_z_grille_mes_sub_zone = struct_grille_mes.z_min+struct_grille_mes.dz*((min(vec_k_mes(vec_np_mes_sub_zone_global)):max(vec_k_mes(vec_np_mes_sub_zone_global)))-1);
                [mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone] = meshgrid(vec_y_grille_mes_sub_zone,vec_x_grille_mes_sub_zone,vec_z_grille_mes_sub_zone);
                coupe_x_mes = [min(vec_x_grille_mes_sub_zone) mean(vec_x_grille_mes_sub_zone) max(vec_x_grille_mes_sub_zone)];
                coupe_y_mes = max(vec_y_grille_mes_sub_zone);
                coupe_z_mes = [min(vec_z_grille_mes_sub_zone) mean(vec_z_grille_mes_sub_zone)];
                
                hh1 = figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Ux_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Ux mes), (m)');
                hh2 = figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Ux_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Ux mes), (m)');
                hh3 = figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uy_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Uy mes), (m)');
                hh4 = figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uy_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Uy mes), (m)');
                hh5 = figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uz_real_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('real(Uz mes), (m)');
                hh6 = figure;slice(mat_y_grille_mes_sub_zone,mat_x_grille_mes_sub_zone,mat_z_grille_mes_sub_zone,mat_Uz_imag_mes_sub_zone_affichage,coupe_y_mes,coupe_x_mes,coupe_z_mes);colorbar;xlabel('y');ylabel('x');zlabel('z');title('imag(Uz mes), (m)');

                saveas(hh1, 'results001.png');
                saveas(hh2, 'results002.png');
                saveas(hh3, 'results003.png');
                saveas(hh4, 'results004.png');
                saveas(hh5, 'results005.png');
                saveas(hh6, 'results006.png');

                Ux = U_global(nb_DDL_K+1:3:end);
                Uy = U_global(nb_DDL_K+2:3:end);
                Uz = U_global(nb_DDL_K+3:3:end);
                Wx = U_global(1:3:nb_DDL_K);
                Wy = U_global(2:3:nb_DDL_K);
                Wz = U_global(3:3:nb_DDL_K);

                






        % champs calcules
                vec_Ux_sub_zone = Ux;
                vec_Uy_sub_zone = Uy;
                vec_Uz_sub_zone = Uz;
                vec_Wx_sub_zone = Wx;
                vec_Wy_sub_zone = Wy;
                vec_Wz_sub_zone = Wz;
                mat_pos_maillage_sig_sub_zone = mat_pos_maillage_sig(vec_n_noeuds_sig_sub_zone,:);
                vec_x_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,1);
                vec_y_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,2);
                vec_z_sig_sub_zone = mat_pos_maillage_sig_sub_zone(:,3);
                F_Ux_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Ux_sub_zone),'linear','none');
                F_Ux_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Ux_sub_zone),'linear','none');
                F_Uy_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Uy_sub_zone),'linear','none');
                F_Uy_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Uy_sub_zone),'linear','none');
                F_Uz_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Uz_sub_zone),'linear','none');
                F_Uz_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Uz_sub_zone),'linear','none');
                F_Wx_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Wx_sub_zone),'linear','none');
                F_Wx_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Wx_sub_zone),'linear','none');
                F_Wy_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Wy_sub_zone),'linear','none');
                F_Wy_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Wy_sub_zone),'linear','none');
                F_Wz_real_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,real(vec_Wz_sub_zone),'linear','none');
                F_Wz_imag_sig = scatteredInterpolant(vec_x_sig_sub_zone,vec_y_sig_sub_zone,vec_z_sig_sub_zone,imag(vec_Wz_sub_zone),'linear','none');
                vec_grille_x_sig_sub_zone = unique(sort(vec_x_sig_sub_zone));
                vec_grille_y_sig_sub_zone = unique(sort(vec_y_sig_sub_zone));
                vec_grille_z_sig_sub_zone = unique(sort(vec_z_sig_sub_zone));
                [mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone] = meshgrid(vec_grille_x_sig_sub_zone,vec_grille_y_sig_sub_zone,vec_grille_z_sig_sub_zone);
                mat_Ux_real_sig_sub_zone_affichage = F_Ux_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                mat_Ux_imag_sig_sub_zone_affichage = F_Ux_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                mat_Uy_real_sig_sub_zone_affichage = F_Uy_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                mat_Uy_imag_sig_sub_zone_affichage = F_Uy_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                mat_Uz_real_sig_sub_zone_affichage = F_Uz_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                mat_Uz_imag_sig_sub_zone_affichage = F_Uz_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                mat_Wx_real_sig_sub_zone_affichage = F_Wx_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                mat_Wx_imag_sig_sub_zone_affichage = F_Wx_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                mat_Wy_real_sig_sub_zone_affichage = F_Wy_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                mat_Wy_imag_sig_sub_zone_affichage = F_Wy_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                mat_Wz_real_sig_sub_zone_affichage = F_Wz_real_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                mat_Wz_imag_sig_sub_zone_affichage = F_Wz_imag_sig(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone);
                coupe_x_sig = [min(vec_grille_x_sig_sub_zone) mean(vec_grille_x_sig_sub_zone) max(vec_grille_x_sig_sub_zone)];
                coupe_y_sig = max(vec_grille_y_sig_sub_zone);
                coupe_z_sig = [min(vec_grille_z_sig_sub_zone) mean(vec_grille_z_sig_sub_zone)];
                
                h1 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Ux), (m)');
                h2 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Ux_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Ux), (m)');
                h3 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uy), (m)');
                h4 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uy_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uy), (m)');
                h5 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Uz), (m)');
                h6 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Uz_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Uz), (m)');
                h7 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wx_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Wx), (m)');
                h8 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wx_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Wx), (m)');
                h9 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wy_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Wy), (m)');
                h10 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wy_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Wy), (m)');
                h11 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wz_real_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('real(Wz), (m)');
                h12 = figure;slice(mat_x_sig_sub_zone,mat_y_sig_sub_zone,mat_z_sig_sub_zone,mat_Wz_imag_sig_sub_zone_affichage,coupe_x_sig,coupe_y_sig,coupe_z_sig);colorbar;xlabel('x');ylabel('y');zlabel('z');title('imag(Wz), (m)');

                saveas(h1, 'results001_A.png');
                saveas(h2, 'results002_A.png');
                saveas(h3, 'results003_A.png');
                saveas(h4, 'results004_A.png');
                saveas(h5, 'results005_A.png');
                saveas(h6, 'results006_A.png');
                saveas(h7, 'results007_A.png');
                saveas(h8, 'results008_A.png');
                saveas(h9, 'results009_A.png');
                saveas(h10, 'results010_A.png');
                saveas(h11, 'results011_A.png');
                saveas(h12, 'results012_A.png');