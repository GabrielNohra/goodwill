sTime(1,i_param) = cputime - t_ini_identification(1,i_param);

valNoise(i_param,1:length(cell2mat(liste_proprietes_iterations))) = cell2mat(liste_proprietes_iterations);

n_iter_LDC_max = n_iter_LDC;
% n_iter_LDC = n_iter_LDC_max;

% for nn_param = 1:size(liste_proprietes_iterations,1)

%     n_param = struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier(nn_param);
%     nom_param = struct_param_comportement_a_identifier.liste_parametres_comportement{n_param};
%     gcf = figure;
%     hold on;
%     plot(real(liste_proprietes_iterations{n_iter_LDC}(nn_param,:)),'-r');
%     plot(imag(liste_proprietes_iterations{n_iter_LDC}(nn_param,:)),'-b');
%     grid;
%     xl = xlabel('Phase number node','interpreter','latex');
%     yl = ylabel('$\mu$ [Pa]','interpreter','latex');
%     lg = legend('Real','Imag','interpreter','latex');
%     [xl.FontSize, yl.FontSize] = deal(12);
%     lg.FontSize = 11;
%     saveas(gcf,sprintf('phaseNum_(noise=%0.2f).png',amplitude_bruit_Gaussien_U(i_param)*100));
%     close gcf;

% end

if j_param == 1
    gcf = figure;
else
    figure(gcf);
end

vec_param_identifie_moyen = zeros(size(liste_proprietes_iterations{n_iter_LDC},1),n_iter_LDC_max);

for n_iter_LDC = 1:n_iter_LDC_max
    for n_param = 1:size(liste_proprietes_iterations{n_iter_LDC},1)
        vec_param_identifie_moyen(n_param,n_iter_LDC) = mean(liste_proprietes_iterations{n_iter_LDC}(n_param,:));
    end
end

for n_param = 1:size(liste_proprietes_iterations{n_iter_LDC},1)
    % nom_param = struct_param_comportement_a_identifier.liste_parametres_comportement{struct_param_comportement_a_identifier.vec_numeros_parametres_a_identifier(n_param)};

    p = [p plot(real(vec_param_identifie_moyen(n_param,:)),'color',colorList{i_param},'linestyle','--')];
    hold on;
    p = [p plot(imag(vec_param_identifie_moyen(n_param,:)),'color',colorList{i_param},'linestyle','--')];

end

cd(path_dir{3});

figure(gcf);

hold on;
p = [p plot(1743*ones(size(vec_param_identifie_moyen(n_param,:))),'-k')];
p = [p plot(174.3*ones(size(vec_param_identifie_moyen(n_param,:))),'-k')];
grid;

tl = title(sprintf('Material property $\\mu$ (noise = %d \\%%)',amplitude_bruit_Gaussien_U(i_param)),'interpreter','latex');
xl = xlabel('Number of iterations','interpreter','latex');
yl = ylabel('$\mu$ [Pa]','interpreter','latex');
lgd = legend([p(1) p(3) p(5) p(7) p(9) p(11) p(12)], n_lg, {'Re $\left( \mu \right)$',...
     'Im $\left( \mu \right)$'});
[tl.FontSize, xl.FontSize, yl.FontSize] = deal(12);
lgd.FontSize = 11;
hold off;

saveas(gcf,sprintf('results_(noise=%0.2f%%).png',amplitude_bruit_Gaussien_U(i_param)*100));
close gcf;

tmp = figure;
hold on;

% for i=1:length(kappa)
%     plot(1:length(kappa), sTime(i,:), '-k');
% end

% tl = title('Code performance', 'interpreter', 'latex');
% xl = xlabel('Noise values', 'interpreter', 'latex');
% yl = ylabel('Simulation time [s]', 'interpreter', 'latex');
% lgd = legend(n_lg);
% [tl.FontSize, xl.FontSize, yl.FontSize] = deal(12);
% lgd.FontSize = 11;
% grid;
% saveas(tmp,'simTime (elastic).png');
% close tmp;

if j_param ~= length(amplitude_bruit_Gaussien_U)
    cd(path_dir{2});
end