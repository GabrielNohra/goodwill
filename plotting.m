function plotting(A,kappa,noise,dir)
% 
% This function plots the values of the material properties
% for different scenarios

    A(A==0) = NaN; % change 0's for NaN's

    color_1 = '[0 0.03 1]';
    color_2 = '[0.2 1 1]';
    color_3 = '[0 0.57 0.85]';
    color_4 = '[1 0 0.5]';
    color_5 = '[0.90 0 0.57]';
    color_6 = '[0.03 0.46 0.02]';
    color_7 = '[0.85 0 0.48]';
    color_8 = '[0 0.49 0.49]';
    color_9 = '[0.67 0 0]';

    colorList = {color_1,color_2,color_3,color_4,...
                    color_5,color_6,color_7,color_8,color_9};

    p = [];

    gcf = figure;
    hold on;

    n = zeros(1,6);
    n(1,1) = noise;

    for i=1:size(A,1)
        p = [p plot(real(A(i,:)),'color',colorList{i},'linestyle','--') 
            plot(imag(A(i,:)),'color',colorList{i},'linestyle','--')];
        if i ~= size(A,1)
            n(1,i+1) = noise + 0.00001*i;
        end
    end

    lgd = @(x) sprintf('noise = %0.4f \\%%',n*100);

    n_lg = {lgd(n(1)), lgd(n(2)), lgd(n(3)), lgd(n(4)), lgd(n(5)), lgd(n(6))};

    p = [p plot(1743*ones(size(A,2),'-k')) plot(174.3*ones(size(A,2),'-k'))];

    tl = title(sprintf('Material property $\mu$ (kappa = %0.0e)',kappa),'interpreter','latex');
    xl = xlabel('Number of iterations','interpreter','latex');
    yl = ylabel('$\mu$ [Pa]','interpreter','latex');
    lgd = legend([p(1) p(3) p(5) p(7) p(9) p(11)], n_lg, {'Re $\left( \mu \right)$',...
        'Im $\left( \mu \right)$'});
    [tl.FontSize, xl.FontSize, yl.FontSize] = deal(12);
    lgd.FontSize = 11;
    grid;

    hold off;

    cd(dir{3});
    saveas(gcf,'results.png');

    close gcf;

end













% cd Results_27102023/algebraic_method/all;

% idx = find(~isnan(liste_proprietes_iterations{2}(1,:)));

% [a,b] = deal( nan(length(idx),length(liste_proprietes_iterations)) );

% close all;

% for i=1:length(idx)
    
%     for j=1:length(liste_proprietes_iterations)-1
%         %a(idx(i),j) = liste_proprietes_iterations{j+1}(1,idx(i));
%         b(idx(i),j) = liste_proprietes_iterations{j+1}(2,idx(i));
%     end
    
% %     gcf = figure;
% %     plot(real(a(idx(i),:)),'b*');
% %     hold on;
% %     plot(imag(a(idx(i),:)),'r*');
% %     legend('Real part','Imaginary part','interpreter','latex');
% %     title(sprintf('Material property $\\lambda$ at node %d',idx(i)),...
% %         'interpreter','latex');
% %     saveas(gcf,sprintf('lambda_%d.png',idx(i)));
% %     close gcf;
    
%     gcf = figure;
%     plot(real(b(idx(i),:)),'b*');
%     hold on;
%     plot(imag(b(idx(i),:)),'r*');
%     legend('Real part','Imaginary part','interpreter','latex');
%     title(sprintf('Material property $\\mu$ at node %d',idx(i)),...
%         'interpreter','latex');
%     saveas(gcf,sprintf('mu_%d.png',idx(i)));
%     close gcf;
    
% end