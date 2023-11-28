function plotting(A,kappa,noise,dir)
% 
% This function plots the values of the material properties
% for different scenarios

    [A(real(A)==0), A(imag(A)==0)] = deal(NaN); % change 0's for NaN's

    n = zeros(1,6); % Initialization of noise storage values
    n(1,1) = noise;

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

    for i=1:size(A,1)

        gcf = figure;
        hold on;

        plot(real(A(i,:)),'color',colorList{1},'linestyle','--');
        plot(imag(A(i,:)),'color',colorList{2},'linestyle','--');
        plot(1743*ones(1,size(A,2)),'-k');
        plot(174.3*ones(1,size(A,2)),'-k');

        strTitle = sprintf('Material property $\\mu$ ($\\kappa$ = %0.0e, noise = %0.4f \\%%)',kappa,n(i)*100);
        fileName = sprintf('results_(noise=%0.4f%%).png',n(i)*100);
        tl = title(strTitle,'interpreter,'latex');
        xl = xlabel('Number of iterations','interpreter','latex');
        yl = ylabel('$\mu$ [Pa]','interpreter','latex');
        lgd = legend({'Re $\left( \tilde{mu} \right)$','Im $\left( \tilde{mu} \right)$',...
            'Re $\left( \mu \right)$','Im $\left( \mu \right)$'};

        [tl.FontSize, xl.FontSize, yl.FontSize] = deal(12);
        lgd.FontSize = 11;
        grid;

        saveas(gcf,fileName);

        close gcf;

        if i ~= size(A,1)
            n(1,i+1) = noise + 0.00001*i;
        end

    end

    cd(dir{2});

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