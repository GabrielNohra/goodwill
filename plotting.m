cd Results_27102023/algebraic_method/all;

idx = find(~isnan(liste_proprietes_iterations{2}(1,:)));

[a,b] = deal( nan(length(idx),length(liste_proprietes_iterations)) );

close all;

for i=1:length(idx)
    
    for j=1:length(liste_proprietes_iterations)-1
        %a(idx(i),j) = liste_proprietes_iterations{j+1}(1,idx(i));
        b(idx(i),j) = liste_proprietes_iterations{j+1}(2,idx(i));
    end
    
%     gcf = figure;
%     plot(real(a(idx(i),:)),'b*');
%     hold on;
%     plot(imag(a(idx(i),:)),'r*');
%     legend('Real part','Imaginary part','interpreter','latex');
%     title(sprintf('Material property $\\lambda$ at node %d',idx(i)),...
%         'interpreter','latex');
%     saveas(gcf,sprintf('lambda_%d.png',idx(i)));
%     close gcf;
    
    gcf = figure;
    plot(real(b(idx(i),:)),'b*');
    hold on;
    plot(imag(b(idx(i),:)),'r*');
    legend('Real part','Imaginary part','interpreter','latex');
    title(sprintf('Material property $\\mu$ at node %d',idx(i)),...
        'interpreter','latex');
    saveas(gcf,sprintf('mu_%d.png',idx(i)));
    close gcf;
    
end