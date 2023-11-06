function [p_G,w] = data_gauss (n_G)

% Points et poids de Gauss
p_G = zeros(n_G,3);
w = zeros(1,n_G);
        
if (n_G == 1)
    w=8; % poids
	p_G=[0,0,0]; % points
elseif (n_G == 6)
    w(1:6)=4/3;
	p_G=[-1,0,0;1,0,0;0,-1,0;0,1,0;0,0,-1;0,0,1];
elseif (n_G == 8)
    w(1:8)=1;
    a=1/sqrt(3);
    p_G(1,:)=[-a,-a,-a];
    p_G(2,:)=[a,-a,-a]; % coordonnées des points dans un espace [-1;1]^3
    p_G(3,:)=[a,a,-a];
    p_G(4,:)=[-a,a,-a];
    p_G(5,:)=[-a,-a,a];
    p_G(6,:)=[a,-a,a];
    p_G(7,:)=[a,a,a];
	p_G(8,:)=[-a,a,a];
elseif (n_G == 9)
    a=sqrt(3/5);
    w(1:8)=5/9;
    w(9)=32/9;
    p_G(1,:)=[-a,-a,-a];
    p_G(2,:)=[a,-a,-a];
    p_G(3,:)=[a,a,-a];
    p_G(4,:)=[-a,a,-a];
    p_G(5,:)=[-a,-a,a];
    p_G(6,:)=[a,-a,a];
    p_G(7,:)=[a,a,a];
    p_G(8,:)=[-a,a,a];
	p_G(9,:)=[0,0,0];
    
    %%%% version 1 : TRICHE (27 noeuds avec autre nomenclature)
elseif (n_G == 26)
    w1 = 5/9;
    w2 = 8/9;
    w3 = w1;
    a1 = -sqrt(3/5);
    a2 = 0;
    a3 = -a1;    
    p_G(1:9,:)=[a1 a1 a1; a2 a1 a1; a3 a1 a1; a1 a2 a1; a2 a2 a1; a3 a2 a1; a1 a3 a1; a2 a3 a1; a3 a3 a1];
    p_G(10:18,:)=[a1 a1 a2; a2 a1 a2; a3 a1 a2; a1 a2 a2; a2 a2 a2; a3 a2 a2; a1 a3 a2; a2 a3 a2; a3 a3 a2];
    p_G(19:27,:)=[a1 a1 a3; a2 a1 a3; a3 a1 a3; a1 a2 a3; a2 a2 a3; a3 a2 a3; a1 a3 a3; a2 a3 a3; a3 a3 a3];
    w(1:9) = [w1^3,w2*w1^2,w1^3,w2*w1^2,w1*w2^2,w2*w1^2,w1^3,w2*w1^2,w1^3];
    w(10:18) = [w2*w1^2,w1*w2^2,w2*w1^2,w1*w2^2,w2^3,w1*w2^2,w2*w1^2,w1*w2^2,w2*w1^2];
    w(19:27) = [w1^3,w2*w1^2,w1^3,w2*w1^2,w1*w2^2,w2*w1^2,w1^3,w2*w1^2,w1^3];
    
elseif (n_G == 27)
    w1 = 5/9;
    w2 = 8/9;
    a1 = -sqrt(3/5);
    a2 = 0;
    a3 = -a1;

    %%%% version 2
    p_G(1:8,:)=[a1 a1 a1; a3 a1 a1; a3 a3 a1; a1 a3 a1; a1 a1 a3; a3 a1 a3; a3 a3 a3; a1 a3 a3]; % sommets
    p_G(9:12,:)=[a1 a1 a2; a3 a1 a2; a3 a3 a2; a1 a3 a2]; % arretes etage intermediaire
    p_G(13:20,:)=[a2 a1 a1; a3 a2 a1; a2 a3 a1; a1 a2 a1; a2 a1 a3; a3 a2 a3; a2 a3 a3; a1 a2 a3]; % arretes bas et haut
    p_G(21:27,:)=[a2 a2 a1; a2 a1 a2; a3 a2 a2; a2 a3 a2; a1 a2 a2; a2 a2 a3; a2 a2 a2];% milieu faces
	w(1:8) = w1^3;
    w(9:20) = w2*w1^2;
    w(21:26) = w1*w2^2;
    w(27)=w2^3;
    
elseif (n_G == 64)
    x(1) = -sqrt((3/7)+(2/7)*sqrt(6/5));
    x(2) = -sqrt((3/7)-(2/7)*sqrt(6/5));
    x(3) = sqrt((3/7)-(2/7)*sqrt(6/5));
    x(4) = sqrt((3/7)+(2/7)*sqrt(6/5));
    m(1)=(18-sqrt(30))/36;
    m(2)=(18+sqrt(30))/36;
    m(3)=(18+sqrt(30))/36;
    m(4)=(18-sqrt(30))/36;
    p_G=zeros(n_G,3);
    w(1:n_G)=0;
    c=1;
    for i=1:4
     for j=1:4
      for k=1:4
       p_G(c,:) = [x(i) x(j) x(k)];
       w(c) = m(i)*m(j)*m(k);
       c=c+1;
      end
     end
    end
    
end

       