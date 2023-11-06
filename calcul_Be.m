function [Be,grad_FF] = calcul_Be(d_Nf,Jaco,nb_DDL_par_noeud)

Be = zeros(2*nb_DDL_par_noeud,nb_DDL_par_noeud*size(d_Nf,2));

J_inv = inv(Jaco);

H = zeros(9,nb_DDL_par_noeud*size(d_Nf,2));
grad_FF = zeros(9,nb_DDL_par_noeud*size(d_Nf,2));

H(1,1:nb_DDL_par_noeud:end) = d_Nf(1,:); % ux,x
H(2,1:nb_DDL_par_noeud:end) = d_Nf(2,:); % ux,y
H(3,1:nb_DDL_par_noeud:end) = d_Nf(3,:); % ux,z
H(4,2:nb_DDL_par_noeud:end) = d_Nf(1,:); % uy,x
H(5,2:nb_DDL_par_noeud:end) = d_Nf(2,:); % uy,y
H(6,2:nb_DDL_par_noeud:end) = d_Nf(3,:); % uy,z
H(7,3:nb_DDL_par_noeud:end) = d_Nf(1,:); % uz,x
H(8,3:nb_DDL_par_noeud:end) = d_Nf(2,:); % uz,y
H(9,3:nb_DDL_par_noeud:end) = d_Nf(3,:); % uz,z

grad_FF(1,:) = H(1,:)*J_inv(1,1)+H(2,:)*J_inv(2,1)+H(3,:)*J_inv(3,1); % ux,x
grad_FF(2,:) = H(1,:)*J_inv(1,2)+H(2,:)*J_inv(2,2)+H(3,:)*J_inv(3,2); % ux,y
grad_FF(3,:) = H(1,:)*J_inv(1,3)+H(2,:)*J_inv(2,3)+H(3,:)*J_inv(3,3); % ux,z
grad_FF(4,:) = H(4,:)*J_inv(1,1)+H(5,:)*J_inv(2,1)+H(6,:)*J_inv(3,1); % uy,x
grad_FF(5,:) = H(4,:)*J_inv(1,2)+H(5,:)*J_inv(2,2)+H(6,:)*J_inv(3,2); % uy,y
grad_FF(6,:) = H(4,:)*J_inv(1,3)+H(5,:)*J_inv(2,3)+H(6,:)*J_inv(3,3); % uy,z
grad_FF(7,:) = H(7,:)*J_inv(1,1)+H(8,:)*J_inv(2,1)+H(9,:)*J_inv(3,1); % uz,x
grad_FF(8,:) = H(7,:)*J_inv(1,2)+H(8,:)*J_inv(2,2)+H(9,:)*J_inv(3,2); % uz,y
grad_FF(9,:) = H(7,:)*J_inv(1,3)+H(8,:)*J_inv(2,3)+H(9,:)*J_inv(3,3); % uz,z

Be(1,:) = grad_FF(1,:); % xx
Be(2,:) = grad_FF(5,:); % yy
Be(3,:) = grad_FF(9,:); % zz
Be(4,:) = grad_FF(2,:)+grad_FF(4,:); % xy
Be(5,:) = grad_FF(3,:)+grad_FF(7,:); % xz
Be(6,:) = grad_FF(6,:)+grad_FF(8,:); % yz

% Be(1,1:nb_DDL_par_noeud:end) = d_Nf(1,:)*J_inv(1,1); % xx
% Be(1,2:nb_DDL_par_noeud:end) = d_Nf(1,:)*J_inv(2,1);
% Be(1,3:nb_DDL_par_noeud:end) = d_Nf(1,:)*J_inv(3,1);
% Be(2,1:nb_DDL_par_noeud:end) = d_Nf(2,:)*J_inv(1,2); % yy
% Be(2,2:nb_DDL_par_noeud:end) = d_Nf(2,:)*J_inv(2,2);
% Be(2,3:nb_DDL_par_noeud:end) = d_Nf(2,:)*J_inv(3,2);
% Be(3,1:nb_DDL_par_noeud:end) = d_Nf(3,:)*J_inv(1,3); % zz
% Be(3,2:nb_DDL_par_noeud:end) = d_Nf(3,:)*J_inv(2,3);
% Be(3,3:nb_DDL_par_noeud:end) = d_Nf(3,:)*J_inv(3,3);
% Be(4,1:nb_DDL_par_noeud:end) = (d_Nf(1,:)*J_inv(1,2)+d_Nf(2,:)*J_inv(1,1)); % xy
% Be(4,2:nb_DDL_par_noeud:end) = (d_Nf(1,:)*J_inv(2,2)+d_Nf(2,:)*J_inv(2,1));
% Be(4,3:nb_DDL_par_noeud:end) = (d_Nf(1,:)*J_inv(3,2)+d_Nf(2,:)*J_inv(3,1));
% Be(5,1:nb_DDL_par_noeud:end) = (d_Nf(1,:)*J_inv(1,3)+d_Nf(3,:)*J_inv(1,1)); % xz
% Be(5,2:nb_DDL_par_noeud:end) = (d_Nf(1,:)*J_inv(2,3)+d_Nf(3,:)*J_inv(2,1));
% Be(5,3:nb_DDL_par_noeud:end) = (d_Nf(1,:)*J_inv(3,3)+d_Nf(3,:)*J_inv(3,1));
% Be(6,1:nb_DDL_par_noeud:end) = (d_Nf(2,:)*J_inv(1,3)+d_Nf(3,:)*J_inv(1,2)); % yz
% Be(6,2:nb_DDL_par_noeud:end) = (d_Nf(2,:)*J_inv(2,3)+d_Nf(3,:)*J_inv(2,2));
% Be(6,3:nb_DDL_par_noeud:end) = (d_Nf(2,:)*J_inv(3,3)+d_Nf(3,:)*J_inv(3,2));

end