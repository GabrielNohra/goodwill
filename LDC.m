function [A] = LDC(data_LDC)

A = sparse(6,6);

%if ( strcmp(struct_param_masse_raideur.type_comportement,'elastique_lineaire_isotrope') == 1 )
%elseif ( strcmp(struct_param_masse_raideur.type_comportement,'elastique_lineaire_isotrope_quasi_incompressible') == 1 )
%elseif ( strcmp(struct_param_masse_raideur.type_comportement,'elastique_lineaire_isotrope_incompressible') == 1 )
%elseif ( strcmp(struct_param_masse_raideur.type_comportement,'visco_elastique_lineaire_isotrope') == 1 )
%elseif ( strcmp(struct_param_masse_raideur.type_comportement,'visco_elastique_lineaire_isotrope_quasi_incompressible') == 1 )
%elseif ( strcmp(struct_param_masse_raideur.type_comportement,'visco_elastique_lineaire_isotrope_incompressible') == 1 )
%end

lambda = data_LDC.lambda;
mu = data_LDC.mu;
a = lambda+2*mu;
b = lambda;
c = mu;
A = [a b b 0 0 0; b a b 0 0 0; b b a 0 0 0; 0 0 0 c 0 0; 0 0 0 0 c 0; 0 0 0 0 0 c];

end
