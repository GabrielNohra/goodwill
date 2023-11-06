function [U] = CL_assemblage (Us,U_impose,vec_n_DDL_conserves,vec_n_DDL_supprimes)

nb_DDL = length(vec_n_DDL_conserves)+length(vec_n_DDL_supprimes);
U = zeros(nb_DDL,1);
if ( ~isempty(vec_n_DDL_conserves) )
 U(vec_n_DDL_conserves) = Us;
end;
U(vec_n_DDL_supprimes) = U_impose;

