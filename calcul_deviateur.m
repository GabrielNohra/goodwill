function deviateur_sigma = calcul_deviateur(vec_sigma_Voigt)
% vec_sigma_Voigt = (sxx syy szz sxy sxz syz)'

deviateur_sigma = [1/3*(2*vec_sigma_Voigt(1)-vec_sigma_Voigt(2)-vec_sigma_Voigt(3)) vec_sigma_Voigt(4) vec_sigma_Voigt(5); ...
                   vec_sigma_Voigt(4) 1/3*(-vec_sigma_Voigt(1)+2*vec_sigma_Voigt(2)-vec_sigma_Voigt(3)) vec_sigma_Voigt(6); ...
                   vec_sigma_Voigt(5) vec_sigma_Voigt(6) 1/3*(-vec_sigma_Voigt(1)-vec_sigma_Voigt(2)+2*vec_sigma_Voigt(3))];

