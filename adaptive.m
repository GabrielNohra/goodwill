function kappa = adaptive(teoMat,simMat,kappa)

    % teoMat: theoretical material property.
    % simMat: simulated material property.
    % kappa: regularization parameter.

    aux = simMat - teoMat;

    exp = floor(log10(abs(power(simMat-teoMat,2))));

    if imag(aux)/real(aux) > 0

        if real(simMat) > real(teoMat)
            kappa = kappa * power(10,exp);
        else
            kappa = kappa / power(10,6*floor(exp));
        end

    end

    if imag(aux)/real(aux) < 0

        if real(simMat) > real(teoMat)
            kappa = kappa * power(10,exp);
        else
            kappa = kappa / power(10,6*floor(exp));
        end

end