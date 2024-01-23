function kappa = adaptive(teoMat,simMat,kappa)

    % teoMat: theoretical material property.
    % simMat: simulated material property.
    % kappa: regularization parameter.

    aux = teoMat - simMat;

    exp = floor(log10(abs(power(teoMat-simMat,2))));

    if imag(aux)/real(aux) > 0
        if exp ~= 0
            kappa = kappa / power(10,exp);
        else
            kappa = kappa / rand()*10;
        end
    end

    if imag(aux)/real(aux) < 0
        if exp ~= 0
            kappa = kappa * power(10,exp);
        else
            kappa = kappa * rand()*10;
        end
    end
    
end