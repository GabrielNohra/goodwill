function [v] = sign_change( v, l, alpha )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

aux1 = find(real(v)<0);
aux2 = find(imag(v)<0);

while ( ~isempty(aux1) || ~isempty(aux2) )

    if ~isempty(aux1)

        for i=1:length(aux1)

            [x,y,m,a,b] = deal(0);

            m = ( imag( l( 2 ) ) -... % 1 + round( aux2(i)/(length(v)+1) )
                imag(v(aux1(i))) )/...
                ( real( l( 2 ) ) -... % 1 + round( aux2(i)/(length(v)+1) )
                real(v(aux1(i))) );
            
            x = 0;
            y = imag(v(aux1(i))) - real(v(aux1(i))) * m;

            a = real( l( 2 ) ) + alpha*... % 1 + round( aux2(i)/(length(v)+1) )
                ( x - real( l( 2 ) ) ); % 1 + round( aux2(i)/(length(v)+1) )
            b = imag( l( 2 ) ) + alpha*... % 1 + round( aux2(i)/(length(v)+1) )
                ( y - imag( l( 2 ) ) ); % 1 + round( aux2(i)/(length(v)+1) )

            v(aux1(i)) = a + 1i*b;

        end

    elseif ~isempty(aux2)

        for i=1:length(aux2)

            [x,y,m,a,b] = deal(0);

            m = ( imag( l( 2 ) ) -... % 1 + round( aux2(i)/(length(v)+1) )
                imag( v(aux2(i)) ) )/...
                ( real( l( 2 ) ) -... % 1 + round( aux2(i)/(length(v)+1) )
                real( v(aux2(i)) ) );
            
            x = real( v(aux2(i)) ) - imag( v(aux2(i)) ) * (1/m);
            y = 0;

            a = real( l( 2 ) ) + alpha*... % 1 + round( aux2(i)/(length(v)+1) )
                ( x - real( l( 2 ) ) ); % 1 + round( aux2(i)/(length(v)+1) )
            b = imag( l( 2 ) ) + alpha*... % 1 + round( aux2(i)/(length(v)+1) )
                ( y - imag( l( 2 ) ) ); % 1 + round( aux2(i)/(length(v)+1) )

            v(aux2(i)) = a + 1i*b;

        end

    end
    
    aux1 = find(real(v)<0);
    aux2 = find(imag(v)<0);

end

end

