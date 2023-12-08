function [flag] = processText(varargin)
%
% BRIEF DESCRIPTION OF THE SCRIPT
%
% This function was only created to read/write text files, in order to
% have structured results.
%
% NOMENCLATURE USED FOR THE SCRIPT
%
% - ampNoise: this is the noise amplitude which it is used in the main
%   file script.
% - kappa: 'regularization' parameter defined to improve the material
%   property resolution (linked to MECE-functional).
% - nIter: number of iterations.
% - filePath: describes the directory in which you want to save your
%   text file.
% - devMes: contains the deviation of the "constructed" and the "noisy"
%   measurements.
% - mesNoise: contains the values of the "noisy" measurements.
% - subMaterial: contains the values of the "identified" material property.
% - difProperty: contains the difference between the "identified" material
%   property and the theoretical value.
% - matList: contains the values of the "identified" material property for
%   iteration nIter.
%
% WRITTEN BY:
% Gabriel Nohra.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 6  % (line 275)

    filePath = varargin{1};
    gaussNoise = varargin{2};
    ampNoise = varargin{3};
    kappa = varargin{4};
    muTheoretical = varargin{5};

    oldPath = cd(filePath);
    fileID = fopen('results.txt','a+');
    fprintf(fileID,'-------------------------------------------------\n');
    fprintf(fileID,'GAUSSIAN NOISE AMPLITUDE:                   %0.4f %%\n',gaussNoise);
    fprintf(fileID,'NOISE AMPLITUDE:                            %0.4f %%\n',ampNoise);
    fprintf(fileID,'REGULARIZATION PARAMETER (KAPPA):           %0.2e\n',kappa);
    fprintf(fileID,'THEORETICAL MATERIAL PROPERTY (MU):         %0.1f + %0.1f*i\n\n',...
            real(muTheoretical), imag(muTheoretical));
    fclose(fileID);
    cd(oldPath);

    flag = true;

elseif nargin < 9 % (line 1300)

    filePath = varargin{1};
    nIter = varargin{2};
    prevFirstTerm = varargin{3};
    prevSecondTerm = varargin{4};
    firstTerm = varargin{5};
    secondTerm = varargin{6};
    devMes = varargin{7};
    mesNoise = varargin{8};

    oldPath = cd(filePath);
    fileID = fopen('results.txt','a+');
    fprintf(fileID,'*** FOR ITERATION NUMBER %0.0f:\n\n',nIter);
    fprintf(fileID,'MECE FUNCTIONAL (FOR %0.0f-th ITERATION):   %0.5e + %0.5e\n',...
            nIter-1,prevFirstTerm, prevSecondTerm);
    fprintf(fileID,'MECE FUNCTIONAL (FOR %0.0f-th ITERATION):   %0.5e + %0.5e\n\n',...
            nIter,firstTerm,secondTerm);
    fprintf(fileID,'-- NORMS OF... \n');
    fprintf(fileID,'MISMATCH:                                   %e\n',norm(devMes));
    fprintf(fileID,'NOISY MEASUREMENT:                          %e\n',norm(mesNoise));
    fprintf(fileID,'-- STANDARD DEVIATIONS OF... \n');
    fprintf(fileID,'REAL PART (MISMATCH):                       %e\n',...
            std(real(devMes))); 
    fprintf(fileID,'IMAGINARY PART (MISMATCH):                  %e\n',...
            std(imag(devMes)));
    fprintf(fileID,'REAL PART (NOISY MEASUREMENT):              %e\n',...
            std(real(mesNoise)));
    fprintf(fileID,'IMAGINARY PART (NOISY MEASUREMENT):         %e\n\n',...
            std(imag(mesNoise)));
    fclose(fileID);
    cd(oldPath);

elseif nargin < 5 && varargin{1} % (line 1910)

    subMaterial = varargin{2};
    matList = varargin{3};
    difProperty = varargin{4};

    oldPath = cd(filePath);
    fileID = fopen('results.txt','a+');
    fprintf(fileID,'-- NORMS OF THE MATERIAL PROPERTY... \n');
    fprintf(fileID,'ABSOLUTE NORM:                              %0.4f\n',...
            sum(abs(subMaterial),2)/size(subMaterial,2));;
    fprintf(fileID,'RELATIVE NORM:                              %e\n\n',...
            norm(difProperty)/norm(matList{nIter}));
    fclose(fileID);
    cd(oldPath);

end
    
end