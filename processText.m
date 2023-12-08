function processText(varargin)
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

if nargin < 7 && ~varargin{2} % (line 275)

    filePath = varargin{1};
    gaussNoise = varargin{3};
    ampNoise = varargin{4};
    kappa = varargin{5};
    muTheoretical = varargin{6};

    oldPath = cd(filePath);
    fileID = fopen('results.txt','a+');
    fprintf(fileID,'-----------------------------------------------------------------------------\n\n');
    fprintf(fileID,'REGULARIZATION PARAMETER (KAPPA):               %0.2e\n',kappa);
    fprintf(fileID,'GAUSSIAN NOISE AMPLITUDE:                       %e\n',gaussNoise);
    fprintf(fileID,'NOISE AMPLITUDE:                                %e\n',ampNoise);
    fprintf(fileID,'THEORETICAL MATERIAL PROPERTY (MU):             %0.1f + %0.1f*i\n\n',...
            real(muTheoretical), imag(muTheoretical));
    fprintf(fileID,'-----------------------------------------------------------------------------\n\n');
    fclose(fileID);
    cd(oldPath);

elseif nargin < 10 && ~varargin{2} % (line 1300)

    filePath = varargin{1};
    nIter = varargin{3};
    prevFirstTerm = varargin{4};
    prevSecondTerm = varargin{5};
    firstTerm = varargin{6};
    secondTerm = varargin{7};
    devMes = varargin{8};
    mesNoise = varargin{9};
    oldPath = cd(filePath);
    fileID = fopen('results.txt','a+');
    fprintf(fileID,'************************ FOR ITERATION NUMBER %0.0f *****************************n\n',nIter);
    fprintf(fileID,'MECE FUNCTIONAL (FOR %0.0f-th ITERATION):           %e + %e\n',...
            nIter-1,prevFirstTerm, prevSecondTerm);
    fprintf(fileID,'MECE FUNCTIONAL (FOR %0.0f-th ITERATION):           %e + %e\n\n',...
            nIter,firstTerm,secondTerm);
    fprintf(fileID,'------------------------------- NORMS ---------------------------------------\n\n');
    fprintf(fileID,'MISMATCH:                                       %e\n',norm(devMes));
    fprintf(fileID,'NOISY MEASUREMENT:                              %e\n\n',norm(mesNoise));
    fprintf(fileID,'-------------------------- STANDARD DEVIATIONS ------------------------------\n\n');
    fprintf(fileID,'REAL PART (MISMATCH):                           %e\n',...
            std(real(devMes))); 
    fprintf(fileID,'IMAGINARY PART (MISMATCH):                      %e\n',...
            std(imag(devMes)));
    fprintf(fileID,'REAL PART (NOISY MEASUREMENT):                  %e\n',...
            std(real(mesNoise)));
    fprintf(fileID,'IMAGINARY PART (NOISY MEASUREMENT):             %e\n\n',...
            std(imag(mesNoise)));
    fclose(fileID);
    cd(oldPath);

elseif nargin < 7 && varargin{2} % (line 1910)

    filePath = varargin{1};
    nIter = varargin{3};
    subMaterial = varargin{4};
    matList = varargin{5};
    difProperty = varargin{6};

    oldPath = cd(filePath);
    fileID = fopen('results.txt','a+');
    fprintf(fileID,'-------------------- NORM OF THE MATERIAL PROPERTY (MU) ---------------------\n\n');
    fprintf(fileID,'ABSOLUTE NORM:                                  %0.4f\n',...
            sum(abs(subMaterial),2)/size(subMaterial,2));;
    fprintf(fileID,'RELATIVE NORM:                                  %e\n\n',...
            norm(difProperty)/norm(matList{nIter}));
    fclose(fileID);
    cd(oldPath);

end
    
end