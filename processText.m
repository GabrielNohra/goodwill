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

ampNoise = varargin{1};
kappa = varargin{2};
nIter = varargin{3};
devMes = varargin{4};
mesNoise = varargin{5};
filePath = varargin{6};
subMaterial = varargin{7};
matList = varargin{8};
difProperty = varargin{9};

oldPath = cd(filePath);
fileID = fopen('results.txt','a+');
fprintf(fileID,'--------------------------------------\n');
fprintf(fileID,'NOISE AMPLITUDE: \t \t \t \t \t %0.4f %%\n',ampNoise*100);
fprintf(fileID,'REGULARIZATION PARAMETER (KAPPA): \t \t \t \t \t %0.2e %%\n',kappa*100);
fprintf(fileID,'THEORETICAL MATERIAL PROPERTY: \t \t \t \t \t 1743 + 174.3*i\n\n');
fprintf(fileID,'*** FOR ITERATION NUMBER %f:\n\n',nIter);
fprintf(fileID,'-- NORMS:\n');
fprintf(fileID,'MISMATCH: \t \t \t \t \t %f\n',norm(devMes));
fprintf(fileID,'MEASUREMENT NOISE: \t \t \t \t \t %f\n',norm(mesNoise));
fprintf(fileID,'MATERIAL PROPERTY (ABS): \t \t \t \t \t %0.4f\n',sum(abs(subMaterial),2)/size(subMaterial,2));
fprintf(fileID,'MATERIAL PROPERTY (REL): \t \t \t \t \t %e\n\n',norm(difProperty)/norm(matList{nIter}));
fprintf(fileID,'-- STANDARD DEVIATIONS:\n');
fprintf(fileID,'REAL PART OF THE MISMATCH: \t \t \t \t \t %e\n',std(real(devMes))); 
fprintf(fileID,'IMAGINARY PART OF THE MISMATCH: \t \t \t \t \t %e\n',std(imag(devMes)));
fprintf(fileID,'REAL PART OF THE MEASUREMENT NOISE: \t \t \t \t \t %e\n',std(real(mesNoise)));
fprintf(fileID,'IMAGINARY PART OF THE MEASUREMENT NOISE: \t \t \t \t \t %e\n\n',std(imag(mesNoise)));
fclose(fileID);
cd(oldPath);
    
end