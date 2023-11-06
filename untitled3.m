mat_coef_mat_square = (mat_coef_mat.'*mat_coef_mat);
vec_second_membre_mat_square = (mat_coef_mat.'*vec_second_membre_mat);
vec_max_square = max(abs(mat_coef_mat_square),[],2);
%vec_max_square = ones(size(mat_coef_mat_square,1),1);
mat_coef_mat_square = mat_coef_mat_square./(vec_max_square*ones(1,size(mat_coef_mat_square,2)));
vec_second_membre_mat_square = vec_second_membre_mat_square./vec_max_square;
vec_sol_square = mat_coef_mat_square\vec_second_membre_mat_square;


if nnz(sum(real(vec_sol_square)<0))~=0
    vec_sol_square(real(vec_sol_square)<0) = vec_sol_square(real(vec_sol_square)<0) + 2*real(vec_sol_square(real(vec_sol_square)<0));
end

if nnz(sum(imag(vec_sol_square)<0))~=0
    vec_sol_square(imag(vec_sol_square)<0) = conj(vec_sol_square(imag(vec_sol_square)<0));
end

% --------

mat2 = (mat_red.'*mat_red);
vsm2 = (mat_red.'*vsm_red);
mat2_max = max(abs(mat2),[],2);
%vec_max_square = ones(size(mat_coef_mat_square,1),1);
mat2 = mat2./(mat2_max*ones(1,size(mat2,2)));
vsm2 = vsm2./mat2_max;
vec2 = mat2\vsm2;


if nnz(sum(real(vec2)<0))~=0
    vec2(real(vec2)<0) = vec2(real(vec2)<0) - 2*real(vec2(real(vec2)<0));
end

if nnz(sum(imag(vec2)<0))~=0
    vec2(imag(vec2)<0) = conj(vec2(imag(vec2)<0));
end

% 
% figure;subplot(2,1,1);hold on;plot(real(vec_second_membre_mat_square),'r');plot(real(mat_coef_mat_square*vec_sol_square),'b');grid;hold off;
% subplot(2,1,2);hold on;plot(real(vsm2),'r');plot(real(mat2*vec2),'b');grid;hold off;
% 
% figure;subplot(2,1,1);hold on;plot(real(vec_second_membre_mat_square-mat_coef_mat_square*vec_sol_square),'r');plot(imag(vec_second_membre_mat_square-mat_coef_mat_square*vec_sol_square),'b');grid;
% subplot(2,1,2);hold on;plot(real(vsm2-mat2*vec2),'r');plot(imag(vsm2-mat2*vec2),'b');grid;

figure;subplot(2,1,1);hold on;plot(real(vec_sol_square),'r');plot(imag(vec_sol_square),'b');grid;
subplot(2,1,2);hold on;plot(real(vec2),'r');plot(imag(vec2),'b');grid;