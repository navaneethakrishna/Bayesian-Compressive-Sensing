clear;
clc;
%% Variable initialization

count = 1;

N_orig = 512;%250;
S = 10;
K = 100;%60;
len = 20;
ro = 0.9995 : 0.0001 : 0.9999;

mse_bay = zeros(1,len);
mse_bp = zeros(1,len);
mse_fast = zeros(1,len);

rce_bay = zeros(1,len);
rce_bp = zeros(1,len);
rce_fast = zeros(1,len);

Cc_bay = zeros(1,len);
Cc_bp  = zeros(1,len);
Cc_fast = zeros(1,len);

for ro_no = 1:length(ro)
    
    ro_vec = zeros(1,K);
    for i =  0: K-1
        ro_vec(1, i+1) = ro(ro_no)^i;
    end
    
    ro_mat = zeros(K,K);
    ro_mat(1,:) = ro_vec;
    for i = 2:K
        ro_mat(i,:) = circshift(ro_mat(i-1,:),1);
    end
for trials  = 100 : 100 : 2000 
    
sig_n = 0.01;
alpha_0 = 1/sig_n;
iter = trials;
alpha = 0.01*ones(1,N_orig); % precisions of non-informative priory x
pos_indx = 1 : N_orig; 

%% Generating phi

% Random Phi
phi_orig = randn(K,N_orig);
for i = 1:K
    phi_orig(i,:) = phi_orig(i,:)/norm(phi_orig(i,:));
end

% Adding correlation using ro_mat
phi_orig = sqrtm(ro_mat)*phi_orig;

%% Generating x and obtaining y
x_orig = randperm(N_orig);
v = randn(1,S);
no = 1;
for i = 1:N_orig
    if(x_orig(i) > S)
        x_orig(i) = 0;
    else
        x_orig(i) = 10*sign(v(no));
        no = no + 1;
    end
end

n = sig_n*randn(K,1);
y = phi_orig*x_orig' + n;

x = x_orig;
phi = phi_orig;

%% Method iterated in paper
for i = 1:iter
    sigma_mat = inv(diag(alpha) + alpha_0*(phi')*phi); % estimation of covariance matrix
    N = size(sigma_mat,1);
    I = find((1 - (diag(sigma_mat))'*diag(alpha)) < 10^-14);
    if(~ isempty(I)) 
        sigma_mat(:,I) = inf;
        sigma_mat(I,:) = inf;
        for a = 0 : 10*N
            h = mod(a,N) + 1;
            if(sigma_mat(h,:) == inf)
                for j = h : N-1
                    sigma_mat(j,:) = sigma_mat(j+1,:);
                end
                sigma_mat(N,:) = inf;
            end
        end
        for a = 0 : 10*N
            h = mod(a,N) + 1;
            if(sigma_mat(:,h) == inf)
                for j = h : N-1
                    sigma_mat(:,j) = sigma_mat(:,j+1);
                end
                sigma_mat(:,N) = inf;
            end
         end
                 
        sigma_mat = sigma_mat(1:(N-size(I,2)),1:(N-size(I,2)));
        
        alpha(I) = inf;
        for a = 0 : 10*N
            h = mod(a,N) + 1;
            if(alpha(h) == inf)
                for j = h : N-1
                    alpha(j) = alpha(j+1);
                end
                alpha(N) = inf;
            end
        end
        alpha = alpha(1 : N-size(I,2));
        pos_indx(I) = inf;
        for a = 0 : 10*N
            h = mod(a,N) + 1;
            if(pos_indx(h) == inf)
                for j = h : N-1
                    pos_indx(j) = pos_indx(j+1);
                end
                pos_indx(N) = inf;
            end
        end
        pos_indx = pos_indx(1 : N-size(I,2));
        
        phi(:,I) = inf;
         for a = 0 : 10*N
             h = mod(a,N) + 1;
            if(phi(:,h) == inf)
                for j = h : N-1
                    phi(:,j) = phi(:,j+1);
                end
                phi(:,N) = inf;
            end
         end
         phi = phi(1:K,1:(N-size(I,2)));
        x(I) = 100;
        for a = 0 : 10*N
            h = mod(a,N) + 1;
            if(x(h) == 100)
                for j = h : N-1
                    x(j) = x(j+1);
                end
                x(N) = 100;
            end
        end
        x = x(1 : N-size(I,2));
        N = N - size(I,2);        
    end
   
    mu_reduced = alpha_0*(sigma_mat)*(phi')*y; % estimation of mean vector
    if(length(mu_reduced) == 5)
       disp(['i = ',num2str(i)]); 
    end
    gamma = zeros(1,N);
    for j = 1:N
        gamma(j) = 1- alpha(j)*sigma_mat(j,j);
        alpha(j) = gamma(j)/power(mu_reduced(j),2); % signal component pecision estimation
    end
    alpha_0 = (N_orig - sum(gamma))/norm(y - phi*x'); % noise precision estimation
end

sigma_mat = inv(diag(alpha) + alpha_0*(phi')*phi);
mu_reduced = alpha_0*(sigma_mat)*(phi')*y;

mu = zeros(1,N_orig);
ref = 1:N_orig;
pos = setdiff(ref, pos_indx);
mu(pos) = 0;
mu(pos_indx) = mu_reduced;

errbars_1 = sqrt(diag(sigma_mat));
err = zeros(N_orig,1);
err(pos_indx) = errbars_1;

%% Basis pursuit

x0 = phi_orig'*inv(phi_orig*phi_orig')*y;
epsilon =  sig_n*sqrt(K)*sqrt(1 + 2*sqrt(2)/sqrt(K));
x_BP = l1qc_logbarrier(x0, phi_orig, [], y, epsilon, 1e-3);

%% Fast algorithm

initsigma_mat2 = std(y)^2/1e2;
[weights,used,sigma_mat2,errbars] = BCS_fast_rvm(phi_orig,y,initsigma_mat2,1e-8);
x_BCS = zeros(N_orig,1); Err = zeros(N_orig,1);
x_BCS(used) = weights; Err(used) = errbars;

%% Analyses

% Mean square error
N = N_orig;
mu = mu';
x_orig = x_orig';
mse_bay(ro_no, count) = sum(power((x_orig - mu),2))/N;
mse_bp(ro_no, count) = sum(power((x_orig - x_BP),2))/N;
mse_fast(ro_no, count) = sum(power((x_orig-x_BCS),2))/N;

% reconstruction error - estimation error by signal
rce_bay(ro_no, count) = norm(x_orig - mu)/norm(x);
rce_bp(ro_no, count) = norm(x_orig - x_BP)/norm(x);
rce_fast(ro_no, count) = norm(x_orig - x_BCS)/norm(x);

% correlation coeffifient
Cc_bay(ro_no, count) = (N*dot(x_orig,mu) - sum(x_orig)*sum(mu))/((sqrt(N*power(norm(x_orig),2) - power(sum(x_orig),2)))*(sqrt(N*power(norm(mu),2) - power(sum(mu),2))));
Cc_bp(ro_no, count) = (N*dot(x_orig,x_BP) - sum(x_orig)*sum(x_BP))/((sqrt(N*power(norm(x_orig),2) - power(sum(x_orig),2)))*(sqrt(N*power(norm(x_BP),2) - power(sum(x_BP),2))));
Cc_fast(ro_no, count) = (N*dot(x_orig,x_BCS) - sum(x_orig)*sum(x_BCS))/((sqrt(N*power(norm(x_orig),2) - power(sum(x_orig),2)))*(sqrt(N*power(norm(x_BCS),2) - power(sum(x_BCS),2))));

count = count + 1;
end
count = 1;
end
%% Plots - to be used to observe output after every iteration for every value of ro

% TO be used to generate output waveforms for individual values of rho

subplot(4,1,1) % input signal plot
plot((1:1:N),x_orig);
axis([1 N+1 -max(abs(x_orig))-0.1 max(abs(x_orig))+0.1]);
title('(a) Input sparse vector');box on;

subplot(4,1,2) % usual RVM Bayesian method
errorbar((1:1:N),mu,err);
axis([1 N+1 -max(abs(x))-0.1 max(abs(x))+0.1]);
title('(b) Reconstructed using RVM Method');box on;

subplot(4,1,3); errorbar((1:1:N),x_BCS,Err); % Fast RVM method
axis([1 N+1 -max(abs(x_orig))-0.1 max(abs(x_orig))+0.1]);
title('(c) Reconstructed using Fast RVM Method');box on;

subplot(4,1,4) % basis pursuit method 
plot((1:1:N),x_BP);
axis([1 N+1 -max(abs(x_orig))-0.1 max(abs(x_orig))+0.1]);
title('(d) Reconstructed using Basis  algorithm');box on;

sgtitle(['N =', num2str(N),', K = ',num2str(K), ', S = ',num2str(S), ', \rho = 0.9995']);

%% Analysis

figure('Name','Mean Square Error - Bayesian','NumberTitle','off');
plot(100:100:2000,mse_bay(1,:),100:100:2000,mse_bay(2,:),100:100:2000,mse_bay(3,:),100:100:2000,mse_bay(4,:),100:100:2000,mse_bay(5,:));
xlabel('No of iterations'); ylabel('MSE');title('Mean Square Error of Bayesian Approach');
lgd1 = legend('0.9995','0.9996','0.9997','0.9998','0.9999');
title(lgd1,'Values of \rho');
figure('Name','Reconstruction error Error','NumberTitle','off');
plot(100:100:2000,rce_bay(1,:),100:100:2000,rce_bay(2,:),100:100:2000,rce_bay(3,:),100:100:2000,rce_bay(4,:),100:100:2000,rce_bay(5,:));
xlabel('No of iterations'); ylabel('Reconstruction Error');title('Reconstruction Error of Bayesian Approach');
lgd2 = legend('0.9995','0.9996','0.9997','0.9998','0.9999');
title(lgd2,'Values of \rho');
figure('Name','Input  - Output Correlation','NumberTitle','off');
plot(100:100:2000,Cc_bay(1,:),100:100:2000,Cc_bay(2,:),100:100:2000,Cc_bay(3,:),100:100:2000,Cc_bay(4,:),100:100:2000,Cc_bay(5,:));
xlabel('No of iterations'); ylabel('Input - Output correlation');title('Input - Output Correlation of Bayesian Approach');
lgd3 = legend('0.9995','0.9996','0.9997','0.9998','0.9999');
title(lgd3,'Values of \rho');

figure('Name','Worst case - MSE','NumberTitle','off');
plot(100:100:2000,mse_bay(5,:),100:100:2000,mse_bp(5,:),100:100:2000,mse_fast(5,:));legend('Bayesian','Basis  pursuit','Fast algorithm');
xlabel('No of iterations'); ylabel('MSE');title({'Worst Case Mean Square Error of the three approaches';'\rho = 0.9999'});
figure('Name','Worst case - Reconstruction error','NumberTitle','off');
plot(100:100:2000,rce_bay(5,:),100:100:2000,rce_bp(5,:),100:100:2000,rce_fast(5,:));legend('Bayesian','Basis  pursuit','Fast algorithm');
xlabel('No of iterations'); ylabel('Reconstruction Error');title({'Worst Case Reconstruction Error of the three approaches';'\rho = 0.9999'});
figure('Name','Worst case - Correlation','NumberTitle','off');
plot(100:100:2000,Cc_bay(5,:),100:100:2000,Cc_bp(5,:),100:100:2000,Cc_fast(5,:));legend('Bayesian','Basis  pursuit','Fast algorithm');
xlabel('No of iterations'); ylabel('Input - Output Correlation');title({'Worst Case Input - Output Correlation of the three approaches';'\rho = 0.9999'});