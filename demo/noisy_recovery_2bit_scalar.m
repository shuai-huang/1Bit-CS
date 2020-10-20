sigma = 2;			% the oversampling ratio
rho = 0.1;			% the sparsity ratio
N = 1000;       	% the signal length
M = sigma*N;      	% the number of measurements
S = ceil(rho*N);  	% the number of nonzero entries

num_c_true = 1; 	% the true number of Gaussian mixtures
noise_std = 0.02;	% the noise standard deviation
bit_num = 2;		% the quantization bit number

LN_num = 2;						% the number of computing threads
LN = maxNumCompThreads(LN_num);	% set the largest number of computing threads

max_pe_ite = 20;		% the maximum number of iterations for AMP-PE and AMP-AWGN
max_pe_inner_ite = 20;	% the maximum number of inner iterations to estimate the parameters
max_oracle_ite = 20;	% the maximum number of iterations for AMP-oracle, note that for some cases, AMP-oracle needs more iterations to get the best performance
max_ite = 1000;			% the maximum number of iterations for IHT, OMP, CoSaMP, L1-min
cvg_thd = 1e-6; 		% convergence threshold
kappa = 1;      		% learning rate or damping rate
verbose = 0;			% "1" - output convergence values in every iteration; "0" - do not output convergence values in every iteration

% set the quantization thresholds and step size properly
% they need to be adjusted for different signals
quant_thd_min = -1;										% the minimum quantization threshold
quant_thd_max = 1;										% the maximum quantization threshold
quant_step = (quant_thd_max-quant_thd_min)/(2^bit_num);	% quantizatin step size

% compute the quantization thresholds
quant_thd = quant_thd_min;
for (i=(1:(2^bit_num -1)))
    quant_thd = [quant_thd quant_thd_min+i*quant_step];
end
quant_thd = [quant_thd quant_thd_max];

snr_seq_mat = [];	% the matrix holding all the signal to noise ration values

% run 10 random trials here
for (trial_num = 1:10)
    fprintf('Trial %d\n', trial_num)

    rng(trial_num); % set random seed
    snr_seq = [];	% the vector holding the snr values at each random trial

    %%%%%%%%%%%%%%%%%%%
    %% create signal %%
    %%%%%%%%%%%%%%%%%%%

    % true Gaussian mixture weights
    omega_true = rand(num_c_true,1);
    omega_true = omega_true/sum(omega_true);

    nonzeroW = [];      % nonzero entries
    theta_true = [];    % true Gaussian mixtures means
    phi_true = [];      % true Gaussian mixtures variances
    for (i=1:num_c_true)
        theta_tmp = 0;  % make sure the mean is 0 for this particular 1bit CS 
        theta_true = [theta_true; theta_tmp];
        phi_tmp = 1;	% fix the variance to make sure the noise level is fixed
        phi_true = [phi_true; phi_tmp];
        S_tmp = round(S*omega_true(i));
        nonzeroW = [nonzeroW; normrnd(theta_tmp, sqrt(phi_tmp), S_tmp, 1)];	% the nonzero entries
    end

    % double check since we rounded the number  before
    if (length(nonzeroW)>N)
        nonzeroW = nonzeroW(1:N);
    end

    ind = randperm(N);
    indice = ind(1:length(nonzeroW));
    x = zeros(N, 1);
    x(indice) = nonzeroW;   % true signal

    % the measurement matrix A
    % each column is normalized so that the rows of A have unit norms
    A = randn(M,N);
    A_norm=sqrt(sum(A.^2));
    for(j=1:N)
        A(:,j)=A(:,j)/A_norm(j);
    end
    
	% needed for scalar-version AMP computation    
    A_sq_sum = norm(A, 'fro')^2;  % compute the squared frobenius norm of A

    y_noiseless = A*x;    % noiseless linear measurements

    y_ori = y_noiseless + normrnd(0, noise_std, size(y_noiseless));	% noisy measurements

    % convert "y_ori" to quantized symbols "y" in the set {2,3,4,5,6,...}
    % "y_quant" is the "low resolution" version of the "y_ori"
    
	y = zeros(size(y_ori));
    y_quant = zeros(size(y_ori));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% note that the quantized symbols start from "2" !!! %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for (i=1:length(y_ori))
        for (j=2:length(quant_thd))
            if ( (y_ori(i)>=quant_thd(j-1))&&(y_ori(i)<quant_thd(j)) )
                y(i) = j;
                y_quant(i) = 0.5*(quant_thd(j-1)+quant_thd(j));
                break;
            end
        end
    end

    % just in case the measurements are beyond the quantization range
    for (i=1:length(y_ori))
        if (y_ori(i)<quant_thd(1))
            y(i) = 2;
            y_quant(i) = 0.5*(quant_thd(1)+quant_thd(2));
        end
        if (y_ori(i)>=quant_thd(length(quant_thd)))
            y(i) = length(quant_thd);
            y_quant(i) = 0.5*(quant_thd(length(quant_thd)-1)+quant_thd(length(quant_thd)));
        end
    end
    

    %%%%%%%%%%%%%%%%%%%%
    %% initialization %%
    %%%%%%%%%%%%%%%%%%%%

    % use the l2-min solution to get an estimate of the distribution parameters
    % if you have some vague notion about the distribution parameters, you can directly set the distribution parameters
    x_hat = zeros(N,1);
    d_cg = zeros(N,1);
    d_cg = -A'*(A*x_hat-y_quant);
    r_cg = d_cg;

	% use conjungate gradient descent to compute the least square solution
    max_cg_ite = 20; 

    for (ite=1:max_cg_ite)
        a_cg_n = sum(r_cg.*r_cg);
        d_cg_tmp = A*d_cg;
        a_cg_d = sum(d_cg_tmp.*d_cg_tmp);
        a_cg = a_cg_n / a_cg_d;
        x_hat_pre = x_hat;
        x_hat = x_hat + a_cg*d_cg;
        cvg_cg_val = norm(x_hat-x_hat_pre,'fro')/norm(x_hat,'fro');
        if (cvg_cg_val<cvg_thd)
            break;
        end

        r_cg_new = r_cg - a_cg*(A'*(A*d_cg));

        b_cg = sum(r_cg_new.*r_cg_new)/sum(r_cg.*r_cg);
        d_cg = r_cg_new + b_cg*d_cg;

        r_cg = r_cg_new;
    end

	% initialize the AMP algorithm
    tau_x = var(x_hat)+1e-12;	% initialize scalar signal variance
    s_hat = zeros(M,1); 		% initialize s_hat with all zeros

    % initialize input distribution parameters
    lambda = 0.5;			% the sparsity ratio
	num_c = 1;				% the number of Gaussian mixture component

    % initialize Gaussian mixtures parameters
    theta=zeros(num_c, 1);  % Gaussian mean
    phi=ones(num_c, 1);     % Gaussian variance
    omega=zeros(num_c,1);   % Gaussian mixture weights

    x_hat_nz = abs(x_hat(x_hat~=0));
    idx_seq = kmeans(x_hat_nz, num_c);
    for (i=1:num_c)
        x_hat_tmp = x_hat_nz(idx_seq==i);
        if (length(x_hat_tmp)>0)
            theta(i) = mean(x_hat_tmp);
            phi(i) = var(x_hat_tmp)+1e-12;  % avoid zero variance
            omega(i) = length(x_hat_tmp)/length(x_hat_nz);
        else 
            theta(i) = 0;
            phi(i) = 1e-12; % avoid zero variance
            omega(i) = 0;
        end
    end

    % initialize white Gaussian noise variance
    tau_w = 1e-6;
    
    % initialize with all zero vector
    x_hat = zeros(N,1);

    % set GAMP parameters
    gamp_par.A_sq_sum	= A_sq_sum;			% the Frobenius norm of the measurement matrix
    gamp_par.max_pe_ite    = max_pe_ite;	% maximum number of iterations for AMP
    gamp_par.max_pe_inner_ite = max_pe_inner_ite;	% maximum number of inner iterations for parameter estimation
    gamp_par.cvg_thd    = cvg_thd; 			% the convergence threshold
    gamp_par.kappa      = kappa;   			% learning rate
    gamp_par.verbose	= verbose;			% "0" or "1", decides whether to output convergence values in each iteration

    gamp_par.x_hat      = x_hat; 		  			% initialize with all zero vector
    gamp_par.tau_x      = tau_x;		% signal variance
    gamp_par.s_hat      = s_hat;   					% dummy variable from output function

    % set input distribution parameters
    input_par.lambda    = lambda;	% the sparsity ratio
    input_par.theta     = theta;	% the means of the Gaussian mixtures
    input_par.phi       = phi;    	% the variances of the Gaussian mixtures
    input_par.omega     = omega;  	% the weights of the Gaussian mixtures
    input_par.num_c     = num_c;  	% the number of Gaussian mixtures

    % set output distribution parameters
    output_par.tau_w    = tau_w; 		% the white-Gaussian noise variance
    output_par.quant_thd = quant_thd; 	% quantization threshold
    
    
    % recovery the signal from quantized measurements using different approaches

    %%%%%%%%%%%%%%%%%%%%%
    %% AMP-PE recovery %%
    %%%%%%%%%%%%%%%%%%%%%

    [res, input_par_new, output_par_new] = gamp_bgm_multi_bit_scalar(A, y, gamp_par, input_par, output_par);

	% for 1-bit CS the magnitude info is lost, we need to normalize it first
    snr_val = snr(x, res.x_hat/sum(abs(res.x_hat))*sum(abs(x))-x);
    snr_seq = [snr_seq snr_val];

    
    %%%%%%%%%%%%%%%%%%%%%%%
    %% AMP-AWGN recovery %%
    %%%%%%%%%%%%%%%%%%%%%%%
    
    [res, input_par_new, output_par_new] = gamp_bgm_scalar(A, y_quant, gamp_par, input_par, output_par);
    
	% for 1-bit CS the magnitude info is lost, we need to normalize it first
    snr_val = snr(x, res.x_hat/sum(abs(res.x_hat))*sum(abs(x))-x);
    snr_seq = [snr_seq snr_val];

    
    %%%%%%%%%%%%%%%%%%
    %% IHT recovery %%
    %%%%%%%%%%%%%%%%%%
    x_hat = zeros(N,1);

    % normalized the matrix and measurements as required by iterative hard thresholding
    [U, S_diag, V] = eig(A'*A);
    A_normalized = A/sqrt(max(diag(S_diag)));
    y_quant_normalized = y_quant/sqrt(max(diag(S_diag)));

    Par.tol = cvg_thd;
    Par.X0 = x_hat;
    Par.maxiter = max_ite;

    K = length(nonzeroW);

    res = ssr_iht_multi_bit_k(y_quant_normalized, A_normalized, Par, K);

	% for 1-bit CS the magnitude info is lost, we need to normalize it first
    snr_val = snr(x, res/sum(abs(res))*sum(abs(x))-x);
    snr_seq = [snr_seq snr_val];

    
    %%%%%%%%%%%%%%%%%%
    %% OMP recovery %%
    %%%%%%%%%%%%%%%%%%
    
    % initialize with zero vector
    x_hat = zeros(N,1);

	clear Par;
    Par.X0 = x_hat;
    Par.maxiter = max_ite;

    K = length(nonzeroW);

    res = OMP_init(A, y_quant, K, Par);

	% for 1-bit CS the magnitude info is lost, we need to normalize it first
    snr_val = snr(x, res/sum(abs(res))*sum(abs(x))-x);
    snr_seq = [snr_seq snr_val];
    
    
    %%%%%%%%%%%%%%%%%%%%%
    %% CoSaMP recovery %%
    %%%%%%%%%%%%%%%%%%%%%
    
    % initialize with zero vector
    x_hat = zeros(N,1);

	clear Par;
    Par.normTol = cvg_thd;
    Par.support_tol = cvg_thd;
    Par.X0 = x_hat;
    Par.maxiter = max_ite;

    K = length(nonzeroW);

    res = CoSaMP_init_fast(A, y_quant, K, Par);

	% for 1-bit CS the magnitude info is lost, we need to normalize it first
    snr_val = snr(x, res/sum(abs(res))*sum(abs(x))-x);
    snr_seq = [snr_seq snr_val];

    
    %%%%%%%%%%%%%%%%%%%%%
    %% L1-Min recovery %%
    %%%%%%%%%%%%%%%%%%%%%
    
    % initialize with zero vector
    x_hat = zeros(N,1);

	clear Par;
    Par.tol = cvg_thd;
    Par.X0 = x_hat;
    Par.maxiter = max_ite;
    Par.innermaxiter = 1;
    Par.epsilon = 1e-12;
    [U, S_diag, V] = eig(A'*A);
    Par.kappa = 2*max(diag(S_diag));    % the Lipschitz constant
    Par.fun = 'l1';
    Par.p = 1;

	% remember to tune the regularization parameter lambda_reg for different types of signals
	lambda_reg = 0.4;

    res = ssr_l1(y_quant, A, Par, lambda_reg);

	% for 1-bit CS the magnitude info is lost, we need to normalize it first
    snr_val = snr(x, res/sum(abs(res))*sum(abs(x))-x);
    snr_seq = [snr_seq snr_val];
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %% AMP-oracle recovery %%
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    clear gamp_par input_par output_par;
    
    % set GAMP parameters
    gamp_par.A_sq_sum	= A_sq_sum;			% the Frobenius norm of the measurement matrix
    gamp_par.max_pe_ite    = max_oracle_ite;	% maximum number of iterations for AMP
    gamp_par.cvg_thd    = cvg_thd;	 		% the convergence threshold
    gamp_par.verbose	= verbose;			% "0" or "1", decides whether to output convergence values in each iteration

    gamp_par.x_hat      = zeros(N,1);   			% estimated signal
    gamp_par.tau_x      = tau_x;   	% signal variance
    gamp_par.s_hat      = s_hat;   					% dummy variable from output function

    % set input distribution parameters
    input_par.lambda    = rho; 			% the sparity ratio
    input_par.theta     = theta_true;  	% the means of the Gaussian mixtures
    input_par.phi       = phi_true;    	% the variances of the Gaussian mixtures
    input_par.omega     = omega_true;  	% the weights of the Gaussian mixtures
    input_par.num_c     = num_c_true;  	% the number of Gaussian mixtures

    % set output distribution parameters
    output_par.tau_w    = noise_std^2; 	% the white-Gaussian noise variance
    output_par.quant_thd = quant_thd; 	% quantization threshold
    
    [res, input_par_new, output_par_new] = gamp_bgm_multi_bit_scalar_oracle(A, y, gamp_par, input_par, output_par);
    
	% for 1-bit CS the magnitude info is lost, we need to normalize it first
    snr_val = snr(x, res.x_hat/sum(abs(res.x_hat))*sum(abs(x))-x);
    snr_seq = [snr_seq snr_val];

	
	% save teh snr values from each random trial to snr_seq_mat
	snr_seq_mat = [snr_seq_mat; snr_seq];
	
end

% plot the snr values from different approaches
snr_seq_mean = mean(snr_seq_mat);
snr_seq_upper = std(snr_seq_mat);
snr_seq_lower = std(snr_seq_mat);

snr_x = categorical({'AMP-PE', 'AMP-AWGN', 'IHT', 'OMP', 'CoSaMP', 'L1-Min', 'AMP-Oracle'});
snr_x = reordercats(snr_x, {'AMP-PE', 'AMP-AWGN', 'IHT', 'OMP', 'CoSaMP', 'L1-Min', 'AMP-Oracle'});

figure;
bar(snr_x, snr_seq_mean);
ylabel('SNR (dB)')
title('2-bit CS: M/N=2, S/N=10%, \sigma_w=0.02')
hold on
er = errorbar(snr_x, snr_seq_mean, snr_seq_lower, snr_seq_upper);
er.Color = [0 0 0];
er.LineStyle = 'none';
hold off

