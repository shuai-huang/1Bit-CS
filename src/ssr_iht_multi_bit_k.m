function Xt = ssr_iht(Y, A, par, K)

	% Iterative hard thresholding
	% Make sure the operator norm of A is normalized so that ||A||_2<=1
	% Shuai Huang

	tol              = par.tol;
	maxiter       	 = par.maxiter;

	Xt               = par.X0; % Initialize X

	G=A'*A;
	C=A'*Y;

	for iter = 1 : maxiter

        Xt_new = Xt + C - G*Xt;

        [Xt_new_abs_sort, sort_idx] = sort(abs(Xt_new), 'descend');
        Xt_new(sort_idx((K+1):end)) = 0;

        cvg_val = norm(Xt_new - Xt) / norm(Xt_new);

        %fprintf('Iter %d:\t%d\n', iter, cvg_val)

		if (cvg_val < tol)
			break;
		end

        Xt = Xt_new;
		
	end

end

