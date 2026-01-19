function kl_terms_matrix = generate_kl_random_field(Lx, Ly, lambdax, lambday, MeanY, VarY, kl_num, Ne)
    % Function to generate Ne sets of KL-expanded random field coefficients
    % and return the kl_term matrix of size [kl_num, Ne]

    % Discretization
    dx = 1;
    dy = 1;
    n = Ly/dy + 1;   % row number
    m = Lx/dx + 1;   % column number

    % Initialize correlation matrix and coordinates
    C = nan(m*n, m*n);
    x = nan(m*n, 2);

    for i = 1:m*n
        a = rem(i, m);
        b = fix(i / m);
        if a == 0
            x(i, 1) = (m - 1) * dx;
            x(i, 2) = (b - 1) * dy;
        else
            x(i, 1) = (a - 1) * dx;
            x(i, 2) = b * dy;
        end
    end

    % Compute correlation matrix
    for i = 1:m*n
        for j = 1:m*n
            dx_ = (x(i,1) - x(j,1)) / lambdax;
            dy_ = (x(i,2) - x(j,2)) / lambday;
            C(i,j) = VarY * exp(-sqrt(dx_^2 + dy_^2));
        end
    end

    % Compute eigenvalues and eigenvectors
    [A1, B1] = eigs(C, kl_num);

    % Compute and store Ne samples of KL terms
    kl_terms_matrix = randn(kl_num, Ne); % [kl_num × Ne]
    
    % Optionally, compute KL-expansion-based fields here
    % Y_samples = MeanY + A1 * sqrt(B1) * kl_terms_matrix; % [m*n × Ne]

    % Save output matrix and transformation kernel
    save('kl_terms_all.mat', 'kl_terms_matrix');  % Save all KL coefficients
    fun = A1 * sqrt(B1);
    save('fun.mat', 'fun');                       % Save the basis transformation

end
