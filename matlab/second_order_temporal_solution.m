function second_order_temporal_solution()
    % Second-order temporal method using Heun's method
    
    % Initial conditions
    rho_L = 1.0; u_L = 0.0; p_L = 1.0;
    rho_R = 0.125; u_R = 0.0; p_R = 0.1;
    gamma = 1.4;
    
    % Domain and time
    N = 200;
    x = linspace(0, 1, N);
    dx = x(2) - x(1);
    t_final = 0.2;
    CFL = 0.5;
    
    % Initialize solution
    W = zeros(3, N);
    for i = 1:N
        if x(i) < 0.5
            W(1,i) = rho_L;
            W(2,i) = rho_L * u_L;
            W(3,i) = p_L/(gamma-1) + 0.5*rho_L*u_L^2;
        else
            W(1,i) = rho_R;
            W(2,i) = rho_R * u_R;
            W(3,i) = p_R/(gamma-1) + 0.5*rho_R*u_R^2;
        end
    end
    
    % Time stepping with Heun's method
    t = 0;
    while t < t_final
        % Calculate time step
        max_speed = 0;
        for i = 1:N
            rho = W(1,i);
            u = W(2,i)/rho;
            E = W(3,i);
            p = (gamma-1)*(E - 0.5*rho*u^2);
            c = sqrt(gamma*p/rho);
            max_speed = max(max_speed, abs(u) + c);
        end
        dt = CFL * dx / max_speed;
        if t + dt > t_final
            dt = t_final - t;
        end
        
        % Step 1: Calculate K1 (initial slope)
        K1 = calculate_rhs(W, dx, gamma);
        
        % Step 2: Predictor step
        W_star = W + dt * K1;
        
        % Apply boundary conditions to predictor
        W_star(:,1) = W_star(:,2);
        W_star(:,N) = W_star(:,N-1);
        
        % Step 3: Calculate K2 (slope at predicted point)
        K2 = calculate_rhs(W_star, dx, gamma);
        
        % Step 4: Corrector step (Heun's method)
        W_new = W + (dt/2) * (K1 + K2);
        
        % Boundary conditions
        W_new(:,1) = W_new(:,2);
        W_new(:,N) = W_new(:,N-1);
        
        W = W_new;
        t = t + dt;
    end
    
    % Extract primitive variables
    rho = W(1,:);
    u = W(2,:) ./ W(1,:);
    E = W(3,:);
    p = (gamma-1)*(E - 0.5*W(1,:).*u.^2);
    
    % Save individual plots as TikZ
    % Density plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, rho, 'b-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Density (\rho)');
    title('Second-Order Temporal Density Distribution');
    legend('2nd Order Temporal', 'Location', 'best');
    grid on;
    matlab2tikz('../figure/density_temporal2.tex');
    
    % Velocity plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, u, 'r-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Velocity (u)');
    title('Second-Order Temporal Velocity Distribution');
    legend('2nd Order Temporal', 'Location', 'best');
    grid on;
    matlab2tikz('../figure/velocity_temporal2.tex');
    
    % Pressure plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, p, 'g-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Pressure (p)');
    title('Second-Order Temporal Pressure Distribution');
    legend('2nd Order Temporal', 'Location', 'best');
    grid on;
    matlab2tikz('../figure/pressure_temporal2.tex');
end

function rhs = calculate_rhs(W, dx, gamma)
    % Calculate right-hand side for ODE system using Local Lax-Friedrichs
    N = size(W, 2);
    rhs = zeros(3, N);
    
    for i = 2:N-1
        % Calculate fluxes
        F_im1 = euler_flux(W(:,i-1), gamma);
        F_i = euler_flux(W(:,i), gamma);
        F_ip1 = euler_flux(W(:,i+1), gamma);
        
        % Local wave speeds
        alpha_left = max(wave_speed(W(:,i-1), gamma), wave_speed(W(:,i), gamma));
        alpha_right = max(wave_speed(W(:,i), gamma), wave_speed(W(:,i+1), gamma));
        
        % LLF fluxes
        F_left = 0.5*(F_im1 + F_i) - 0.5*alpha_left*(W(:,i) - W(:,i-1));
        F_right = 0.5*(F_i + F_ip1) - 0.5*alpha_right*(W(:,i+1) - W(:,i));
        
        % RHS calculation
        rhs(:,i) = -(F_right - F_left) / dx;
    end
end

function F = euler_flux(W, gamma)
    rho = W(1);
    u = W(2)/rho;
    E = W(3);
    p = (gamma-1)*(E - 0.5*rho*u^2);
    
    F = [rho*u; rho*u^2 + p; u*(E + p)];
end

function alpha = wave_speed(W, gamma)
    rho = W(1);
    u = W(2)/rho;
    E = W(3);
    p = (gamma-1)*(E - 0.5*rho*u^2);
    c = sqrt(gamma*p/rho);
    alpha = abs(u) + c;
end