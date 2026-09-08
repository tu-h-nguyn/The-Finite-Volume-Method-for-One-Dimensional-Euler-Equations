function second_order_combined_solution()
    % Combined second-order method (spatial + temporal)
    
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
    
    % Time stepping with Heun's method + 2nd order spatial
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
        K1 = calculate_rhs_spatial2(W, dx, gamma);
        
        % Step 2: Predictor step
        W_star = W + dt * K1;
        
        % Apply boundary conditions to predictor
        W_star(:,1) = W_star(:,2);
        W_star(:,N) = W_star(:,N-1);
        
        % Step 3: Calculate K2 (slope at predicted point)
        K2 = calculate_rhs_spatial2(W_star, dx, gamma);
        
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
    title('Combined Second-Order Density Distribution');
    legend('2nd Order Combined', 'Location', 'best');
    grid on;
    matlab2tikz('../figure/density_combined.tex');
    
    % Velocity plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, u, 'r-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Velocity (u)');
    title('Combined Second-Order Velocity Distribution');
    legend('2nd Order Combined', 'Location', 'best');
    grid on;
    matlab2tikz('../figure/velocity_combined.tex');
    
    % Pressure plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, p, 'g-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Pressure (p)');
    title('Combined Second-Order Pressure Distribution');
    legend('2nd Order Combined', 'Location', 'best');
    grid on;
    matlab2tikz('../figure/pressure_combined.tex');
end

function rhs = calculate_rhs_spatial2(W, dx, gamma)
    % Calculate RHS with second-order spatial reconstruction
    N = size(W, 2);
    rhs = zeros(3, N);
    
    % Calculate slopes using minmod limiter
    W_slope = zeros(3, N);
    for i = 2:N-1
        for k = 1:3
            slope_left = (W(k,i) - W(k,i-1)) / dx;
            slope_right = (W(k,i+1) - W(k,i)) / dx;
            W_slope(k,i) = minmod(slope_left, slope_right);
        end
    end
    
    for i = 2:N-1
        % Reconstruct at interfaces
        W_left = W(:,i) + W_slope(:,i) * dx/2;
        W_right = W(:,i+1) - W_slope(:,i+1) * dx/2;
        
        F_left = euler_flux(W_left, gamma);
        F_right = euler_flux(W_right, gamma);
        alpha_right = max(wave_speed(W_left, gamma), wave_speed(W_right, gamma));
        F_right_interface = 0.5*(F_left + F_right) - 0.5*alpha_right*(W_right - W_left);
        
        W_left_prev = W(:,i-1) + W_slope(:,i-1) * dx/2;
        W_right_prev = W(:,i) - W_slope(:,i) * dx/2;
        
        F_left_prev = euler_flux(W_left_prev, gamma);
        F_right_prev = euler_flux(W_right_prev, gamma);
        alpha_left = max(wave_speed(W_left_prev, gamma), wave_speed(W_right_prev, gamma));
        F_left_interface = 0.5*(F_left_prev + F_right_prev) - 0.5*alpha_left*(W_right_prev - W_left_prev);
        
        rhs(:,i) = -(F_right_interface - F_left_interface) / dx;
    end
end

function result = minmod(a, b)
    if a*b > 0
        if abs(a) <= abs(b)
            result = a;
        else
            result = b;
        end
    else
        result = 0;
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