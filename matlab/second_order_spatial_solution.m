function second_order_spatial_solution()
    % Second-order spatial method with piecewise linear reconstruction
    
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
    
    % Time stepping
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
        
        % Calculate slopes using minmod limiter
        W_slope = zeros(3, N);
        for i = 2:N-1
            for k = 1:3
                slope_left = (W(k,i) - W(k,i-1)) / dx;
                slope_right = (W(k,i+1) - W(k,i)) / dx;
                W_slope(k,i) = minmod(slope_left, slope_right);
            end
        end
        
        % Second-order spatial reconstruction
        W_new = W;
        for i = 2:N-1
            % Left and right states at interface i+1/2
            W_left = W(:,i) + W_slope(:,i) * dx/2;
            W_right = W(:,i+1) - W_slope(:,i+1) * dx/2;
            
            % Local Lax-Friedrichs flux
            F_left = euler_flux(W_left, gamma);
            F_right = euler_flux(W_right, gamma);
            alpha = max(wave_speed(W_left, gamma), wave_speed(W_right, gamma));
            F_half = 0.5*(F_left + F_right) - 0.5*alpha*(W_right - W_left);
            
            % Left interface i-1/2
            W_left_prev = W(:,i-1) + W_slope(:,i-1) * dx/2;
            W_right_prev = W(:,i) - W_slope(:,i) * dx/2;
            F_left_prev = euler_flux(W_left_prev, gamma);
            F_right_prev = euler_flux(W_right_prev, gamma);
            alpha_prev = max(wave_speed(W_left_prev, gamma), wave_speed(W_right_prev, gamma));
            F_half_prev = 0.5*(F_left_prev + F_right_prev) - 0.5*alpha_prev*(W_right_prev - W_left_prev);
            
            % Update
            W_new(:,i) = W(:,i) - (dt/dx)*(F_half - F_half_prev);
        end
        
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
    title('Second-Order Spatial Density Distribution');
    legend('2nd Order Spatial', 'Location', 'best');
    grid on;
    matlab2tikz('../figure/density_spatial2.tex');
    
    % Velocity plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, u, 'r-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Velocity (u)');
    title('Second-Order Spatial Velocity Distribution');
    legend('2nd Order Spatial', 'Location', 'best');
    grid on;
    matlab2tikz('../figure/velocity_spatial2.tex');
    
    % Pressure plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, p, 'g-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Pressure (p)');
    title('Second-Order Spatial Pressure Distribution');
    legend('2nd Order Spatial', 'Location', 'best');
    grid on;
    matlab2tikz('../figure/pressure_spatial2.tex');
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