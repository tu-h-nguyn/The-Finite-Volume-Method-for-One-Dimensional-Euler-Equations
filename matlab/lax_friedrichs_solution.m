function lax_friedrichs_solution()
    % Lax-Friedrichs method for Sod's shock tube problem
    
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
        
        % Lax-Friedrichs update
        W_new = zeros(3, N);
        for i = 2:N-1
            % Calculate fluxes
            F_left = euler_flux(W(:,i-1), gamma);
            F_right = euler_flux(W(:,i+1), gamma);
            
            % Lax-Friedrichs scheme
            W_new(:,i) = 0.5*(W(:,i-1) + W(:,i+1)) - (dt/(2*dx))*(F_right - F_left);
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
    title('Lax-Friedrichs Density Distribution');
    legend('LF Solution', 'Location', 'best');
    grid on;
    matlab2tikz('../report/figure/density_lf.tex');
    
    % Velocity plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, u, 'r-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Velocity (u)');
    title('Lax-Friedrichs Velocity Distribution');
    legend('LF Solution', 'Location', 'best');
    grid on;
    matlab2tikz('../report/figure/velocity_lf.tex');
    
    % Pressure plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, p, 'g-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Pressure (p)');
    title('Lax-Friedrichs Pressure Distribution');
    legend('LF Solution', 'Location', 'best');
    grid on;
    matlab2tikz('../report/figure/pressure_lf.tex');
end

function F = euler_flux(W, gamma)
    rho = W(1);
    u = W(2)/rho;
    E = W(3);
    p = (gamma-1)*(E - 0.5*rho*u^2);
    
    F = [rho*u; rho*u^2 + p; u*(E + p)];
end