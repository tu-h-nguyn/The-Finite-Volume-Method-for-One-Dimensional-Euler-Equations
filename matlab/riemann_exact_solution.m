function riemann_exact_solution()
    % Exact solution for Sod's shock tube problem
    
    % Initial conditions
    rho_L = 1.0; u_L = 0.0; p_L = 1.0;
    rho_R = 0.125; u_R = 0.0; p_R = 0.1;
    gamma = 1.4;
    
    % Domain and time
    x = linspace(0, 1, 1000);
    t = 0.2;
    x0 = 0.5; % Initial membrane position
    
    % Sound speeds
    a_L = sqrt(gamma * p_L / rho_L);
    a_R = sqrt(gamma * p_R / rho_R);
    
    % Solve for star pressure using Newton-Raphson
    p_star = solve_star_pressure(p_L, p_R, u_L, u_R, rho_L, rho_R, gamma);
    
    % Calculate star velocity
    u_star = u_L + (2*a_L/(gamma-1)) * (1 - (p_star/p_L)^((gamma-1)/(2*gamma)));
    
    % Star densities
    rho_L_star = rho_L * (p_star/p_L)^(1/gamma);
    beta = (gamma+1)/(gamma-1);
    rho_R_star = rho_R * (1 + beta*(p_star/p_R)) / (beta + (p_star/p_R));
    
    % Wave speeds
    a_L_star = sqrt(gamma * p_star / rho_L_star);
    S_HL = u_L - a_L;
    S_TL = u_star - a_L_star;
    S_contact = u_star;
    S_shock = u_R + a_R * sqrt((gamma+1)/(2*gamma) * (p_star/p_R) + (gamma-1)/(2*gamma));
    
    % Initialize solution arrays
    rho = zeros(size(x));
    u = zeros(size(x));
    p = zeros(size(x));
    
    % Calculate exact solution
    for i = 1:length(x)
        xi = (x(i) - x0) / t;
        
        if xi < S_HL
            % Left state
            rho(i) = rho_L;
            u(i) = u_L;
            p(i) = p_L;
        elseif xi < S_TL
            % Rarefaction fan
            u_fan = (2/(gamma+1)) * (a_L + xi + (gamma-1)/2 * u_L);
            a_fan = u_fan - xi;
            rho(i) = rho_L * (a_fan/a_L)^(2/(gamma-1));
            u(i) = u_fan;
            p(i) = p_L * (a_fan/a_L)^(2*gamma/(gamma-1));
        elseif xi < S_contact
            % Left star state
            rho(i) = rho_L_star;
            u(i) = u_star;
            p(i) = p_star;
        elseif xi < S_shock
            % Right star state
            rho(i) = rho_R_star;
            u(i) = u_star;
            p(i) = p_star;
        else
            % Right state
            rho(i) = rho_R;
            u(i) = u_R;
            p(i) = p_R;
        end
    end
    
    % Save individual plots as PNG
    % Density plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, rho, 'b-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Density (\rho)');
    title('Exact Density Distribution');
    legend('Exact Solution', 'Location', 'best');
    grid on;
    matlab2tikz('../report/figure/density_exact.tex');
    
    % Velocity plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, u, 'r-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Velocity (u)');
    title('Exact Velocity Distribution');
    legend('Exact Solution', 'Location', 'best');
    grid on;
    matlab2tikz('../report/figure/velocity_exact.tex');
    
    % Pressure plot
    figure('Position', [100, 100, 400, 300]);
    set(gcf, 'Color', 'white');
    plot(x, p, 'g-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Pressure (p)');
    title('Exact Pressure Distribution');
    legend('Exact Solution', 'Location', 'best');
    grid on;
    matlab2tikz('../report/figure/pressure_exact.tex');
    
    % Combined plot with all 4 subplots
    figure('Position', [100, 100, 800, 800]);
    
    subplot(2,2,1);
    plot(x, rho, 'b-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Density (\rho)');
    title('Exact Density Distribution');
    legend('Exact Solution', 'Location', 'best');
    grid on;
    
    subplot(2,2,2);
    plot(x, u, 'r-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Velocity (u)');
    title('Exact Velocity Distribution');
    legend('Exact Solution', 'Location', 'best');
    grid on;
    
    subplot(2,2,3);
    plot(x, p, 'g-', 'LineWidth', 2);
    xlabel('x');
    ylabel('Pressure (p)');
    title('Exact Pressure Distribution');
    legend('Exact Solution', 'Location', 'best');
    grid on;
    
    % Fourth subplot for wave structure diagram
    subplot(2,2,4);
    axis([0 1 0 0.4]);
    hold on;
    
    % Draw wave structure at t=0.3
    x_head = x0 + S_HL * t;
    x_tail = x0 + S_TL * t;
    x_contact = x0 + S_contact * t;
    x_shock = x0 + S_shock * t;
    
    % Plot regions
    fill([0 x_head x_head 0], [0 0 0.4 0.4], [0.8 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    fill([x_head x_tail x_tail x_head], [0 0 0.4 0.4], [0.6 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    fill([x_tail x_contact x_contact x_tail], [0 0 0.4 0.4], [1 0.8 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    fill([x_contact x_shock x_shock x_contact], [0 0 0.4 0.4], [1 0.6 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    fill([x_shock 1 1 x_shock], [0 0 0.4 0.4], [1 0.8 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    
    % Draw wave lines
    plot([x_head x_head], [0 0.4], 'k--', 'LineWidth', 1.5);
    plot([x_tail x_tail], [0 0.4], 'k--', 'LineWidth', 1.5);
    plot([x_contact x_contact], [0 0.4], 'k-', 'LineWidth', 2);
    plot([x_shock x_shock], [0 0.4], 'k-', 'LineWidth', 3);
    
    % Labels
    text(x_head/2, 0.35, 'L', 'FontSize', 12, 'HorizontalAlignment', 'center');
    text((x_head+x_tail)/2, 0.35, 'Fan', 'FontSize', 10, 'HorizontalAlignment', 'center');
    text((x_tail+x_contact)/2, 0.35, 'L*', 'FontSize', 12, 'HorizontalAlignment', 'center');
    text((x_contact+x_shock)/2, 0.35, 'R*', 'FontSize', 12, 'HorizontalAlignment', 'center');
    text((x_shock+1)/2, 0.35, 'R', 'FontSize', 12, 'HorizontalAlignment', 'center');
    
    xlabel('x');
    ylabel('Wave Structure');
    title('Wave Structure at t = 0.3');
    grid on;
    
    sgtitle(sprintf('Exact Riemann Solution at t = %.1f', t));
    
    % Display wave positions
    fprintf('Wave positions at t = %.1f:\n', t);
    fprintf('Head of rarefaction: x = %.3f\n', x0 + S_HL * t);
    fprintf('Tail of rarefaction: x = %.3f\n', x0 + S_TL * t);
    fprintf('Contact discontinuity: x = %.3f\n', x0 + S_contact * t);
    fprintf('Shock wave: x = %.3f\n', x0 + S_shock * t);
end

function p_star = solve_star_pressure(p_L, p_R, u_L, u_R, rho_L, rho_R, gamma)
    % Newton-Raphson iteration for star pressure
    
    % Sound speeds
    a_L = sqrt(gamma * p_L / rho_L);
    a_R = sqrt(gamma * p_R / rho_R);
    
    % Initial guess (average)
    p = (p_L + p_R) / 2;
    
    % Constants for right shock
    A_R = 2 / ((gamma + 1) * rho_R);
    B_R = (gamma - 1) / (gamma + 1) * p_R;
    
    tol = 1e-6;
    max_iter = 100;
    
    for iter = 1:max_iter
        % Left rarefaction velocity
        u_L_star = u_L + (2*a_L/(gamma-1)) * (1 - (p/p_L)^((gamma-1)/(2*gamma)));
        
        % Right shock velocity
        u_R_star = u_R + (p - p_R) * sqrt(A_R / (p + B_R));
        
        % Function and derivative
        f = u_L_star - u_R_star;
        
        % Derivatives
        du_L_dp = -(1/(rho_L * a_L)) * (p/p_L)^(-(gamma+1)/(2*gamma));
        du_R_dp = sqrt(A_R / (p + B_R)) * (1 - (p - p_R)/(2*(p + B_R)));
        
        f_prime = du_L_dp - du_R_dp;
        
        % Newton-Raphson update
        p_new = p - f / f_prime;
        
        % Check convergence
        if abs(p_new - p) / p < tol
            p_star = p_new;
            return;
        end
        
        p = p_new;
    end
    
    error('Newton-Raphson did not converge');
end