function compare_all_methods()
    % Compare all numerical methods with exact solution
    
    % Common parameters
    rho_L = 1.0; u_L = 0.0; p_L = 1.0;
    rho_R = 0.125; u_R = 0.0; p_R = 0.1;
    gamma = 1.4;
    N = 200;
    x = linspace(0, 1, N);
    t_final = 0.2;
    
    % Get all solutions
    fprintf('Computing exact solution...\n');
    [rho_exact, u_exact, p_exact] = solve_exact(x, t_final, gamma);
    
    fprintf('Computing Lax-Friedrichs solution...\n');
    [rho_lf, u_lf, p_lf] = solve_lax_friedrichs(x, t_final, gamma);
    
    fprintf('Computing Local Lax-Friedrichs solution...\n');
    [rho_llf, u_llf, p_llf] = solve_local_lax_friedrichs(x, t_final, gamma);
    
    fprintf('Computing 2nd order spatial solution...\n');
    [rho_spatial2, u_spatial2, p_spatial2] = solve_spatial2(x, t_final, gamma);
    
    fprintf('Computing 2nd order temporal solution...\n');
    [rho_temporal2, u_temporal2, p_temporal2] = solve_temporal2(x, t_final, gamma);
    
    fprintf('Computing combined 2nd order solution...\n');
    [rho_combined, u_combined, p_combined] = solve_combined(x, t_final, gamma);
    
    % Create comparison plots
    figure('Position', [100, 100, 1200, 400]);
    set(gcf, 'Color', 'white');
    
    % Density comparison
    subplot(1,3,1);
    plot(x, rho_exact, 'k-', 'LineWidth', 3, 'DisplayName', 'Exact');
    hold on;
    plot(x, rho_lf, 'r:', 'LineWidth', 2, 'DisplayName', 'Lax-Friedrichs');
    plot(x, rho_llf, 'b:', 'LineWidth', 2, 'DisplayName', 'Local LF');
    plot(x, rho_spatial2, 'g:', 'LineWidth', 2, 'DisplayName', '2nd Spatial');
    plot(x, rho_temporal2, 'm:', 'LineWidth', 2, 'DisplayName', '2nd Temporal');
    plot(x, rho_combined, 'c:', 'LineWidth', 2, 'DisplayName', '2nd Combined');
    xlabel('x');
    ylabel('Density (\rho)');
    title('Density Comparison');
    legend('Location', 'best');
    grid on;
    
    % Velocity comparison
    subplot(1,3,2);
    plot(x, u_exact, 'k-', 'LineWidth', 3, 'DisplayName', 'Exact');
    hold on;
    plot(x, u_lf, 'r:', 'LineWidth', 2, 'DisplayName', 'Lax-Friedrichs');
    plot(x, u_llf, 'b:', 'LineWidth', 2, 'DisplayName', 'Local LF');
    plot(x, u_spatial2, 'g:', 'LineWidth', 2, 'DisplayName', '2nd Spatial');
    plot(x, u_temporal2, 'm:', 'LineWidth', 2, 'DisplayName', '2nd Temporal');
    plot(x, u_combined, 'c:', 'LineWidth', 2, 'DisplayName', '2nd Combined');
    xlabel('x');
    ylabel('Velocity (u)');
    title('Velocity Comparison');
    legend('Location', 'best');
    grid on;
    
    % Pressure comparison
    subplot(1,3,3);
    plot(x, p_exact, 'k-', 'LineWidth', 3, 'DisplayName', 'Exact');
    hold on;
    plot(x, p_lf, 'r:', 'LineWidth', 2, 'DisplayName', 'Lax-Friedrichs');
    plot(x, p_llf, 'b:', 'LineWidth', 2, 'DisplayName', 'Local LF');
    plot(x, p_spatial2, 'g:', 'LineWidth', 2, 'DisplayName', '2nd Spatial');
    plot(x, p_temporal2, 'm:', 'LineWidth', 2, 'DisplayName', '2nd Temporal');
    plot(x, p_combined, 'c:', 'LineWidth', 2, 'DisplayName', '2nd Combined');
    xlabel('x');
    ylabel('Pressure (p)');
    title('Pressure Comparison');
    legend('Location', 'best');
    grid on;
    
    sgtitle('Comparison of Numerical Methods at t = 0.2');
    
    % Save as TikZ
    matlab2tikz('../report/figure/methods_comparison.tex');
    fprintf('Comparison complete. Results saved to methods_comparison.tex\n');
end

function [rho, u, p] = solve_exact(x, t, gamma)
    rho_L = 1.0; u_L = 0.0; p_L = 1.0;
    rho_R = 0.125; u_R = 0.0; p_R = 0.1;
    x0 = 0.5;
    
    a_L = sqrt(gamma * p_L / rho_L);
    p_star = solve_star_pressure(p_L, p_R, u_L, u_R, rho_L, rho_R, gamma);
    u_star = u_L + (2*a_L/(gamma-1)) * (1 - (p_star/p_L)^((gamma-1)/(2*gamma)));
    
    rho_L_star = rho_L * (p_star/p_L)^(1/gamma);
    beta = (gamma+1)/(gamma-1);
    rho_R_star = rho_R * (1 + beta*(p_star/p_R)) / (beta + (p_star/p_R));
    
    a_L_star = sqrt(gamma * p_star / rho_L_star);
    S_HL = u_L - a_L;
    S_TL = u_star - a_L_star;
    S_contact = u_star;
    a_R = sqrt(gamma * p_R / rho_R);
    S_shock = u_R + a_R * sqrt((gamma+1)/(2*gamma) * (p_star/p_R) + (gamma-1)/(2*gamma));
    
    N = length(x);
    rho = zeros(1, N); u = zeros(1, N); p = zeros(1, N);
    
    for i = 1:N
        xi = (x(i) - x0) / t;
        if xi < S_HL
            rho(i) = rho_L; u(i) = u_L; p(i) = p_L;
        elseif xi < S_TL
            u_fan = (2/(gamma+1)) * (a_L + xi + (gamma-1)/2 * u_L);
            a_fan = u_fan - xi;
            rho(i) = rho_L * (a_fan/a_L)^(2/(gamma-1));
            u(i) = u_fan;
            p(i) = p_L * (a_fan/a_L)^(2*gamma/(gamma-1));
        elseif xi < S_contact
            rho(i) = rho_L_star; u(i) = u_star; p(i) = p_star;
        elseif xi < S_shock
            rho(i) = rho_R_star; u(i) = u_star; p(i) = p_star;
        else
            rho(i) = rho_R; u(i) = u_R; p(i) = p_R;
        end
    end
end

function [rho, u, p] = solve_lax_friedrichs(x, t_final, gamma)
    [W] = run_method(x, t_final, gamma, @update_lf);
    [rho, u, p] = extract_primitives(W, gamma);
end

function [rho, u, p] = solve_local_lax_friedrichs(x, t_final, gamma)
    [W] = run_method(x, t_final, gamma, @update_llf);
    [rho, u, p] = extract_primitives(W, gamma);
end

function [rho, u, p] = solve_spatial2(x, t_final, gamma)
    [W] = run_method(x, t_final, gamma, @update_spatial2);
    [rho, u, p] = extract_primitives(W, gamma);
end

function [rho, u, p] = solve_temporal2(x, t_final, gamma)
    [W] = run_method_heun(x, t_final, gamma, @update_llf);
    [rho, u, p] = extract_primitives(W, gamma);
end

function [rho, u, p] = solve_combined(x, t_final, gamma)
    [W] = run_method_heun(x, t_final, gamma, @update_spatial2);
    [rho, u, p] = extract_primitives(W, gamma);
end

function W = run_method(x, t_final, gamma, update_func)
    W = initialize_solution(x, gamma);
    N = length(x); dx = x(2) - x(1); t = 0; CFL = 0.5;
    
    while t < t_final
        dt = min(CFL * dx / calculate_max_speed(W, gamma), t_final - t);
        W = update_func(W, dt, dx, gamma);
        W(:,1) = W(:,2); W(:,N) = W(:,N-1);
        t = t + dt;
    end
end

function W = run_method_heun(x, t_final, gamma, rhs_func)
    W = initialize_solution(x, gamma);
    N = length(x); dx = x(2) - x(1); t = 0; CFL = 0.5;
    
    while t < t_final
        dt = min(CFL * dx / calculate_max_speed(W, gamma), t_final - t);
        
        K1 = rhs_func(W, dt, dx, gamma);
        W_star = W + dt * K1;
        W_star(:,1) = W_star(:,2); W_star(:,N) = W_star(:,N-1);
        
        K2 = rhs_func(W_star, dt, dx, gamma);
        W = W + (dt/2) * (K1 + K2);
        W(:,1) = W(:,2); W(:,N) = W(:,N-1);
        
        t = t + dt;
    end
end

function W = initialize_solution(x, gamma)
    rho_L = 1.0; u_L = 0.0; p_L = 1.0;
    rho_R = 0.125; u_R = 0.0; p_R = 0.1;
    N = length(x);
    W = zeros(3, N);
    
    for i = 1:N
        if x(i) < 0.5
            W(1,i) = rho_L; W(2,i) = rho_L * u_L; W(3,i) = p_L/(gamma-1) + 0.5*rho_L*u_L^2;
        else
            W(1,i) = rho_R; W(2,i) = rho_R * u_R; W(3,i) = p_R/(gamma-1) + 0.5*rho_R*u_R^2;
        end
    end
end

function W_new = update_lf(W, dt, dx, gamma)
    N = size(W, 2);
    W_new = zeros(3, N);
    for i = 2:N-1
        F_left = euler_flux(W(:,i-1), gamma);
        F_right = euler_flux(W(:,i+1), gamma);
        W_new(:,i) = 0.5*(W(:,i-1) + W(:,i+1)) - (dt/(2*dx))*(F_right - F_left);
    end
end

function rhs = update_llf(W, dt, dx, gamma)
    N = size(W, 2);
    rhs = zeros(3, N);
    for i = 2:N-1
        F_i = euler_flux(W(:,i), gamma);
        F_ip1 = euler_flux(W(:,i+1), gamma);
        F_im1 = euler_flux(W(:,i-1), gamma);
        
        alpha_right = max(wave_speed(W(:,i), gamma), wave_speed(W(:,i+1), gamma));
        alpha_left = max(wave_speed(W(:,i-1), gamma), wave_speed(W(:,i), gamma));
        
        F_right = 0.5*(F_i + F_ip1) - 0.5*alpha_right*(W(:,i+1) - W(:,i));
        F_left = 0.5*(F_im1 + F_i) - 0.5*alpha_left*(W(:,i) - W(:,i-1));
        
        rhs(:,i) = -(F_right - F_left) / dx;
    end
end

function rhs = update_spatial2(W, dt, dx, gamma)
    N = size(W, 2);
    rhs = zeros(3, N);
    
    % Calculate slopes with minmod
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

% Helper functions
function result = minmod(a, b)
    if a*b > 0
        result = sign(a) * min(abs(a), abs(b));
    else
        result = 0;
    end
end

function F = euler_flux(W, gamma)
    rho = W(1); u = W(2)/rho; E = W(3);
    p = (gamma-1)*(E - 0.5*rho*u^2);
    F = [rho*u; rho*u^2 + p; u*(E + p)];
end

function alpha = wave_speed(W, gamma)
    rho = W(1); u = W(2)/rho; E = W(3);
    p = (gamma-1)*(E - 0.5*rho*u^2);
    c = sqrt(gamma*p/rho);
    alpha = abs(u) + c;
end

function max_speed = calculate_max_speed(W, gamma)
    N = size(W, 2);
    max_speed = 0;
    for i = 1:N
        max_speed = max(max_speed, wave_speed(W(:,i), gamma));
    end
end

function [rho, u, p] = extract_primitives(W, gamma)
    rho = W(1,:);
    u = W(2,:) ./ W(1,:);
    E = W(3,:);
    p = (gamma-1)*(E - 0.5*W(1,:).*u.^2);
end

function p_star = solve_star_pressure(p_L, p_R, u_L, u_R, rho_L, rho_R, gamma)
    a_L = sqrt(gamma * p_L / rho_L);
    p = (p_L + p_R) / 2;
    A_R = 2 / ((gamma + 1) * rho_R);
    B_R = (gamma - 1) / (gamma + 1) * p_R;
    
    for iter = 1:100
        u_L_star = u_L + (2*a_L/(gamma-1)) * (1 - (p/p_L)^((gamma-1)/(2*gamma)));
        u_R_star = u_R + (p - p_R) * sqrt(A_R / (p + B_R));
        f = u_L_star - u_R_star;
        
        du_L_dp = -(1/(rho_L * a_L)) * (p/p_L)^(-(gamma+1)/(2*gamma));
        du_R_dp = sqrt(A_R / (p + B_R)) * (1 - (p - p_R)/(2*(p + B_R)));
        f_prime = du_L_dp - du_R_dp;
        
        p_new = p - f / f_prime;
        if abs(p_new - p) / p < 1e-6
            p_star = p_new; return;
        end
        p = p_new;
    end
    p_star = p;
end