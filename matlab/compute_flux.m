function f = compute_flux(W, gamma)
    % Tính vector thông lượng f(W) cho phương trình Euler 1D
    % Input: W - vector biến bảo toàn [rho; rho*u; E]
    %        gamma - hệ số nhiệt dung riêng
    % Output: f - vector thông lượng [rho*u; rho*u^2+p; (E+p)*u]
    
    rho = W(1);
    rho_u = W(2);
    E = W(3);
    
    u = rho_u / rho;
    p = (gamma - 1) * (E - 0.5 * rho * u^2);
    
    f = zeros(3, 1);
    f(1) = rho_u;              % rho * u
    f(2) = rho_u * u + p;      % rho * u^2 + p
    f(3) = (E + p) * u;        % (E + p) * u
end