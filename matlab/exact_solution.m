function [rho_exact, u_exact, p_exact] = exact_solution(x, t, gamma)
    % Nghiệm chính xác cho bài toán Sod shock tube
    % Input: x - vector vị trí
    %        t - thời gian
    %        gamma - hệ số nhiệt dung riêng
    % Output: rho_exact, u_exact, p_exact - nghiệm chính xác
    
    % Điều kiện đầu
    rho_L = 1.0; u_L = 0.0; p_L = 1.0;
    rho_R = 0.125; u_R = 0.0; p_R = 0.1;
    
    % Tính các tham số
    c_L = sqrt(gamma * p_L / rho_L);  % vận tốc âm thanh bên trái
    c_R = sqrt(gamma * p_R / rho_R);  % vận tốc âm thanh bên phải
    
    % Giải bài toán Riemann để tìm các vận tốc sóng
    % Sử dụng phương pháp lặp Newton-Raphson đơn giản
    p_star = 0.3;  % ước lượng ban đầu
    for iter = 1:10
        % Tính các hàm f_L và f_R
        if p_star > p_L
            % Shock wave bên trái
            A_L = 2 / ((gamma + 1) * rho_L);
            B_L = (gamma - 1) / (gamma + 1) * p_L;
            f_L = (p_star - p_L) * sqrt(A_L / (p_star + B_L));
            df_L = sqrt(A_L / (p_star + B_L)) * (1 - (p_star - p_L) / (2 * (p_star + B_L)));
        else
            % Rarefaction wave bên trái
            f_L = 2 * c_L / (gamma - 1) * ((p_star / p_L)^((gamma - 1) / (2 * gamma)) - 1);
            df_L = (c_L / gamma) * (p_star / p_L)^(-(gamma + 1) / (2 * gamma)) / p_L;
        end
        
        if p_star > p_R
            % Shock wave bên phải
            A_R = 2 / ((gamma + 1) * rho_R);
            B_R = (gamma - 1) / (gamma + 1) * p_R;
            f_R = (p_star - p_R) * sqrt(A_R / (p_star + B_R));
            df_R = sqrt(A_R / (p_star + B_R)) * (1 - (p_star - p_R) / (2 * (p_star + B_R)));
        else
            % Rarefaction wave bên phải
            f_R = 2 * c_R / (gamma - 1) * ((p_star / p_R)^((gamma - 1) / (2 * gamma)) - 1);
            df_R = (c_R / gamma) * (p_star / p_R)^(-(gamma + 1) / (2 * gamma)) / p_R;
        end
        
        % Newton-Raphson update
        F = f_L + f_R + (u_R - u_L);
        dF = df_L + df_R;
        p_star = p_star - F / dF;
        
        if abs(F) < 1e-6
            break;
        end
    end
    
    % Tính vận tốc trong vùng star
    u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L);
    
    % Khởi tạo output
    N = length(x);
    rho_exact = zeros(1, N);
    u_exact = zeros(1, N);
    p_exact = zeros(1, N);
    
    % Tính nghiệm tại mỗi điểm
    for i = 1:N
        xi = (x(i) - 0.5) / t;  % tọa độ tự tương tự
        
        if xi <= u_star
            % Bên trái contact discontinuity
            if p_star > p_L
                % Left shock
                S_L = u_L - c_L * sqrt((gamma + 1) / (2 * gamma) * p_star / p_L + (gamma - 1) / (2 * gamma));
                if xi <= S_L
                    % Vùng 1 (trạng thái ban đầu bên trái)
                    rho_exact(i) = rho_L;
                    u_exact(i) = u_L;
                    p_exact(i) = p_L;
                else
                    % Vùng 2 (sau shock bên trái)
                    rho_exact(i) = rho_L * (p_star / p_L + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * p_star / p_L + 1);
                    u_exact(i) = u_star;
                    p_exact(i) = p_star;
                end
            else
                % Left rarefaction
                S_HL = u_L - c_L;  % head của rarefaction
                c_star_L = c_L * (p_star / p_L)^((gamma - 1) / (2 * gamma));
                S_TL = u_star - c_star_L;  % tail của rarefaction
                
                if xi <= S_HL
                    % Vùng 1
                    rho_exact(i) = rho_L;
                    u_exact(i) = u_L;
                    p_exact(i) = p_L;
                elseif xi <= S_TL
                    % Trong rarefaction fan
                    u_exact(i) = 2 / (gamma + 1) * (c_L + (gamma - 1) / 2 * u_L + xi);
                    c = 2 / (gamma + 1) * (c_L + (gamma - 1) / 2 * (u_L - xi));
                    rho_exact(i) = rho_L * (c / c_L)^(2 / (gamma - 1));
                    p_exact(i) = p_L * (c / c_L)^(2 * gamma / (gamma - 1));
                else
                    % Vùng 2
                    rho_exact(i) = rho_L * (p_star / p_L)^(1 / gamma);
                    u_exact(i) = u_star;
                    p_exact(i) = p_star;
                end
            end
        else
            % Bên phải contact discontinuity
            if p_star > p_R
                % Right shock
                S_R = u_R + c_R * sqrt((gamma + 1) / (2 * gamma) * p_star / p_R + (gamma - 1) / (2 * gamma));
                if xi >= S_R
                    % Vùng 4 (trạng thái ban đầu bên phải)
                    rho_exact(i) = rho_R;
                    u_exact(i) = u_R;
                    p_exact(i) = p_R;
                else
                    % Vùng 3 (sau shock bên phải)
                    rho_exact(i) = rho_R * (p_star / p_R + (gamma - 1) / (gamma + 1)) / ((gamma - 1) / (gamma + 1) * p_star / p_R + 1);
                    u_exact(i) = u_star;
                    p_exact(i) = p_star;
                end
            else
                % Right rarefaction
                S_HR = u_R + c_R;  % head của rarefaction
                c_star_R = c_R * (p_star / p_R)^((gamma - 1) / (2 * gamma));
                S_TR = u_star + c_star_R;  % tail của rarefaction
                
                if xi >= S_HR
                    % Vùng 4
                    rho_exact(i) = rho_R;
                    u_exact(i) = u_R;
                    p_exact(i) = p_R;
                elseif xi >= S_TR
                    % Trong rarefaction fan
                    u_exact(i) = 2 / (gamma + 1) * (-c_R + (gamma - 1) / 2 * u_R + xi);
                    c = 2 / (gamma + 1) * (c_R - (gamma - 1) / 2 * (u_R - xi));
                    rho_exact(i) = rho_R * (c / c_R)^(2 / (gamma - 1));
                    p_exact(i) = p_R * (c / c_R)^(2 * gamma / (gamma - 1));
                else
                    % Vùng 3
                    rho_exact(i) = rho_R * (p_star / p_R)^(1 / gamma);
                    u_exact(i) = u_star;
                    p_exact(i) = p_star;
                end
            end
        end
    end
end