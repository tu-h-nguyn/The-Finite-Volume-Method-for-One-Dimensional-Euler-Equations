function result = minmod(a, b)
    % Hàm minmod cho hai tham số
    % Input: a, b - hai giá trị
    % Output: result - kết quả minmod
    
    if a * b > 0
        if abs(a) <= abs(b)
            result = a;
        else
            result = b;
        end
    else
        result = 0;
    end
end