function run_all()
    % Run all numerical methods for Sod's shock tube problem
    
    fprintf('=== Running All Numerical Methods ===\n\n');
    
    % 1. Exact solution
    fprintf('1. Computing exact solution...\n');
    try
        riemann_exact_solution();
        fprintf('   ✓ Exact solution completed successfully\n\n');
    catch ME
        fprintf('   ✗ Error in exact solution: %s\n\n', ME.message);
    end
    
    % 2. Lax-Friedrichs
    fprintf('2. Computing Lax-Friedrichs solution...\n');
    try
        lax_friedrichs_solution();
        fprintf('   ✓ Lax-Friedrichs completed successfully\n\n');
    catch ME
        fprintf('   ✗ Error in Lax-Friedrichs: %s\n\n', ME.message);
    end
    
    % 3. Local Lax-Friedrichs
    fprintf('3. Computing Local Lax-Friedrichs solution...\n');
    try
        local_lax_friedrichs_solution();
        fprintf('   ✓ Local Lax-Friedrichs completed successfully\n\n');
    catch ME
        fprintf('   ✗ Error in Local Lax-Friedrichs: %s\n\n', ME.message);
    end
    
    % 4. Second-order spatial
    fprintf('4. Computing second-order spatial solution...\n');
    try
        second_order_spatial_solution();
        fprintf('   ✓ Second-order spatial completed successfully\n\n');
    catch ME
        fprintf('   ✗ Error in second-order spatial: %s\n\n', ME.message);
    end
    
    % 5. Second-order temporal
    fprintf('5. Computing second-order temporal solution...\n');
    try
        second_order_temporal_solution();
        fprintf('   ✓ Second-order temporal completed successfully\n\n');
    catch ME
        fprintf('   ✗ Error in second-order temporal: %s\n\n', ME.message);
    end
    
    % 6. Combined second-order
    fprintf('6. Computing combined second-order solution...\n');
    try
        second_order_combined_solution();
        fprintf('   ✓ Combined second-order completed successfully\n\n');
    catch ME
        fprintf('   ✗ Error in combined second-order: %s\n\n', ME.message);
    end
    
    % 7. Comparison of all methods
    fprintf('7. Comparing all methods...\n');
    try
        compare_all_methods();
        fprintf('   ✓ Comparison completed successfully\n\n');
    catch ME
        fprintf('   ✗ Error in comparison: %s\n\n', ME.message);
    end
    
    fprintf('=== All computations completed! ===\n');
    fprintf('Check the ../figure/ directory for TikZ output files.\n');
end