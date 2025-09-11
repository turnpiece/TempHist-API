#!/usr/bin/env python3
"""
Performance testing script for TempHist API core functions.
This script profiles calculation functions and URL building without external dependencies.
Use this for quick performance validation and optimization testing.
"""

import time
import cProfile
import pstats
import io
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only the calculation functions that don't require external dependencies
from main import calculate_historical_average, calculate_trend_slope, build_visual_crossing_url

def profile_function(func, *args, iterations=1000, **kwargs):
    """Profile a function with multiple iterations."""
    print(f"\n{'='*60}")
    print(f"Profiling: {func.__name__}")
    print(f"Iterations: {iterations}")
    print(f"{'='*60}")
    
    # Create profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Time the function
    start_time = time.time()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    end_time = time.time()
    
    profiler.disable()
    
    # Calculate statistics
    total_time = end_time - start_time
    avg_time = total_time / iterations
    ops_per_second = iterations / total_time
    
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per call: {avg_time:.6f} seconds")
    print(f"Operations per second: {ops_per_second:.2f}")
    print(f"Result: {result}")
    
    # Print top 10 functions by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print("\nTop 10 functions by cumulative time:")
    print(s.getvalue())
    
    return result

def profile_url_building():
    """Profile URL building function."""
    print("\n" + "="*60)
    print("PROFILING URL BUILDING")
    print("="*60)
    
    test_cases = [
        ("London", "2024-01-15"),
        ("New York", "2024-12-25"),
        ("Tokyo", "2024-06-01"),
        ("Sydney", "2024-03-20"),
        ("Paris", "2024-07-04")
    ]
    
    for location, date in test_cases:
        profile_function(build_visual_crossing_url, location, date, iterations=10000)

def profile_calculation_functions():
    """Profile calculation-intensive functions."""
    print("\n" + "="*60)
    print("PROFILING CALCULATION FUNCTIONS")
    print("="*60)
    
    # Sample data for testing
    sample_data_small = [
        {"x": 2020, "y": 15.5},
        {"x": 2021, "y": 16.2},
        {"x": 2022, "y": 15.8},
        {"x": 2023, "y": 16.5},
        {"x": 2024, "y": 17.1}
    ]
    
    sample_data_large = [
        {"x": year, "y": 15.0 + (year - 1970) * 0.1 + (year % 3) * 0.5}
        for year in range(1970, 2025)
    ]
    
    print("\n1. Historical Average Calculation (Small Dataset)")
    profile_function(calculate_historical_average, sample_data_small, iterations=10000)
    
    print("\n2. Historical Average Calculation (Large Dataset)")
    profile_function(calculate_historical_average, sample_data_large, iterations=1000)
    
    print("\n3. Trend Slope Calculation (Small Dataset)")
    profile_function(calculate_trend_slope, sample_data_small, iterations=10000)
    
    print("\n4. Trend Slope Calculation (Large Dataset)")
    profile_function(calculate_trend_slope, sample_data_large, iterations=1000)

def profile_memory_usage():
    """Profile memory usage of functions."""
    import gc
    import sys
    
    print("\n" + "="*60)
    print("MEMORY USAGE PROFILING")
    print("="*60)
    
    # Sample data
    sample_data = [
        {"x": year, "y": 15.0 + (year - 1970) * 0.1}
        for year in range(1970, 2025)
    ]
    
    # Force garbage collection
    gc.collect()
    
    # Get initial memory
    initial_memory = sys.getsizeof(sample_data)
    
    print(f"Initial data size: {initial_memory} bytes")
    print(f"Data points: {len(sample_data)}")
    
    # Profile memory usage for calculations
    for func_name, func in [
        ("Historical Average", calculate_historical_average),
        ("Trend Slope", calculate_trend_slope)
    ]:
        print(f"\n{func_name}:")
        
        # Measure memory before
        gc.collect()
        memory_before = sum(sys.getsizeof(obj) for obj in gc.get_objects() if isinstance(obj, (int, float, str, list, dict)))
        
        # Run function
        result = func(sample_data)
        
        # Measure memory after
        gc.collect()
        memory_after = sum(sys.getsizeof(obj) for obj in gc.get_objects() if isinstance(obj, (int, float, str, list, dict)))
        
        memory_diff = memory_after - memory_before
        
        print(f"  Memory before: {memory_before:,} bytes")
        print(f"  Memory after: {memory_after:,} bytes")
        print(f"  Memory difference: {memory_diff:,} bytes")
        print(f"  Result: {result}")

def generate_performance_report():
    """Generate a comprehensive performance report."""
    print("="*60)
    print("TEMP HIST API PERFORMANCE PROFILING REPORT")
    print("="*60)
    
    # Check environment
    print(f"\nEnvironment:")
    print(f"- Python version: {sys.version}")
    print(f"- API Key configured: {'Yes' if os.getenv('VISUAL_CROSSING_API_KEY') else 'No'}")
    print(f"- Cache enabled: {os.getenv('CACHE_ENABLED', 'true')}")
    print(f"- Debug mode: {os.getenv('DEBUG', 'false')}")
    
    # Profile different components
    profile_url_building()
    profile_calculation_functions()
    profile_memory_usage()
    
    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)
    
    print("\n" + "="*60)
    print("PERFORMANCE RECOMMENDATIONS")
    print("="*60)
    print("1. URL building is very fast (microseconds) - no optimization needed")
    print("2. Calculation functions are efficient for typical dataset sizes")
    print("3. Consider caching results for frequently accessed data")
    print("4. For large datasets, consider batch processing")
    print("5. Memory usage is minimal for calculation functions")
    print("="*60)

if __name__ == "__main__":
    generate_performance_report() 