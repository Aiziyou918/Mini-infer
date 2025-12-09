"""
Memory Usage Comparison Script

Compare memory usage between configurations with/without memory planning
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        return {}

def format_memory(kb: float) -> str:
    """Format memory size"""
    if kb < 1024:
        return f"{kb:.2f} KB"
    else:
        return f"{kb/1024:.2f} MB"

def print_separator(char='=', length=80):
    """Print separator line"""
    print(char * length)

def compare_memory_usage(no_memory_file: str, with_memory_file: str):
    """Compare memory usage between with/without memory planning"""
    
    print_separator()
    print("Memory Usage Comparison Report")
    print_separator()
    print()
    
    # Load results
    no_memory = load_json(no_memory_file) if Path(no_memory_file).exists() else {}
    with_memory = load_json(with_memory_file) if Path(with_memory_file).exists() else {}
    
    if not no_memory and not with_memory:
        print("[ERROR] No result files found")
        return
    
    # Configuration 1: WITHOUT Memory Planning
    print("Configuration 1: Graph Optimization (WITHOUT Memory Planning)")
    print("-" * 80)
    if no_memory:
        time_ms = no_memory.get('inference_time_ms', 0)
        samples = no_memory.get('total_samples', 0)
        print(f"  Inference time: {time_ms} ms")
        print(f"  Samples:        {samples}")
        print(f"  Memory:         N/A (no memory planning)")
    else:
        print("  [NOT AVAILABLE]")
    print()
    
    # Configuration 2: WITH Memory Planning
    print("Configuration 2: Graph Optimization + Memory Planning")
    print("-" * 80)
    if with_memory:
        time_ms = with_memory.get('inference_time_ms', 0)
        samples = with_memory.get('total_samples', 0)
        memory_planning = with_memory.get('memory_planning_enabled', False)
        
        print(f"  Inference time: {time_ms} ms")
        print(f"  Samples:        {samples}")
        
        if memory_planning and 'memory_stats' in with_memory:
            mem_stats = with_memory['memory_stats']
            original_kb = mem_stats.get('original_memory_kb', 0)
            optimized_kb = mem_stats.get('optimized_memory_kb', 0)
            saving_ratio = mem_stats.get('saving_ratio', 0)
            num_pools = mem_stats.get('num_pools', 0)
            
            print(f"  Memory Planning:")
            print(f"    Original memory:  {format_memory(original_kb)}")
            print(f"    Optimized memory: {format_memory(optimized_kb)}")
            print(f"    Memory saving:    {saving_ratio*100:.2f}%")
            print(f"    Number of pools:  {num_pools}")
        else:
            print(f"  Memory:         N/A (memory planning disabled or no stats)")
    else:
        print("  [NOT AVAILABLE]")
    print()
    
    # Comparison summary
    print_separator()
    print("Memory Usage Comparison")
    print_separator()
    print()
    
    if with_memory and 'memory_stats' in with_memory:
        mem_stats = with_memory['memory_stats']
        original_kb = mem_stats.get('original_memory_kb', 0)
        optimized_kb = mem_stats.get('optimized_memory_kb', 0)
        saving_ratio = mem_stats.get('saving_ratio', 0)
        
        print(f"  Without Memory Planning: {format_memory(original_kb)}")
        print(f"  With Memory Planning:    {format_memory(optimized_kb)}")
        print(f"  Memory Saved:            {format_memory(original_kb - optimized_kb)} ({saving_ratio*100:.2f}%)")
        print()
        
        if saving_ratio > 0.3:
            print(f"  [SUCCESS] Excellent memory optimization (>{saving_ratio*100:.0f}% saved)")
        elif saving_ratio > 0.1:
            print(f"  [SUCCESS] Good memory optimization ({saving_ratio*100:.0f}% saved)")
        else:
            print(f"  [WARNING] Limited memory optimization ({saving_ratio*100:.0f}% saved)")
    else:
        print(f"  [WARNING] Memory planning data not available")
    
    print()
    print_separator()

def main():
    """Main function"""
    # Default file paths
    no_memory_file = "test_samples/optimized_no_memory_outputs.json"
    with_memory_file = "test_samples/optimized_memory_outputs.json"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        no_memory_file = sys.argv[1]
    if len(sys.argv) > 2:
        with_memory_file = sys.argv[2]
    
    compare_memory_usage(no_memory_file, with_memory_file)

if __name__ == "__main__":
    main()
