#!/usr/bin/env python3
"""
JFR Queue Size Analysis Script

Analyzes CSV/JSON benchmark results to create comprehensive visualizations showing:
- Loss percentage vs configured queue size
- p99 queue size vs configured queue size  
- p99.9 queue size vs configured queue size

Creates separate graphs for each native duration and Renaissance tests.

Usage:
    python3 analyze_benchmark_results.py --csv <csv_file>
    python3 analyze_benchmark_results.py --json <json_file>
    python3 analyze_benchmark_results.py --latest  # Use latest results
    python3 analyze_benchmark_results.py --regenerate  # Re-run analysis to extract queue stats
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import os

# Configuration
RESULTS_DIR = Path("benchmark_results")
LOGS_DIR = RESULTS_DIR / "logs"
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR = RESULTS_DIR / "plots"

class BenchmarkAnalyzer:
    def __init__(self):
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        for dir_path in [PLOTS_DIR]:
            dir_path.mkdir(exist_ok=True)
    
    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, bool]:
        """Load data from CSV or JSON file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        has_queue_stats = False
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            print(f"📊 Loaded CSV with {len(df)} records from {file_path}")
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f"📊 Loaded JSON with {len(df)} records from {file_path}")
            
            # Check if queue_stats data is available
            if 'queue_stats' in df.columns:
                has_queue_stats = any(
                    isinstance(stats, dict) and stats for stats in df['queue_stats'].fillna({})
                )
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Filter only successful tests
        if 'success' in df.columns:
            df = df[df['success'] == True].copy()
            print(f"📊 Filtered to {len(df)} successful tests")
        
        print(f"📊 Queue stats available: {'✅' if has_queue_stats else '❌'}")
        
        return df, has_queue_stats
    
    def extract_queue_percentiles(self, queue_stats: Dict) -> Dict:
        """Extract p99 and p99.9 values from queue_stats"""
        percentiles = {}
        
        if not isinstance(queue_stats, dict):
            return percentiles
            
        for category, stats in queue_stats.items():
            if isinstance(stats, dict):
                # Look for p99 and p99_9 (99.9%)
                if 'p99' in stats:
                    percentiles[f'{category}_p99'] = stats['p99']
                if 'p99_9' in stats:
                    percentiles[f'{category}_p99_9'] = stats['p99_9']
                elif 'p99.9' in stats:
                    percentiles[f'{category}_p99_9'] = stats['p99.9']
        
        return percentiles
    
    def prepare_data_with_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Expand dataframe to include queue percentile columns"""
        if 'queue_stats' not in df.columns:
            print("⚠️ No queue_stats column found - percentile analysis not available")
            return df
        
        # Extract percentiles and add as new columns
        percentile_data = []
        
        for idx, row in df.iterrows():
            row_data = row.to_dict()
            queue_stats = row.get('queue_stats', {})
            
            if isinstance(queue_stats, dict):
                percentiles = self.extract_queue_percentiles(queue_stats)
                row_data.update(percentiles)
            
            percentile_data.append(row_data)
        
        expanded_df = pd.DataFrame(percentile_data)
        
        # Report what percentile columns we found
        percentile_cols = [col for col in expanded_df.columns if '_p99' in col]
        if percentile_cols:
            print(f"📊 Found percentile columns: {percentile_cols}")
        else:
            print("⚠️ No percentile data found in queue_stats")
        
        return expanded_df
    
    def create_loss_analysis(self, df: pd.DataFrame):
        """Create loss percentage analysis graphs"""
        print("📊 Creating loss percentage analysis...")
        
        # Separate native and renaissance data
        native_df = df[df.get('native_duration').notna()].copy() if 'native_duration' in df.columns else pd.DataFrame()
        renaissance_df = df[df.get('iterations').notna()].copy() if 'iterations' in df.columns else pd.DataFrame()
        
        if native_df.empty and renaissance_df.empty:
            print("⚠️ No data found for analysis")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Native duration analysis
        if not native_df.empty:
            self.plot_native_loss_analysis(native_df)
        
        # 2. Renaissance analysis  
        if not renaissance_df.empty:
            self.plot_renaissance_loss_analysis(renaissance_df)
    
    def plot_native_loss_analysis(self, df: pd.DataFrame):
        """Create loss analysis for different native durations"""
        durations = sorted(df['native_duration'].unique())
        intervals = sorted(df['interval'].unique())
        
        # Create subplots for each duration
        n_durations = len(durations)
        fig, axes = plt.subplots(1, n_durations, figsize=(5*n_durations, 6))
        if n_durations == 1:
            axes = [axes]
        
        fig.suptitle('Loss Percentage vs Configured Queue Size (Native Tests)', fontsize=16)
        
        for i, duration in enumerate(durations):
            df_dur = df[df['native_duration'] == duration]
            
            for interval in intervals:
                df_int = df_dur[df_dur['interval'] == interval]
                if not df_int.empty:
                    df_int = df_int.sort_values('queue_size')
                    axes[i].plot(df_int['queue_size'], df_int['loss_percentage'], 
                               marker='o', label=f'{interval}', linewidth=2, markersize=6)
            
            axes[i].set_title(f'Native Duration: {duration}s')
            axes[i].set_xlabel('Configured Queue Size')
            axes[i].set_ylabel('Loss Percentage (%)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'loss_analysis_native.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📁 Saved: {PLOTS_DIR / 'loss_analysis_native.png'}")
    
    def plot_renaissance_loss_analysis(self, df: pd.DataFrame):
        """Create loss analysis for Renaissance tests"""
        intervals = sorted(df['interval'].unique())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for interval in intervals:
            df_int = df[df['interval'] == interval]
            if not df_int.empty:
                df_int = df_int.sort_values('queue_size')
                ax.plot(df_int['queue_size'], df_int['loss_percentage'], 
                       marker='o', label=f'{interval}', linewidth=2, markersize=6)
        
        ax.set_title('Loss Percentage vs Configured Queue Size (Renaissance Tests)', fontsize=14)
        ax.set_xlabel('Configured Queue Size')
        ax.set_ylabel('Loss Percentage (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'loss_analysis_renaissance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📁 Saved: {PLOTS_DIR / 'loss_analysis_renaissance.png'}")
    
    def create_percentile_analysis(self, df: pd.DataFrame):
        """Create p99 and p99.9 queue size analysis graphs"""
        print("📊 Creating queue size percentile analysis...")
        
        # Check if we have percentile data
        percentile_cols = [col for col in df.columns if '_p99' in col]
        if not percentile_cols:
            print("⚠️ No percentile data available - skipping percentile analysis")
            return
        
        # Separate native and renaissance data
        native_df = df[df.get('native_duration').notna()].copy() if 'native_duration' in df.columns else pd.DataFrame()
        renaissance_df = df[df.get('iterations').notna()].copy() if 'iterations' in df.columns else pd.DataFrame()
        
        # 1. Native duration analysis
        if not native_df.empty:
            self.plot_native_percentile_analysis(native_df, percentile_cols)
        
        # 2. Renaissance analysis  
        if not renaissance_df.empty:
            self.plot_renaissance_percentile_analysis(renaissance_df, percentile_cols)
    
    def plot_native_percentile_analysis(self, df: pd.DataFrame, percentile_cols: List[str]):
        """Create percentile analysis for different native durations"""
        durations = sorted(df['native_duration'].unique())
        intervals = sorted(df['interval'].unique())
        
        # Find categories and percentile types
        categories = set()
        percentile_types = set()
        
        for col in percentile_cols:
            parts = col.split('_p99')
            if len(parts) == 2:
                category = parts[0]
                perc_type = 'p99' + parts[1]  # p99 or p99_9
                categories.add(category)
                percentile_types.add(perc_type)
        
        categories = sorted(list(categories))
        percentile_types = sorted(list(percentile_types))
        
        print(f"   📊 Found categories: {categories}")
        print(f"   📊 Found percentiles: {percentile_types}")
        
        for category in categories:
            for perc_type in percentile_types:
                col_name = f'{category}_{perc_type}'
                if col_name not in df.columns:
                    continue
                
                # Create subplots for each duration
                n_durations = len(durations)
                fig, axes = plt.subplots(1, n_durations, figsize=(5*n_durations, 6))
                if n_durations == 1:
                    axes = [axes]
                
                percentile_display = perc_type.replace('_', '.')
                fig.suptitle(f'{percentile_display.upper()} Queue Size vs Configured Size - {category.upper()} (Native Tests)', fontsize=16)
                
                for i, duration in enumerate(durations):
                    df_dur = df[df['native_duration'] == duration]
                    
                    for interval in intervals:
                        df_int = df_dur[df_dur['interval'] == interval]
                        if not df_int.empty and col_name in df_int.columns:
                            df_int = df_int.sort_values('queue_size')
                            # Filter out NaN values
                            df_int = df_int[df_int[col_name].notna()]
                            if not df_int.empty:
                                axes[i].plot(df_int['queue_size'], df_int[col_name], 
                                           marker='o', label=f'{interval}', linewidth=2, markersize=6)
                    
                    # Add diagonal line showing configured = actual
                    if not df_dur.empty:
                        min_queue = df_dur['queue_size'].min()
                        max_queue = df_dur['queue_size'].max()
                        axes[i].plot([min_queue, max_queue], [min_queue, max_queue], 
                                   'k--', alpha=0.5, label='Configured Size')
                    
                    axes[i].set_title(f'Native Duration: {duration}s')
                    axes[i].set_xlabel('Configured Queue Size')
                    axes[i].set_ylabel(f'{percentile_display.upper()} Actual Queue Size')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                    axes[i].set_xscale('log')
                    axes[i].set_yscale('log')
                
                plt.tight_layout()
                filename = f'percentile_analysis_native_{category}_{perc_type}.png'
                plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   📁 Saved: {PLOTS_DIR / filename}")
    
    def plot_renaissance_percentile_analysis(self, df: pd.DataFrame, percentile_cols: List[str]):
        """Create percentile analysis for Renaissance tests"""
        intervals = sorted(df['interval'].unique())
        
        # Find categories and percentile types
        categories = set()
        percentile_types = set()
        
        for col in percentile_cols:
            parts = col.split('_p99')
            if len(parts) == 2:
                category = parts[0]
                perc_type = 'p99' + parts[1]  # p99 or p99_9
                categories.add(category)
                percentile_types.add(perc_type)
        
        categories = sorted(list(categories))
        percentile_types = sorted(list(percentile_types))
        
        for category in categories:
            for perc_type in percentile_types:
                col_name = f'{category}_{perc_type}'
                if col_name not in df.columns:
                    continue
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for interval in intervals:
                    df_int = df[df['interval'] == interval]
                    if not df_int.empty and col_name in df_int.columns:
                        df_int = df_int.sort_values('queue_size')
                        # Filter out NaN values
                        df_int = df_int[df_int[col_name].notna()]
                        if not df_int.empty:
                            ax.plot(df_int['queue_size'], df_int[col_name], 
                                   marker='o', label=f'{interval}', linewidth=2, markersize=6)
                
                # Add diagonal line showing configured = actual
                if not df.empty:
                    min_queue = df['queue_size'].min()
                    max_queue = df['queue_size'].max()
                    ax.plot([min_queue, max_queue], [min_queue, max_queue], 
                           'k--', alpha=0.5, label='Configured Size')
                
                percentile_display = perc_type.replace('_', '.')
                ax.set_title(f'{percentile_display.upper()} Queue Size vs Configured Size - {category.upper()} (Renaissance Tests)', fontsize=14)
                ax.set_xlabel('Configured Queue Size')
                ax.set_ylabel(f'{percentile_display.upper()} Actual Queue Size')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xscale('log')
                ax.set_yscale('log')
                
                plt.tight_layout()
                filename = f'percentile_analysis_renaissance_{category}_{perc_type}.png'
                plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   📁 Saved: {PLOTS_DIR / filename}")
    
    def regenerate_with_queue_stats(self):
        """Re-run the benchmark analysis to extract queue statistics"""
        print("🔄 Regenerating analysis with queue size statistics extraction...")
        
        # Import and run the updated benchmark extraction
        from benchmark_queue_sizes import BenchmarkRunner
        
        # Load latest JSON results
        latest_json = DATA_DIR / "native_results_latest.json"
        if not latest_json.exists():
            print("❌ No latest results found. Run benchmark first.")
            return
        
        with open(latest_json, 'r') as f:
            results = json.load(f)
        
        print(f"📊 Re-analyzing {len(results)} test results...")
        
        # Create a benchmark runner to use its extraction methods
        runner = BenchmarkRunner()
        
        updated_results = []
        for i, result in enumerate(results):
            print(f"   Processing {i+1}/{len(results)}: {result.get('log_file', 'unknown')}")
            
            log_file = result.get('log_file')
            if log_file:
                log_path = LOGS_DIR / log_file
                if log_path.exists():
                    # Re-extract with updated method
                    extracted_data = runner.extract_loss_percentage(log_path)
                    
                    # Update result with new data
                    if isinstance(extracted_data, dict):
                        result['loss_percentage'] = extracted_data.get('loss_percentage', result.get('loss_percentage'))
                        result['queue_stats'] = extracted_data.get('queue_stats', {})
                    else:
                        result['loss_percentage'] = extracted_data or result.get('loss_percentage')
                        result['queue_stats'] = {}
                    
                    updated_results.append(result)
                else:
                    print(f"     ⚠️ Log file not found: {log_path}")
                    updated_results.append(result)
            else:
                updated_results.append(result)
        
        # Save updated results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        updated_json = DATA_DIR / f"native_updated_with_queue_stats_{timestamp}.json"
        
        with open(updated_json, 'w') as f:
            json.dump(updated_results, f, indent=2)
        
        # Also update latest
        with open(latest_json, 'w') as f:
            json.dump(updated_results, f, indent=2)
        
        print(f"✅ Updated results saved to {updated_json}")
        print(f"✅ Latest results updated")
        
        # Count how many have queue stats
        with_stats = sum(1 for r in updated_results if r.get('queue_stats'))
        print(f"📊 Results with queue stats: {with_stats}/{len(updated_results)}")
        
        return updated_json
    
    def analyze_file(self, file_path: str):
        """Analyze a specific file"""
        print(f"🔍 Analyzing file: {file_path}")
        
        df, has_queue_stats = self.load_data(file_path)
        
        if df.empty:
            print("❌ No data to analyze")
            return
        
        # Show data summary
        print(f"\n📊 Data Summary:")
        print(f"   Total records: {len(df)}")
        if 'native_duration' in df.columns:
            native_count = df['native_duration'].notna().sum()
            print(f"   Native tests: {native_count}")
        if 'iterations' in df.columns:
            renaissance_count = df['iterations'].notna().sum()
            print(f"   Renaissance tests: {renaissance_count}")
        
        # Create loss percentage analysis
        self.create_loss_analysis(df)
        
        # Create percentile analysis if data is available
        if has_queue_stats:
            df_expanded = self.prepare_data_with_percentiles(df)
            self.create_percentile_analysis(df_expanded)
        else:
            print("⚠️ No queue size percentile data available")
            print("💡 Use --regenerate to extract queue statistics from log files")

def main():
    parser = argparse.ArgumentParser(description='JFR Queue Size Benchmark Analysis')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv', type=str, help='Path to CSV file to analyze')
    group.add_argument('--json', type=str, help='Path to JSON file to analyze')
    group.add_argument('--latest', action='store_true', help='Analyze latest results')
    group.add_argument('--regenerate', action='store_true', help='Regenerate analysis with queue stats extraction')
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer()
    
    if args.regenerate:
        updated_file = analyzer.regenerate_with_queue_stats()
        if updated_file:
            print(f"\n🔍 Analyzing regenerated data...")
            analyzer.analyze_file(str(updated_file))
    elif args.latest:
        # Look for latest files
        latest_json = DATA_DIR / "native_results_latest.json"
        latest_csv = DATA_DIR / "native_results_latest.csv"
        
        if latest_json.exists():
            analyzer.analyze_file(str(latest_json))
        elif latest_csv.exists():
            analyzer.analyze_file(str(latest_csv))
        else:
            print("❌ No latest results found")
    elif args.csv:
        analyzer.analyze_file(args.csv)
    elif args.json:
        analyzer.analyze_file(args.json)

if __name__ == "__main__":
    main()
