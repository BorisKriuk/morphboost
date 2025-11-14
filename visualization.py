import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BenchmarkVisualizer:
    def __init__(self, output_dir='benchmark_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Color scheme for models
        self.model_colors = {
            'MorphBoost': '#FF6B6B',
            'XGBoost': '#4ECDC4',
            'LightGBM': '#95E1D3',
            'CatBoost': '#F38181',
            'HistGradBoost': '#AA96DA',
            'GradBoost': '#FCBAD3',
            'RandomForest': '#A8E6CF',
            'ExtraTrees': '#FFD3B6',
            'AdaBoost': '#FFAAA5',
            'BaggingDT': '#C7CEEA',
            'VotingEnsemble': '#B4E7CE',
            'LogisticReg': '#FFDAC1',
            'RidgeClass': '#E2F0CB',
            'SGD': '#B5EAD7',
            'NaiveBayes': '#C7CEEA',
            'QDA': '#E0BBE4'
        }
    
    def create_all_visualizations(self, all_results, datasets, models, results_matrix):
        """Generate all visualization plots"""
        print("\n" + "="*120)
        print(" "*40 + "🎨 GENERATING VISUALIZATIONS")
        print("="*120)
        
        viz_count = 0
        
        # 1. Overall Rankings Bar Chart
        print(f"\n📊 Creating visualization {viz_count+1}: Overall Rankings...")
        self.plot_overall_rankings(all_results, models)
        viz_count += 1
        
        # 2. Accuracy Heatmap
        print(f"📊 Creating visualization {viz_count+1}: Accuracy Heatmap...")
        self.plot_accuracy_heatmap(results_matrix, models, datasets)
        viz_count += 1
        
        # 3. Performance by Difficulty
        print(f"📊 Creating visualization {viz_count+1}: Performance by Difficulty...")
        self.plot_difficulty_performance(all_results, models)
        viz_count += 1
        
        # 4. Win Distribution
        print(f"📊 Creating visualization {viz_count+1}: Win Distribution...")
        self.plot_win_distribution(all_results, models)
        viz_count += 1
        
        # 5. Top-3 Distribution
        print(f"📊 Creating visualization {viz_count+1}: Top-3 Distribution...")
        self.plot_top3_distribution(all_results, models)
        viz_count += 1
        
        # 6. Accuracy Box Plot
        print(f"📊 Creating visualization {viz_count+1}: Accuracy Distribution...")
        self.plot_accuracy_boxplot(all_results, models)
        viz_count += 1
        
        # 7. Training Time Comparison
        print(f"📊 Creating visualization {viz_count+1}: Training Time Comparison...")
        self.plot_training_times(all_results, models)
        viz_count += 1
        
        # 8. Performance-Speed Tradeoff
        print(f"📊 Creating visualization {viz_count+1}: Performance-Speed Tradeoff...")
        self.plot_performance_speed_tradeoff(all_results, models)
        viz_count += 1
        
        # 9. Top-3 Finish Rate
        print(f"📊 Creating visualization {viz_count+1}: Top-3 Finish Rate...")
        self.plot_top3_rate(all_results, models)
        viz_count += 1
        
        # 10. Consistency Analysis
        print(f"📊 Creating visualization {viz_count+1}: Consistency Analysis...")
        self.plot_consistency_analysis(all_results, models)
        viz_count += 1
        
        # 11. Dataset-wise Performance Radar
        print(f"📊 Creating visualization {viz_count+1}: Dataset Performance Radar...")
        self.plot_dataset_radar(results_matrix, models, datasets)
        viz_count += 1
        
        # 12. MorphBoost vs XGBoost Head-to-Head
        print(f"📊 Creating visualization {viz_count+1}: MorphBoost vs XGBoost...")
        self.plot_morphboost_vs_xgboost(all_results, datasets, results_matrix, models)
        viz_count += 1
        
        # 13. MorphBoost Performance by Dataset
        print(f"📊 Creating visualization {viz_count+1}: MorphBoost by Dataset...")
        self.plot_morphboost_by_dataset(all_results, datasets, results_matrix, models)
        viz_count += 1
        
        # 14. MorphBoost Win/Loss Record
        print(f"📊 Creating visualization {viz_count+1}: MorphBoost Competition Record...")
        self.plot_morphboost_competition_record(all_results, datasets)
        viz_count += 1
        
        # 15. MorphBoost Difficulty Analysis
        print(f"📊 Creating visualization {viz_count+1}: MorphBoost Difficulty Analysis...")
        self.plot_morphboost_difficulty(all_results, models)
        viz_count += 1
        
        # 16. MorphBoost Statistics Summary
        print(f"📊 Creating visualization {viz_count+1}: MorphBoost Statistics...")
        self.plot_morphboost_statistics(all_results, models)
        viz_count += 1
        
        # 17. Model Type Comparison - Accuracy
        print(f"📊 Creating visualization {viz_count+1}: Model Type Accuracy...")
        self.plot_model_type_accuracy(all_results, models)
        viz_count += 1
        
        # 18. Model Type Comparison - Speed
        print(f"📊 Creating visualization {viz_count+1}: Model Type Speed...")
        self.plot_model_type_speed(all_results, models)
        viz_count += 1
        
        # 19. Accuracy Range (Min-Max)
        print(f"📊 Creating visualization {viz_count+1}: Accuracy Range...")
        self.plot_accuracy_range(all_results, models)
        viz_count += 1
        
        # 20. Median Accuracy Comparison
        print(f"📊 Creating visualization {viz_count+1}: Median Accuracy...")
        self.plot_median_accuracy(all_results, models)
        viz_count += 1
        
        # 21. Performance Consistency Score
        print(f"📊 Creating visualization {viz_count+1}: Consistency Score...")
        self.plot_consistency_score(all_results, models)
        viz_count += 1
        
        # 22. Dataset Difficulty Distribution
        print(f"📊 Creating visualization {viz_count+1}: Dataset Difficulty...")
        self.plot_dataset_difficulty(datasets, results_matrix, models)
        viz_count += 1
        
        # 23. Model Rankings Evolution
        print(f"📊 Creating visualization {viz_count+1}: Rankings Evolution...")
        self.plot_rankings_evolution(results_matrix, models, datasets)
        viz_count += 1
        
        # 24. Accuracy Violin Plot
        print(f"📊 Creating visualization {viz_count+1}: Accuracy Violin Plot...")
        self.plot_accuracy_violin(all_results, models)
        viz_count += 1
        
        # 25. Speed Efficiency Index
        print(f"📊 Creating visualization {viz_count+1}: Speed Efficiency...")
        self.plot_speed_efficiency(all_results, models)
        viz_count += 1
        
        print(f"\n✅ Generated {viz_count} visualizations in '{self.output_dir}/' directory")
        print("="*120)
    
    def plot_overall_rankings(self, all_results, models):
        """Bar chart of average accuracy rankings"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            if stats['accuracies']:
                summary_data.append({
                    'name': model_name,
                    'avg_acc': np.mean(stats['accuracies']),
                    'std_acc': np.std(stats['accuracies'])
                })
        
        summary_data.sort(key=lambda x: x['avg_acc'], reverse=True)
        
        names = [d['name'] for d in summary_data]
        accs = [d['avg_acc'] for d in summary_data]
        stds = [d['std_acc'] for d in summary_data]
        colors = [self.model_colors.get(name, '#95a5a6') for name in names]
        
        bars = ax.barh(names, accs, xerr=stds, color=colors, alpha=0.8, 
                       error_kw={'linewidth': 2, 'ecolor': 'gray', 'alpha': 0.6})
        
        # Highlight MorphBoost
        morph_idx = next((i for i, n in enumerate(names) if n == 'MorphBoost'), None)
        if morph_idx is not None:
            bars[morph_idx].set_edgecolor('red')
            bars[morph_idx].set_linewidth(3)
        
        ax.set_xlabel('Average Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Overall Model Rankings (10 Datasets Average)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (acc, std) in enumerate(zip(accs, stds)):
            ax.text(acc + 0.002, i, f'{acc:.4f}±{std:.4f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_overall_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_accuracy_heatmap(self, results_matrix, models, datasets):
        """Heatmap showing accuracy across all model-dataset combinations"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        model_names = [m['name'] for m in models]
        dataset_names = [d['name'] for d in datasets]
        
        # Create heatmap
        im = ax.imshow(results_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
        
        # Set ticks
        ax.set_xticks(np.arange(len(dataset_names)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(model_names, fontsize=11)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(model_names)):
            for j in range(len(dataset_names)):
                if results_matrix[i, j] > 0:
                    text_color = 'white' if results_matrix[i, j] < 0.75 else 'black'
                    ax.text(j, i, f'{results_matrix[i, j]:.3f}',
                           ha='center', va='center', color=text_color, fontsize=8)
        
        ax.set_title('Accuracy Heatmap: Models vs Datasets', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_difficulty_performance(self, all_results, models):
        """Line plot showing performance across difficulty levels"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        difficulties = ['Easy', 'Medium', 'Hard', 'Very Hard']
        
        # Get top 7 models by average accuracy
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            if stats['accuracies']:
                summary_data.append({
                    'name': model_name,
                    'avg_acc': np.mean(stats['accuracies'])
                })
        summary_data.sort(key=lambda x: x['avg_acc'], reverse=True)
        top_models = [d['name'] for d in summary_data[:7]]
        
        for model_name in top_models:
            stats = all_results[model_name]['by_difficulty']
            avg_accs = []
            for diff in difficulties:
                if stats[diff]:
                    avg_accs.append(np.mean(stats[diff]))
                else:
                    avg_accs.append(np.nan)
            
            color = self.model_colors.get(model_name, '#95a5a6')
            linewidth = 3 if model_name == 'MorphBoost' else 2
            linestyle = '-' if model_name == 'MorphBoost' else '--'
            marker = 'o' if model_name == 'MorphBoost' else 's'
            markersize = 10 if model_name == 'MorphBoost' else 7
            
            ax.plot(difficulties, avg_accs, marker=marker, linewidth=linewidth,
                   linestyle=linestyle, label=model_name, color=color, 
                   markersize=markersize, alpha=0.8)
        
        ax.set_xlabel('Dataset Difficulty', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Performance Across Difficulty Levels (Top 7 Models)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_difficulty_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_win_distribution(self, all_results, models):
        """Pie chart showing distribution of dataset wins"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Wins pie chart
        win_data = [(m['name'], all_results[m['name']]['wins']) 
                    for m in models if all_results[m['name']]['wins'] > 0]
        win_data.sort(key=lambda x: x[1], reverse=True)
        
        if win_data:
            names, wins = zip(*win_data)
            colors = [self.model_colors.get(name, '#95a5a6') for name in names]
            
            wedges, texts, autotexts = ax.pie(wins, labels=names, autopct='%1.1f%%',
                                               colors=colors, startangle=90,
                                               textprops={'fontsize': 12, 'fontweight': 'bold'})
            
            # Highlight MorphBoost
            morph_idx = next((i for i, n in enumerate(names) if n == 'MorphBoost'), None)
            if morph_idx is not None:
                wedges[morph_idx].set_edgecolor('red')
                wedges[morph_idx].set_linewidth(4)
        
        ax.set_title('Distribution of Dataset Wins (10 Datasets Total)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_win_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_top3_distribution(self, all_results, models):
        """Pie chart for top-3 finishes"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        top3_data = [(m['name'], all_results[m['name']]['top3']) 
                     for m in models if all_results[m['name']]['top3'] > 0]
        top3_data.sort(key=lambda x: x[1], reverse=True)
        
        if top3_data:
            names, top3s = zip(*top3_data)
            colors = [self.model_colors.get(name, '#95a5a6') for name in names]
            
            wedges, texts, autotexts = ax.pie(top3s, labels=names, autopct='%1.1f%%',
                                               colors=colors, startangle=90,
                                               textprops={'fontsize': 12, 'fontweight': 'bold'})
            
            # Highlight MorphBoost
            morph_idx = next((i for i, n in enumerate(names) if n == 'MorphBoost'), None)
            if morph_idx is not None:
                wedges[morph_idx].set_edgecolor('red')
                wedges[morph_idx].set_linewidth(4)
        
        ax.set_title('Distribution of Top-3 Finishes (30 Possible Total)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_top3_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_accuracy_boxplot(self, all_results, models):
        """Box plot showing accuracy distribution for each model"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Prepare data
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            if stats['accuracies']:
                summary_data.append({
                    'name': model_name,
                    'avg_acc': np.mean(stats['accuracies']),
                    'accuracies': stats['accuracies']
                })
        
        summary_data.sort(key=lambda x: x['avg_acc'], reverse=True)
        
        names = [d['name'] for d in summary_data]
        data = [d['accuracies'] for d in summary_data]
        colors = [self.model_colors.get(name, '#95a5a6') for name in names]
        
        # Create box plot
        bp = ax.boxplot(data, labels=names, patch_artist=True, 
                        notch=True, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Highlight MorphBoost
        morph_idx = next((i for i, n in enumerate(names) if n == 'MorphBoost'), None)
        if morph_idx is not None:
            bp['boxes'][morph_idx].set_edgecolor('red')
            bp['boxes'][morph_idx].set_linewidth(3)
        
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Accuracy Distribution Across All Datasets', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '06_accuracy_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_times(self, all_results, models):
        """Bar chart comparing average training times"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            if stats['times']:
                summary_data.append({
                    'name': model_name,
                    'avg_time': np.mean(stats['times']),
                    'std_time': np.std(stats['times'])
                })
        
        summary_data.sort(key=lambda x: x['avg_time'])
        
        names = [d['name'] for d in summary_data]
        times = [d['avg_time'] for d in summary_data]
        stds = [d['std_time'] for d in summary_data]
        colors = [self.model_colors.get(name, '#95a5a6') for name in names]
        
        bars = ax.barh(names, times, xerr=stds, color=colors, alpha=0.8,
                       error_kw={'linewidth': 2, 'ecolor': 'gray', 'alpha': 0.6})
        
        # Highlight MorphBoost
        morph_idx = next((i for i, n in enumerate(names) if n == 'MorphBoost'), None)
        if morph_idx is not None:
            bars[morph_idx].set_edgecolor('red')
            bars[morph_idx].set_linewidth(3)
        
        ax.set_xlabel('Average Training Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('Training Time Comparison (Lower is Better)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (time, std) in enumerate(zip(times, stds)):
            ax.text(time + 0.01, i, f'{time:.3f}s', 
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '07_training_times.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_performance_speed_tradeoff(self, all_results, models):
        """Scatter plot of accuracy vs training time"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            if stats['accuracies'] and stats['times']:
                summary_data.append({
                    'name': model_name,
                    'avg_acc': np.mean(stats['accuracies']),
                    'avg_time': np.mean(stats['times'])
                })
        
        for data in summary_data:
            name = data['name']
            color = self.model_colors.get(name, '#95a5a6')
            size = 300 if name == 'MorphBoost' else 150
            marker = 'D' if name == 'MorphBoost' else 'o'
            edgecolor = 'red' if name == 'MorphBoost' else 'black'
            linewidth = 3 if name == 'MorphBoost' else 1.5
            
            ax.scatter(data['avg_time'], data['avg_acc'], 
                      s=size, color=color, marker=marker, alpha=0.7,
                      edgecolors=edgecolor, linewidths=linewidth, label=name)
        
        ax.set_xlabel('Average Training Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Performance-Speed Tradeoff\n(Top-Left is Best: Fast & Accurate)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add quadrant lines
        median_time = np.median([d['avg_time'] for d in summary_data])
        median_acc = np.median([d['avg_acc'] for d in summary_data])
        ax.axvline(median_time, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax.axhline(median_acc, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '08_performance_speed_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_top3_rate(self, all_results, models):
        """Bar chart showing top-3 finish rates"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            summary_data.append({
                'name': model_name,
                'top3_rate': stats['top3'] / 30.0  # 30 possible (10 datasets × 3 positions)
            })
        
        summary_data.sort(key=lambda x: x['top3_rate'], reverse=True)
        
        names = [d['name'] for d in summary_data]
        rates = [d['top3_rate'] * 100 for d in summary_data]
        colors = [self.model_colors.get(name, '#95a5a6') for name in names]
        
        bars = ax.barh(names, rates, color=colors, alpha=0.8)
        
        # Highlight MorphBoost
        morph_idx = next((i for i, n in enumerate(names) if n == 'MorphBoost'), None)
        if morph_idx is not None:
            bars[morph_idx].set_edgecolor('red')
            bars[morph_idx].set_linewidth(3)
        
        ax.set_xlabel('Top-3 Finish Rate (%)', fontsize=14, fontweight='bold')
        ax.set_title('Consistency Metric: Top-3 Finish Rate', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, rate in enumerate(rates):
            ax.text(rate + 1, i, f'{rate:.1f}%', 
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '09_top3_rate.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_consistency_analysis(self, all_results, models):
        """Scatter plot of mean accuracy vs standard deviation"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            if stats['accuracies']:
                summary_data.append({
                    'name': model_name,
                    'avg_acc': np.mean(stats['accuracies']),
                    'std_acc': np.std(stats['accuracies'])
                })
        
        for data in summary_data:
            name = data['name']
            color = self.model_colors.get(name, '#95a5a6')
            size = 300 if name == 'MorphBoost' else 150
            marker = 'D' if name == 'MorphBoost' else 'o'
            edgecolor = 'red' if name == 'MorphBoost' else 'black'
            linewidth = 3 if name == 'MorphBoost' else 1.5
            
            ax.scatter(data['std_acc'], data['avg_acc'], 
                      s=size, color=color, marker=marker, alpha=0.7,
                      edgecolors=edgecolor, linewidths=linewidth, label=name)
        
        ax.set_xlabel('Standard Deviation of Accuracy (Lower = More Consistent)', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Consistency Analysis\n(Top-Left is Best: High Accuracy & Low Variance)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add quadrant lines
        median_std = np.median([d['std_acc'] for d in summary_data])
        median_acc = np.median([d['avg_acc'] for d in summary_data])
        ax.axvline(median_std, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax.axhline(median_acc, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '10_consistency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_dataset_radar(self, results_matrix, models, datasets):
        """Radar chart showing performance across datasets for top models"""
        # Get top 5 models
        avg_accs = np.mean(results_matrix, axis=1)
        top_indices = np.argsort(avg_accs)[-5:][::-1]
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        dataset_names = [d['name'] for d in datasets]
        
        for idx in top_indices:
            model_name = models[idx]['name']
            values = results_matrix[idx].tolist()
            values += values[:1]  # Complete the circle
            
            color = self.model_colors.get(model_name, '#95a5a6')
            linewidth = 3 if model_name == 'MorphBoost' else 2
            alpha = 0.3 if model_name == 'MorphBoost' else 0.15
            
            ax.plot(angles, values, 'o-', linewidth=linewidth, 
                   label=model_name, color=color)
            ax.fill(angles, values, alpha=alpha, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dataset_names, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.set_title('Dataset-wise Performance Radar (Top 5 Models)', 
                    fontsize=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '11_dataset_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_morphboost_vs_xgboost(self, all_results, datasets, results_matrix, models):
        """Head-to-head comparison"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        morph_idx = next((i for i, m in enumerate(models) if m['name'] == 'MorphBoost'), None)
        xgb_idx = next((i for i, m in enumerate(models) if m['name'] == 'XGBoost'), None)
        
        if morph_idx is not None and xgb_idx is not None:
            x = np.arange(len(datasets))
            width = 0.35
            
            dataset_names = [d['name'] for d in datasets]
            
            bars1 = ax.bar(x - width/2, results_matrix[morph_idx], width, 
                   label='MorphBoost', color='#FF6B6B', alpha=0.8, edgecolor='black')
            bars2 = ax.bar(x + width/2, results_matrix[xgb_idx], width, 
                   label='XGBoost', color='#4ECDC4', alpha=0.8, edgecolor='black')
            
            ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
            ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
            ax.set_title('MorphBoost vs XGBoost: Head-to-Head Comparison', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=11)
            ax.legend(fontsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '12_morphboost_vs_xgboost.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_morphboost_by_dataset(self, all_results, datasets, results_matrix, models):
        """MorphBoost performance across all datasets"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        morph_idx = next((i for i, m in enumerate(models) if m['name'] == 'MorphBoost'), None)
        
        if morph_idx is not None:
            dataset_names = [d['name'] for d in datasets]
            morph_accs = results_matrix[morph_idx]
            
            colors_grad = ['#FF6B6B' if acc == max(morph_accs) else '#FFA07A' 
                          if acc >= np.mean(morph_accs) else '#FFB6B9' for acc in morph_accs]
            
            bars = ax.bar(dataset_names, morph_accs, color=colors_grad, alpha=0.8, 
                         edgecolor='black', linewidth=1.5)
            ax.axhline(np.mean(morph_accs), color='red', linestyle='--', linewidth=2, label='Mean')
            
            ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
            ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
            ax.set_title('MorphBoost Performance Across All Datasets', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.legend(fontsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.xticks(rotation=45, ha='right', fontsize=11)
            
            # Add value labels
            for bar, acc in zip(bars, morph_accs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '13_morphboost_by_dataset.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_morphboost_competition_record(self, all_results, datasets):
        """Win/Loss record for MorphBoost"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        morph_stats = all_results.get('MorphBoost')
        if morph_stats:
            wins = morph_stats['wins']
            top3 = morph_stats['top3']
            total_datasets = len(datasets)
            
            categories = ['Wins', 'Top-3', 'Participated']
            values = [wins, top3, total_datasets]
            colors_bar = ['#4CAF50', '#FFC107', '#2196F3']
            
            bars = ax.bar(categories, values, color=colors_bar, alpha=0.8, 
                         edgecolor='black', linewidth=2)
            ax.set_ylabel('Count', fontsize=14, fontweight='bold')
            ax.set_title('MorphBoost Competition Record', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_ylim(0, max(values) * 1.2)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                       f'{int(val)}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '14_morphboost_competition_record.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_morphboost_difficulty(self, all_results, models):
        """Performance vs difficulty for MorphBoost"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        morph_stats = all_results.get('MorphBoost')
        if morph_stats:
            difficulties = ['Easy', 'Medium', 'Hard', 'Very Hard']
            diff_accs = []
            for diff in difficulties:
                if morph_stats['by_difficulty'][diff]:
                    diff_accs.append(np.mean(morph_stats['by_difficulty'][diff]))
                else:
                    diff_accs.append(0)
            
            colors_diff = ['#90EE90', '#FFD700', '#FFA500', '#FF6347']
            ax.plot(difficulties, diff_accs, marker='o', linewidth=3, markersize=12, 
                   color='#FF6B6B', markerfacecolor='#FF6B6B', markeredgecolor='black', 
                   markeredgewidth=2)
            ax.fill_between(range(len(difficulties)), diff_accs, alpha=0.3, color='#FF6B6B')
            
            ax.set_ylabel('Average Accuracy', fontsize=14, fontweight='bold')
            ax.set_xlabel('Dataset Difficulty', fontsize=14, fontweight='bold')
            ax.set_title('MorphBoost: Performance vs Difficulty', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(min(diff_accs) * 0.95, max(diff_accs) * 1.05)
            
            # Add value labels
            for i, (diff, acc) in enumerate(zip(difficulties, diff_accs)):
                ax.text(i, acc + 0.005, f'{acc:.4f}', ha='center', va='bottom', 
                       fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '15_morphboost_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_morphboost_statistics(self, all_results, models):
        """Statistics summary for MorphBoost"""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        morph_stats = all_results.get('MorphBoost')
        if morph_stats and morph_stats['accuracies']:
            stats_text = f"""
╔═══════════════════════════════════════════════════════╗
║        MORPHBOOST PERFORMANCE SUMMARY                 ║
╚═══════════════════════════════════════════════════════╝

📊 ACCURACY STATISTICS
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Mean:          {np.mean(morph_stats['accuracies']):.6f}
   • Median:        {np.median(morph_stats['accuracies']):.6f}
   • Std Dev:       {np.std(morph_stats['accuracies']):.6f}
   • Min:           {np.min(morph_stats['accuracies']):.6f}
   • Max:           {np.max(morph_stats['accuracies']):.6f}
   • Range:         {np.max(morph_stats['accuracies']) - np.min(morph_stats['accuracies']):.6f}

🏆 COMPETITION RECORD
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Wins:          {morph_stats['wins']}/10 ({morph_stats['wins']/10*100:.1f}%)
   • Top-3:         {morph_stats['top3']}/30 ({morph_stats['top3']/30*100:.1f}%)
   • Win Rate:      {morph_stats['wins']/10*100:.1f}%
   • Podium Rate:   {morph_stats['top3']/30*100:.1f}%

⚡ SPEED METRICS
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Avg Training:  {np.mean(morph_stats['times']):.4f}s
   • Min Training:  {np.min(morph_stats['times']):.4f}s
   • Max Training:  {np.max(morph_stats['times']):.4f}s
   • Total Time:    {np.sum(morph_stats['times']):.4f}s

📈 EFFICIENCY
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Acc/Second:    {np.mean(morph_stats['accuracies'])/np.mean(morph_stats['times']):.6f}
   • Consistency:   {1 - np.std(morph_stats['accuracies']):.6f}
            """
            
            ax.text(0.5, 0.5, stats_text, transform=ax.transAxes, 
                   fontsize=13, verticalalignment='center', horizontalalignment='center',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=1.5', facecolor='#FFE5E5', 
                            edgecolor='#FF6B6B', linewidth=3, alpha=0.9))
        
        fig.suptitle('🚀 MorphBoost Statistics Summary', fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '16_morphboost_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_type_accuracy(self, all_results, models):
        """Accuracy comparison by model type"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by type
        type_stats = {}
        for model_info in models:
            model_name = model_info['name']
            model_type = model_info['type']
            stats = all_results[model_name]
            
            if stats['accuracies']:
                if model_type not in type_stats:
                    type_stats[model_type] = []
                type_stats[model_type].extend(stats['accuracies'])
        
        types = list(type_stats.keys())
        avg_accs = [np.mean(type_stats[t]) for t in types]
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
        
        bars = ax.barh(types, avg_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_xlabel('Average Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Performance by Model Type', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, acc in enumerate(avg_accs):
            ax.text(acc + 0.002, i, f'{acc:.4f}', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '17_model_type_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_type_speed(self, all_results, models):
        """Speed comparison by model type"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by type
        type_stats = {}
        for model_info in models:
            model_name = model_info['name']
            model_type = model_info['type']
            stats = all_results[model_name]
            
            if stats['times']:
                if model_type not in type_stats:
                    type_stats[model_type] = []
                type_stats[model_type].extend(stats['times'])
        
        types = list(type_stats.keys())
        avg_times = [np.mean(type_stats[t]) for t in types]
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
        
        bars = ax.barh(types, avg_times, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.set_xlabel('Average Training Time (s)', fontsize=14, fontweight='bold')
        ax.set_title('Speed by Model Type (Lower is Better)', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, time in enumerate(avg_times):
            ax.text(time + 0.01, i, f'{time:.3f}s', va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '18_model_type_speed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_accuracy_range(self, all_results, models):
        """Min-Max accuracy range"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            if stats['accuracies']:
                summary_data.append({
                    'name': model_name,
                    'min_acc': np.min(stats['accuracies']),
                    'max_acc': np.max(stats['accuracies']),
                    'avg_acc': np.mean(stats['accuracies'])
                })
        
        summary_data.sort(key=lambda x: x['avg_acc'], reverse=True)
        
        names = [d['name'] for d in summary_data]
        mins = [d['min_acc'] for d in summary_data]
        maxs = [d['max_acc'] for d in summary_data]
        avgs = [d['avg_acc'] for d in summary_data]
        
        y_pos = np.arange(len(names))
        colors = [self.model_colors.get(name, '#95a5a6') for name in names]
        
        for i, (name, min_val, max_val, avg_val, color) in enumerate(zip(names, mins, maxs, avgs, colors)):
            ax.plot([min_val, max_val], [i, i], 'o-', linewidth=3, markersize=8, 
                   color=color, alpha=0.7)
            ax.plot([avg_val], [i], 'D', markersize=10, color='red', 
                   markeredgecolor='black', markeredgewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('Accuracy Range (Min ◯ —— Average ◆ —— Max ◯)', fontsize=14, fontweight='bold')
        ax.set_title('Accuracy Range: Min-Max-Average', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '19_accuracy_range.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_median_accuracy(self, all_results, models):
        """Median accuracy comparison"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            if stats['accuracies']:
                summary_data.append({
                    'name': model_name,
                    'median_acc': np.median(stats['accuracies'])
                })
        
        summary_data.sort(key=lambda x: x['median_acc'], reverse=True)
        
        names = [d['name'] for d in summary_data]
        medians = [d['median_acc'] for d in summary_data]
        colors = [self.model_colors.get(name, '#95a5a6') for name in names]
        
        bars = ax.barh(names, medians, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Highlight MorphBoost
        morph_idx = next((i for i, n in enumerate(names) if n == 'MorphBoost'), None)
        if morph_idx is not None:
            bars[morph_idx].set_edgecolor('red')
            bars[morph_idx].set_linewidth(3)
        
        ax.set_xlabel('Median Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Median Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, median in enumerate(medians):
            ax.text(median + 0.002, i, f'{median:.4f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '20_median_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_consistency_score(self, all_results, models):
        """Consistency score (1 - std_dev)"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            if stats['accuracies']:
                consistency = 1 - np.std(stats['accuracies'])
                summary_data.append({
                    'name': model_name,
                    'consistency': consistency
                })
        
        summary_data.sort(key=lambda x: x['consistency'], reverse=True)
        
        names = [d['name'] for d in summary_data]
        scores = [d['consistency'] for d in summary_data]
        colors = [self.model_colors.get(name, '#95a5a6') for name in names]
        
        bars = ax.barh(names, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Highlight MorphBoost
        morph_idx = next((i for i, n in enumerate(names) if n == 'MorphBoost'), None)
        if morph_idx is not None:
            bars[morph_idx].set_edgecolor('red')
            bars[morph_idx].set_linewidth(3)
        
        ax.set_xlabel('Consistency Score (Higher is Better)', fontsize=14, fontweight='bold')
        ax.set_title('Model Consistency Score (1 - Std Dev)', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, score in enumerate(scores):
            ax.text(score + 0.002, i, f'{score:.4f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '21_consistency_score.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_dataset_difficulty(self, datasets, results_matrix, models):
        """Average performance per dataset (difficulty indicator)"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        dataset_names = [d['name'] for d in datasets]
        avg_accs = np.mean(results_matrix, axis=0)
        difficulties = [d['difficulty'] for d in datasets]
        
        color_map = {'Easy': '#90EE90', 'Medium': '#FFD700', 'Hard': '#FFA500', 'Very Hard': '#FF6347'}
        colors = [color_map.get(diff, '#95a5a6') for diff in difficulties]
        
        bars = ax.bar(dataset_names, avg_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Average Accuracy (All Models)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_title('Dataset Difficulty Analysis\n(Lower Avg = Harder Dataset)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right', fontsize=11)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[diff], label=diff, alpha=0.8, edgecolor='black') 
                          for diff in ['Easy', 'Medium', 'Hard', 'Very Hard']]
        ax.legend(handles=legend_elements, loc='best', fontsize=11)
        
        # Add value labels
        for bar, acc in zip(bars, avg_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '22_dataset_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rankings_evolution(self, results_matrix, models, datasets):
        """Model rankings across datasets"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        model_names = [m['name'] for m in models]
        dataset_names = [d['name'] for d in datasets]
        
        # Get rankings for each dataset
        rankings = []
        for d_idx in range(len(datasets)):
            dataset_accs = results_matrix[:, d_idx]
            ranks = len(dataset_accs) - np.argsort(np.argsort(dataset_accs))  # Reverse rank
            rankings.append(ranks)
        
        rankings = np.array(rankings).T  # Shape: (n_models, n_datasets)
        
        # Plot lines
        for m_idx, model_name in enumerate(model_names):
            color = self.model_colors.get(model_name, '#95a5a6')
            linewidth = 3 if model_name == 'MorphBoost' else 2
            linestyle = '-' if model_name == 'MorphBoost' else '--'
            marker = 'D' if model_name == 'MorphBoost' else 'o'
            markersize = 10 if model_name == 'MorphBoost' else 6
            alpha = 0.9 if model_name == 'MorphBoost' else 0.6
            
            ax.plot(range(len(datasets)), rankings[m_idx], marker=marker, 
                   linewidth=linewidth, linestyle=linestyle, label=model_name, 
                   color=color, markersize=markersize, alpha=alpha)
        
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([d['name'] for d in datasets], rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Rank (1 = Best)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_title('Model Rankings Evolution Across Datasets', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()  # Best rank at top
        ax.legend(loc='best', fontsize=9, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '23_rankings_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_accuracy_violin(self, all_results, models):
        """Violin plot for accuracy distribution"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            if stats['accuracies']:
                summary_data.append({
                    'name': model_name,
                    'avg_acc': np.mean(stats['accuracies']),
                    'accuracies': stats['accuracies']
                })
        
        summary_data.sort(key=lambda x: x['avg_acc'], reverse=True)
        
        names = [d['name'] for d in summary_data]
        data = [d['accuracies'] for d in summary_data]
        colors = [self.model_colors.get(name, '#95a5a6') for name in names]
        
        parts = ax.violinplot(data, positions=range(len(names)), 
                             showmeans=True, showmedians=True)
        
        # Color violins
        for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title('Accuracy Distribution (Violin Plot)', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '24_accuracy_violin.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_speed_efficiency(self, all_results, models):
        """Speed efficiency index (accuracy/time)"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        summary_data = []
        for model_info in models:
            model_name = model_info['name']
            stats = all_results[model_name]
            if stats['accuracies'] and stats['times']:
                efficiency = np.mean(stats['accuracies']) / (np.mean(stats['times']) + 0.001)
                summary_data.append({
                    'name': model_name,
                    'efficiency': efficiency
                })
        
        summary_data.sort(key=lambda x: x['efficiency'], reverse=True)
        
        names = [d['name'] for d in summary_data]
        efficiencies = [d['efficiency'] for d in summary_data]
        colors = [self.model_colors.get(name, '#95a5a6') for name in names]
        
        bars = ax.barh(names, efficiencies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Highlight MorphBoost
        morph_idx = next((i for i, n in enumerate(names) if n == 'MorphBoost'), None)
        if morph_idx is not None:
            bars[morph_idx].set_edgecolor('red')
            bars[morph_idx].set_linewidth(3)
        
        ax.set_xlabel('Efficiency (Accuracy per Second)', fontsize=14, fontweight='bold')
        ax.set_title('Speed Efficiency Index (Higher is Better)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, eff in enumerate(efficiencies):
            ax.text(eff + 0.002, i, f'{eff:.4f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '25_speed_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()