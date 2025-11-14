import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits, load_iris,
    make_classification, make_moons, make_circles, make_blobs,
    fetch_covtype, fetch_olivetti_faces
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    HistGradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, BaggingClassifier, StackingClassifier,
    VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# Import models
from xgboost import XGBClassifier
from morphboost import MorphBoost

# Try importing optional models
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not available, skipping...")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not available, skipping...")

class ModelBenchmark:
    def __init__(self):
        self.results = []
        
    def get_datasets(self):
        """Load 10 diverse datasets for comprehensive testing"""
        datasets = []
        
        print("Loading datasets...")
        
        # 1. Breast Cancer (Binary, 569 samples, 30 features)
        data = load_breast_cancer()
        datasets.append({
            'name': 'Breast Cancer',
            'X': data.data,
            'y': data.target,
            'type': 'binary',
            'n_samples': len(data.target),
            'n_features': data.data.shape[1],
            'n_classes': 2,
            'difficulty': 'Easy'
        })
        
        # 2. Wine (Multiclass, 178 samples, 13 features)
        data = load_wine()
        datasets.append({
            'name': 'Wine',
            'X': data.data,
            'y': data.target,
            'type': 'multiclass',
            'n_samples': len(data.target),
            'n_features': data.data.shape[1],
            'n_classes': 3,
            'difficulty': 'Easy'
        })
        
        # 3. Iris (Multiclass, 150 samples, 4 features)
        data = load_iris()
        datasets.append({
            'name': 'Iris',
            'X': data.data,
            'y': data.target,
            'type': 'multiclass',
            'n_samples': len(data.target),
            'n_features': data.data.shape[1],
            'n_classes': 3,
            'difficulty': 'Easy'
        })
        
        # 4. Digits (Multiclass, 1797 samples, 64 features)
        data = load_digits()
        datasets.append({
            'name': 'Digits',
            'X': data.data,
            'y': data.target,
            'type': 'multiclass',
            'n_samples': len(data.target),
            'n_features': data.data.shape[1],
            'n_classes': 10,
            'difficulty': 'Medium'
        })
        
        # 5. Moons (Binary, nonlinear, 1000 samples)
        X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
        datasets.append({
            'name': 'Two Moons',
            'X': X,
            'y': y,
            'type': 'binary',
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_classes': 2,
            'difficulty': 'Medium'
        })
        
        # 6. Circles (Binary, nonlinear, 1000 samples)
        X, y = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=42)
        datasets.append({
            'name': 'Circles',
            'X': X,
            'y': y,
            'type': 'binary',
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_classes': 2,
            'difficulty': 'Medium'
        })
        
        # 7. Blobs (Multiclass, 1500 samples, 10 features, 5 classes)
        X, y = make_blobs(n_samples=1500, n_features=10, centers=5, 
                         cluster_std=1.5, random_state=42)
        datasets.append({
            'name': 'Blobs-5',
            'X': X,
            'y': y,
            'type': 'multiclass',
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_classes': 5,
            'difficulty': 'Easy'
        })
        
        # 8. High-dim Binary (2000 samples, 50 features)
        X, y = make_classification(n_samples=2000, n_features=50, n_informative=30,
                                  n_redundant=10, n_classes=2, flip_y=0.05,
                                  random_state=42)
        datasets.append({
            'name': 'HighDim-Binary',
            'X': X,
            'y': y,
            'type': 'binary',
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_classes': 2,
            'difficulty': 'Hard'
        })
        
        # 9. Imbalanced Multiclass (2000 samples, 20 features, 4 classes)
        X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                                  n_redundant=5, n_classes=4, n_clusters_per_class=2,
                                  weights=[0.5, 0.25, 0.15, 0.1], flip_y=0.05,
                                  random_state=42)
        datasets.append({
            'name': 'Imbalanced-4Class',
            'X': X,
            'y': y,
            'type': 'multiclass',
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_classes': 4,
            'difficulty': 'Hard'
        })
        
        # 10. Complex Synthetic (3000 samples, 100 features, 3 classes)
        X, y = make_classification(n_samples=3000, n_features=100, n_informative=50,
                                  n_redundant=25, n_classes=3, n_clusters_per_class=3,
                                  flip_y=0.1, class_sep=0.5, random_state=42)
        datasets.append({
            'name': 'Complex-100D',
            'X': X,
            'y': y,
            'type': 'multiclass',
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_classes': 3,
            'difficulty': 'Very Hard'
        })
        
        return datasets
    
    def get_models(self):
        """Initialize 10 competitive models"""
        models = [
            # 1. Our novel algorithm
            {
                'name': 'MorphBoost',
                'model': MorphBoost(
                    n_estimators=100,
                    learning_rate=0.3,
                    max_depth=6,
                    random_state=42
                ),
                'color': '🚀',
                'type': 'Boosting'
            },
            
            # 2. XGBoost - Current SOTA
            {
                'name': 'XGBoost',
                'model': XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.3,
                    max_depth=6,
                    random_state=42,
                    verbosity=0
                ),
                'color': '📈',
                'type': 'Boosting'
            },
            
            # 3. LightGBM (if available)
            {
                'name': 'LightGBM',
                'model': LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.3,
                    max_depth=6,
                    random_state=42,
                    verbosity=-1
                ) if HAS_LIGHTGBM else None,
                'color': '💡',
                'type': 'Boosting'
            },
            
            # 4. CatBoost (if available)
            {
                'name': 'CatBoost',
                'model': CatBoostClassifier(
                    n_estimators=100,
                    learning_rate=0.3,
                    max_depth=6,
                    random_state=42,
                    verbose=False
                ) if HAS_CATBOOST else None,
                'color': '🐱',
                'type': 'Boosting'
            },
            
            # 5. Scikit-learn's HistGradientBoosting
            {
                'name': 'HistGradBoost',
                'model': HistGradientBoostingClassifier(
                    max_iter=100,
                    learning_rate=0.3,
                    max_depth=6,
                    random_state=42
                ),
                'color': '📊',
                'type': 'Boosting'
            },
            
            # 6. Classic GradientBoosting
            {
                'name': 'GradBoost',
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.3,
                    max_depth=6,
                    random_state=42
                ),
                'color': '📉',
                'type': 'Boosting'
            },
            
            # 7. Random Forest (Strong baseline)
            {
                'name': 'RandomForest',
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'color': '🌲',
                'type': 'Bagging'
            },
            
            # 8. Extra Trees (Extremely Randomized Trees)
            {
                'name': 'ExtraTrees',
                'model': ExtraTreesClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'color': '🌳',
                'type': 'Bagging'
            },
            
            # 9. AdaBoost (Classic boosting)
            {
                'name': 'AdaBoost',
                'model': AdaBoostClassifier(
                    n_estimators=100,
                    learning_rate=1.0,
                    random_state=42
                ),
                'color': '🎯',
                'type': 'Boosting'
            },
            
            # 10. Advanced Bagging with Decision Trees
            {
                'name': 'BaggingDT',
                'model': BaggingClassifier(
                    estimator=DecisionTreeClassifier(max_depth=10),
                    n_estimators=100,
                    max_samples=0.8,
                    random_state=42,
                    n_jobs=-1
                ),
                'color': '🎲',
                'type': 'Bagging'
            },
            
            # 11. Soft Voting Ensemble (Fast models)
            {
                'name': 'VotingEnsemble',
                'model': VotingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)),
                        ('et', ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=42)),
                        ('lr', LogisticRegression(max_iter=1000, random_state=42))
                    ],
                    voting='soft',
                    n_jobs=-1
                ),
                'color': '🗳️',
                'type': 'Ensemble'
            },
            
            # 12. Logistic Regression (Strong linear baseline)
            {
                'name': 'LogisticReg',
                'model': LogisticRegression(
                    max_iter=1000,
                    solver='lbfgs',
                    random_state=42,
                    n_jobs=-1
                ),
                'color': '📐',
                'type': 'Linear'
            },
            
            # 13. Ridge Classifier (Fast regularized linear)
            {
                'name': 'RidgeClass',
                'model': RidgeClassifier(
                    alpha=1.0,
                    random_state=42
                ),
                'color': '⛰️',
                'type': 'Linear'
            },
            
            # 14. SGD Classifier (Scalable linear)
            {
                'name': 'SGD',
                'model': SGDClassifier(
                    loss='log_loss',
                    max_iter=1000,
                    random_state=42,
                    n_jobs=-1
                ),
                'color': '⚙️',
                'type': 'Linear'
            },
            
            # 15. Naive Bayes (Probabilistic baseline)
            {
                'name': 'NaiveBayes',
                'model': GaussianNB(),
                'color': '🎓',
                'type': 'Probabilistic'
            },
            
            # 16. Quadratic Discriminant Analysis
            {
                'name': 'QDA',
                'model': QuadraticDiscriminantAnalysis(),
                'color': '📏',
                'type': 'Discriminant'
            }
        ]
        
        # Filter out None models (missing dependencies)
        models = [m for m in models if m['model'] is not None]
        
        # Return top 9 most competitive
        priority_order = [
            'MorphBoost', 'XGBoost', 'LightGBM', 'CatBoost', 'HistGradBoost',
            'RandomForest', 'ExtraTrees', 'GradBoost', 'VotingEnsemble'
        ]
        
        # Sort by priority
        def get_priority(model):
            try:
                return priority_order.index(model['name'])
            except ValueError:
                return 999
        
        models_sorted = sorted(models, key=get_priority)
        return models_sorted[:10]
    
    def run_benchmark(self):
        """Run complete benchmark"""
        print("="*120)
        print(" "*40 + "🔬 COMPREHENSIVE MODEL BENCHMARK")
        print(" "*35 + "10 Models × 10 Datasets = 100 Experiments")
        print("="*120)
        
        datasets = self.get_datasets()
        models = self.get_models()
        
        print(f"\n📊 Testing {len(models)} models on {len(datasets)} datasets")
        print(f"⏱️  Estimated time: 5-10 minutes\n")
        
        # Print model lineup
        print("🏁 Model Lineup:")
        for i, model in enumerate(models, 1):
            print(f"   {i:2d}. {model['color']} {model['name']:20s} ({model['type']})")
        
        # Store all results
        all_results = {model['name']: {
            'accuracies': [], 'times': [], 'wins': 0, 'top3': 0, 
            'by_difficulty': {'Easy': [], 'Medium': [], 'Hard': [], 'Very Hard': []}
        } for model in models}
        
        # Results matrix for heatmap
        results_matrix = np.zeros((len(models), len(datasets)))
        
        for d_idx, dataset in enumerate(datasets):
            print(f"\n{'='*120}")
            print(f"📁 Dataset {d_idx+1}/10: {dataset['name']} ({dataset['difficulty']})")
            print(f"   Samples: {dataset['n_samples']:,} | Features: {dataset['n_features']} | Classes: {dataset['n_classes']}")
            print("-"*120)
            
            # Prepare data
            X_train, X_test, y_train, y_test = train_test_split(
                dataset['X'], dataset['y'], test_size=0.2, random_state=42, stratify=dataset['y']
            )
            
            # Standardize
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            dataset_results = []
            
            for m_idx, model in enumerate(models):
                print(f"{model['color']} {model['name']:20s}", end=' ')
                
                try:
                    # Training
                    start_time = time.time()
                    model['model'].fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    # Prediction
                    y_pred = model['model'].predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Store results
                    dataset_results.append({
                        'name': model['name'],
                        'accuracy': accuracy,
                        'time': train_time
                    })
                    
                    all_results[model['name']]['accuracies'].append(accuracy)
                    all_results[model['name']]['times'].append(train_time)
                    all_results[model['name']]['by_difficulty'][dataset['difficulty']].append(accuracy)
                    results_matrix[m_idx, d_idx] = accuracy
                    
                    print(f"| Acc: {accuracy:.4f} | Time: {train_time:6.3f}s")
                    
                except Exception as e:
                    print(f"| ✗ Failed: {str(e)[:30]}")
                    results_matrix[m_idx, d_idx] = 0
                    continue
            
            # Rankings for this dataset
            dataset_results.sort(key=lambda x: x['accuracy'], reverse=True)
            
            print(f"\n🏆 Top 3 for {dataset['name']}:")
            for i, result in enumerate(dataset_results[:3], 1):
                print(f"   {i}. {result['name']:20s} - {result['accuracy']:.4f}")
            
            # Track wins and top-3 finishes
            if dataset_results:
                all_results[dataset_results[0]['name']]['wins'] += 1
                for result in dataset_results[:3]:
                    all_results[result['name']]['top3'] += 1
        
        # Print comprehensive summary
        self.print_final_summary(all_results, datasets, models, results_matrix)
        
        # Return data for visualization
        return all_results, datasets, models, results_matrix
    
    def print_final_summary(self, all_results, datasets, models, results_matrix):
        """Print detailed final summary"""
        print(f"\n{'='*120}")
        print(" "*45 + "📈 FINAL RESULTS")
        print("="*120)
        
        # Overall Performance Table
        summary_data = []
        for model_name, stats in all_results.items():
            if stats['accuracies']:
                summary_data.append({
                    'name': model_name,
                    'avg_acc': np.mean(stats['accuracies']),
                    'std_acc': np.std(stats['accuracies']),
                    'median_acc': np.median(stats['accuracies']),
                    'min_acc': np.min(stats['accuracies']),
                    'max_acc': np.max(stats['accuracies']),
                    'avg_time': np.mean(stats['times']),
                    'wins': stats['wins'],
                    'top3': stats['top3']
                })
        
        summary_data.sort(key=lambda x: x['avg_acc'], reverse=True)
        
        print("\n🎯 OVERALL RANKINGS")
        print("-"*120)
        print(f"{'Rank':<6} {'Model':<20} {'Avg Accuracy':<15} {'Median':<10} {'Min':<10} {'Max':<10} {'Wins':<8} {'Top-3':<8} {'Avg Time':<10}")
        print("-"*120)
        
        for i, model in enumerate(summary_data, 1):
            medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'{i:2d}.'
            print(f"{medal:<6} {model['name']:<20} {model['avg_acc']:.4f}         "
                  f"{model['median_acc']:.4f}    {model['min_acc']:.4f}    {model['max_acc']:.4f}    "
                  f"{model['wins']}/10     {model['top3']}/30     {model['avg_time']:.3f}s")
        
        # Performance by Difficulty
        print("\n📊 PERFORMANCE BY DIFFICULTY")
        print("-"*120)
        print(f"{'Model':<20} {'Easy':<12} {'Medium':<12} {'Hard':<12} {'Very Hard':<12} {'Avg Delta':<12}")
        print("-"*120)
        
        for model in summary_data[:7]:  # Top 7
            name = model['name']
            stats = all_results[name]['by_difficulty']
            easy = np.mean(stats['Easy']) if stats['Easy'] else 0
            medium = np.mean(stats['Medium']) if stats['Medium'] else 0
            hard = np.mean(stats['Hard']) if stats['Hard'] else 0
            vhard = np.mean(stats['Very Hard']) if stats['Very Hard'] else 0
            
            # Calculate difficulty robustness (lower is better)
            accs = [a for a in [easy, medium, hard, vhard] if a > 0]
            delta = np.std(accs) if len(accs) > 1 else 0
            
            print(f"{name:<20} {easy:.4f}      {medium:.4f}      {hard:.4f}      {vhard:.4f}      {delta:.4f}")
        
        # Key Insights
        print("\n💡 KEY INSIGHTS")
        print("-"*120)
        
        winner = summary_data[0]
        print(f"🏆 Overall Winner: {winner['name']} with {winner['avg_acc']:.4f} average accuracy")
        print(f"   • Wins: {winner['wins']}/10 | Top-3: {winner['top3']}/30 | Avg Time: {winner['avg_time']:.3f}s")
        
        # MorphBoost analysis
        morph_stats = next((m for m in summary_data if m['name'] == 'MorphBoost'), None)
        if morph_stats:
            rank = next(i for i, m in enumerate(summary_data, 1) if m['name'] == 'MorphBoost')
            print(f"\n🚀 MorphBoost Performance:")
            print(f"   • Rank: {rank}/{len(summary_data)}")
            print(f"   • Average Accuracy: {morph_stats['avg_acc']:.4f} (σ={morph_stats['std_acc']:.4f})")
            print(f"   • Accuracy Range: [{morph_stats['min_acc']:.4f}, {morph_stats['max_acc']:.4f}]")
            print(f"   • Wins: {morph_stats['wins']}/10 datasets")
            print(f"   • Top-3 Finishes: {morph_stats['top3']}/30 possible")
            
            # Comparison with XGBoost
            xgb_stats = next((m for m in summary_data if m['name'] == 'XGBoost'), None)
            if xgb_stats:
                if morph_stats['avg_acc'] > xgb_stats['avg_acc']:
                    improvement = ((morph_stats['avg_acc'] - xgb_stats['avg_acc']) / xgb_stats['avg_acc']) * 100
                    print(f"   • ✅ Outperforms XGBoost by {improvement:.2f}%")
                else:
                    gap = ((xgb_stats['avg_acc'] - morph_stats['avg_acc']) / xgb_stats['avg_acc']) * 100
                    print(f"   • ❌ {gap:.2f}% behind XGBoost")
                
                speed_ratio = morph_stats['avg_time'] / xgb_stats['avg_time']
                print(f"   • Speed: {speed_ratio:.1f}x {'slower' if speed_ratio > 1 else 'faster'} than XGBoost")
            
            # Difficulty analysis
            morph_diff_stats = all_results['MorphBoost']['by_difficulty']
            easy_acc = np.mean(morph_diff_stats['Easy']) if morph_diff_stats['Easy'] else 0
            hard_acc = np.mean(morph_diff_stats['Very Hard']) if morph_diff_stats['Very Hard'] else 0
            if easy_acc > 0 and hard_acc > 0:
                degradation = ((easy_acc - hard_acc) / easy_acc) * 100
                print(f"   • Difficulty Degradation: {degradation:.1f}% (Easy→Very Hard)")
        
        # Best performers by category
        print("\n🏅 CATEGORY WINNERS:")
        print(f"   • Highest Accuracy: {summary_data[0]['name']} ({summary_data[0]['avg_acc']:.4f})")
        print(f"   • Most Consistent: {min(summary_data, key=lambda x: x['std_acc'])['name']} (σ={min(summary_data, key=lambda x: x['std_acc'])['std_acc']:.4f})")
        print(f"   • Most Robust: {max(summary_data, key=lambda x: x['min_acc'])['name']} (min_acc={max(summary_data, key=lambda x: x['min_acc'])['min_acc']:.4f})")
        print(f"   • Fastest: {min(summary_data, key=lambda x: x['avg_time'])['name']} ({min(summary_data, key=lambda x: x['avg_time'])['avg_time']:.3f}s)")
        print(f"   • Most Wins: {max(summary_data, key=lambda x: x['wins'])['name']} ({max(summary_data, key=lambda x: x['wins'])['wins']} wins)")
        print(f"   • Most Top-3: {max(summary_data, key=lambda x: x['top3'])['name']} ({max(summary_data, key=lambda x: x['top3'])['top3']}/30)")
        
        # Performance-Speed Trade-off
        print("\n⚖️  PERFORMANCE-SPEED TRADE-OFF:")
        for model in summary_data[:5]:
            efficiency = model['avg_acc'] / (model['avg_time'] + 0.001)  # Accuracy per second
            print(f"   • {model['name']:20s}: {efficiency:.4f} acc/sec")
        
        print("\n" + "="*120)

def main():
    print("\n" + "="*120)
    print(" "*35 + "🚀 MORPHBOOST MEGA-BENCHMARK v2.0")
    print(" "*25 + "Testing 10 Competitive Models on 10 Diverse Datasets")
    print("="*120)
    
    benchmark = ModelBenchmark()
    
    start_time = time.time()
    all_results, datasets, models, results_matrix = benchmark.run_benchmark()
    total_time = time.time() - start_time
    
    print(f"\n✅ Benchmark complete! Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    # Generate visualizations
    try:
        from visualization import BenchmarkVisualizer
        print("\n" + "="*120)
        visualizer = BenchmarkVisualizer(output_dir='benchmark_results')
        visualizer.create_all_visualizations(all_results, datasets, models, results_matrix)
    except ImportError:
        print("\n⚠️  visualization.py not found. Skipping visualization generation.")
        print("   Create visualization.py in the same directory to enable visualizations.")
    except Exception as e:
        print(f"\n⚠️  Error generating visualizations: {e}")
        print("   Continuing without visualizations...")
    
    print("="*120)

if __name__ == "__main__":
    main()