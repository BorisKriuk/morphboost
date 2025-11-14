import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MorphNode:
    """Self-morphing tree node with adaptive architecture"""
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['MorphNode'] = None
    right: Optional['MorphNode'] = None
    value: Optional[float] = None
    depth: int = 0
    weight: float = 1.0
    morph_score: float = 0.0
    split_type: str = 'adaptive'
    interaction_features: List[int] = field(default_factory=list)
    gradient_history: List[float] = field(default_factory=list)
    can_morph: bool = True
    subtask_id: int = 0
    information_gain: float = 0.0
    
class MorphBoost(BaseEstimator):
    """
    MorphBoost: Self-Organizing Universal Gradient Boosting
    Fixed for multiclass support while maintaining core structure
    """
    
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.3,
                 max_depth=None,
                 min_samples_split=20,
                 min_samples_leaf=10,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 reg_alpha=0.0,
                 reg_lambda=1.0,
                 morph_rate=0.1,
                 evolution_pressure=0.2,
                 interaction_detection=True,
                 auto_morphing=True,
                 adaptive_learning=True,
                 quantum_splits=False,
                 neural_embeddings=False,
                 fast_mode=True,
                 random_state=None):
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.morph_rate = morph_rate
        self.evolution_pressure = evolution_pressure
        self.interaction_detection = interaction_detection
        self.auto_morphing = auto_morphing
        self.adaptive_learning = adaptive_learning
        self.quantum_splits = quantum_splits
        self.neural_embeddings = neural_embeddings
        self.fast_mode = fast_mode
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.trees_ = []
        self.feature_importances_ = None
        self.morphing_history_ = []
        self._split_cache = {}
        self._interaction_map = {}
        self._gradient_stats = {'mean': 0, 'std': 1, 'skew': 0, 'kurt': 0}
        self._hessian_stats = {'mean': 1, 'std': 0.1}
        self._best_iteration = 0
        self._problem_fingerprint = None
        self._feature_embeddings = None
        self._meta_learner = None
        self._boosting_rate = learning_rate
        
    def _detect_problem_structure(self, X, y):
        """Detect problem structure and complexity - OPTIMIZED"""
        n_samples, n_features = X.shape
        
        y_unique = np.unique(y)
        n_unique = len(y_unique)
        
        if n_unique < 0.05 * n_samples and n_unique < 20:
            problem_type = 'classification'
            if n_unique == 2:
                problem_type = 'binary'
            else:
                problem_type = 'multiclass'  # FIX: Properly detect multiclass
        else:
            problem_type = 'regression'
        
        # Fast mode: simplified complexity detection    
        if self.fast_mode:
            complexity = 0.3
            non_linearity = 0.2
            interaction_strength = 0.15 if self.interaction_detection else 0
            noise_level = 0.1
        else:
            feature_stds = np.std(X, axis=0)
            feature_ranges = np.ptp(X, axis=0)
            complexity = np.mean(feature_stds / (feature_ranges + 1e-10))
            
            non_linearity = 0
            for i in range(min(5, n_features)):
                try:
                    corr_linear = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                    corr_quad = np.abs(np.corrcoef(X[:, i]**2, y)[0, 1])
                    non_linearity = max(non_linearity, corr_quad - corr_linear)
                except:
                    pass
            
            interaction_strength = 0
            if self.interaction_detection and n_features < 100:
                for i in range(min(5, n_features)):
                    for j in range(i+1, min(5, n_features)):
                        if n_samples > 100:
                            samples = np.random.choice(n_samples, 50, replace=False)
                            interaction = np.abs(np.corrcoef(X[samples, i] * X[samples, j], y[samples])[0, 1])
                            interaction_strength = max(interaction_strength, interaction)
            
            noise_level = 0.1
        
        self._problem_fingerprint = {
            'type': problem_type,
            'n_unique': n_unique,
            'complexity': complexity,
            'non_linearity': non_linearity,
            'interaction_strength': interaction_strength,
            'n_samples': n_samples,
            'n_features': n_features,
            'sparsity': 0,
            'noise_level': noise_level
        }
        
        return problem_type
    
    def _morph_split_function(self, gradients, hessians, features, iteration):
        """Morphing split function - OPTIMIZED"""
        
        if self.fast_mode or iteration < 5:
            score = gradients**2 / (hessians + self.reg_lambda)
        else:
            alpha = 0.05
            self._gradient_stats['mean'] = (1-alpha) * self._gradient_stats['mean'] + alpha * np.mean(gradients)
            self._gradient_stats['std'] = (1-alpha) * self._gradient_stats['std'] + alpha * np.std(gradients)
            
            g_mean = self._gradient_stats['mean']
            g_std = self._gradient_stats['std'] + 1e-10
            
            normalized_g = (gradients - g_mean) / g_std
            
            gradient_score = gradients**2 / (hessians + self.reg_lambda)
            
            info_smoothing = 1.0 + self.evolution_pressure * iteration / self.n_estimators
            info_score = np.abs(normalized_g) * np.log1p(np.abs(gradients) + 1e-10) / info_smoothing
            
            morph_weight = np.tanh(iteration / 20.0)
            
            score = 0.7 * gradient_score + 0.3 * info_score * morph_weight
                
        return score
    
    def _find_best_split_morphing(self, X, gradients, hessians, iteration):
        """Find best split - OPTIMIZED with vectorization"""
        n_samples, n_features = X.shape
        
        n_features_sample = max(1, int(self.colsample_bytree * n_features))
        
        if self.feature_importances_ is not None and iteration > 10 and not self.fast_mode:
            probs = self.feature_importances_ + 0.1
            probs = probs / np.sum(probs)
            feature_indices = np.random.choice(n_features, n_features_sample, 
                                             replace=False, p=probs)
        else:
            feature_indices = np.random.choice(n_features, n_features_sample, replace=False)
        
        best_score = -np.inf
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            
            sorted_idx = np.argsort(feature_values)
            sorted_features = feature_values[sorted_idx]
            sorted_gradients = gradients[sorted_idx]
            sorted_hessians = hessians[sorted_idx]
            
            unique_values, unique_indices = np.unique(sorted_features, return_index=True)
            
            if len(unique_values) <= 1:
                continue
            
            if len(unique_values) > 64 and self.fast_mode:
                thresholds_idx = np.linspace(0, len(unique_values)-2, min(16, len(unique_values)-1), dtype=int)
                thresholds = unique_values[thresholds_idx]
            elif len(unique_values) > 256:
                thresholds_idx = np.linspace(0, len(unique_values)-2, 32, dtype=int)
                thresholds = unique_values[thresholds_idx]
            else:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                left_g = np.sum(gradients[left_mask])
                left_h = np.sum(hessians[left_mask])
                right_g = np.sum(gradients[right_mask])
                right_h = np.sum(hessians[right_mask])
                
                gain = (left_g**2 / (left_h + self.reg_lambda) + 
                       right_g**2 / (right_h + self.reg_lambda) - 
                       (left_g + right_g)**2 / (left_h + right_h + self.reg_lambda))
                
                if not self.fast_mode and iteration >= 5:
                    complexity_penalty = self.reg_alpha * (1.0 + iteration / self.n_estimators)
                    balance_ratio = min(n_left, n_right) / n_samples
                    if balance_ratio < 0.1:
                        gain -= 0.5 * (1.0 - np.exp(-10 * balance_ratio))
                    gain -= complexity_penalty
                
                if gain > best_score:
                    best_score = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_idx = np.where(left_mask)[0]
                    best_right_idx = np.where(right_mask)[0]
        
        return best_feature, best_threshold, best_score, best_left_idx, best_right_idx
    
    def _build_morphing_tree(self, X, gradients, hessians, depth, iteration, sample_indices=None):
        """Build self-morphing tree - OPTIMIZED"""
        
        if sample_indices is None:
            sample_indices = np.arange(len(X))
        
        n_samples = len(sample_indices)
        
        if self.max_depth is None:
            if self.fast_mode:
                effective_max_depth = 8
            else:
                complexity = self._problem_fingerprint.get('complexity', 0.3)
                effective_max_depth = 10 if complexity > 0.5 else 8
        else:
            effective_max_depth = self.max_depth
        
        node = MorphNode(depth=depth)
        
        gradient_variance = np.var(gradients[sample_indices])
        stop_growing = (
            depth >= effective_max_depth or
            n_samples < self.min_samples_split or
            gradient_variance < 1e-7
        )
        
        if stop_growing:
            numerator = -np.sum(gradients[sample_indices])
            denominator = np.sum(hessians[sample_indices]) + self.reg_lambda
            
            depth_penalty = 0.9 ** (depth / 3)
            iteration_shrinkage = 1.0 - self.morph_rate * min(1.0, iteration / self.n_estimators)
            shrinkage = self.learning_rate * depth_penalty * iteration_shrinkage
            
            node.value = shrinkage * numerator / denominator
            node.weight = n_samples / len(X)
            
            if abs(node.value) > 10:
                node.value = np.sign(node.value) * 10
                
            return node
        
        best_feature, best_threshold, best_score, left_idx, right_idx = \
            self._find_best_split_morphing(X[sample_indices], gradients[sample_indices], 
                                          hessians[sample_indices], iteration)
        
        if best_feature is None:
            numerator = -np.sum(gradients[sample_indices])
            denominator = np.sum(hessians[sample_indices]) + self.reg_lambda
            node.value = self.learning_rate * numerator / denominator
            return node
        
        node.feature = best_feature
        node.threshold = best_threshold
        node.morph_score = best_score
        node.information_gain = best_score / (gradient_variance + 1e-10)
        
        if self.interaction_detection and depth < 3 and not self.fast_mode:
            left_samples = sample_indices[left_idx]
            if len(left_samples) > 30:
                correlations = []
                for f in range(min(5, X.shape[1])):
                    if f != best_feature:
                        mult_corr = np.abs(np.corrcoef(X[left_samples, best_feature] * X[left_samples, f], 
                                                       gradients[left_idx])[0, 1])
                        if not np.isnan(mult_corr):
                            correlations.append((f, mult_corr))
                
                if correlations:
                    correlations.sort(key=lambda x: x[1], reverse=True)
                    node.interaction_features = [f for f, _ in correlations[:2]]
        
        node.gradient_history = [np.mean(gradients[sample_indices[left_idx]]),
                                 np.mean(gradients[sample_indices[right_idx]])]
        
        node.left = self._build_morphing_tree(X, gradients, hessians, depth + 1, 
                                             iteration, sample_indices[left_idx])
        node.right = self._build_morphing_tree(X, gradients, hessians, depth + 1, 
                                              iteration, sample_indices[right_idx])
        
        return node
    
    def _predict_tree_vectorized(self, node, X):
        """Vectorized tree prediction - MUCH FASTER"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        n_samples = len(X)
        predictions = np.zeros(n_samples)
        
        node_samples = {id(node): np.arange(n_samples)}
        leaf_values = {}
        
        queue = [node]
        while queue:
            current = queue.pop(0)
            current_id = id(current)
            
            if current_id not in node_samples:
                continue
                
            sample_indices = node_samples[current_id]
            if len(sample_indices) == 0:
                continue
            
            if current.value is not None:
                leaf_values[current_id] = (sample_indices, current.value)
            elif current.feature is not None:
                feature_values = X[sample_indices, current.feature]
                left_mask = feature_values <= current.threshold
                right_mask = ~left_mask
                
                left_indices = sample_indices[left_mask]
                right_indices = sample_indices[right_mask]
                
                if current.left and len(left_indices) > 0:
                    node_samples[id(current.left)] = left_indices
                    queue.append(current.left)
                    
                if current.right and len(right_indices) > 0:
                    node_samples[id(current.right)] = right_indices
                    queue.append(current.right)
        
        for sample_indices, value in leaf_values.values():
            predictions[sample_indices] = value
        
        return predictions
    
    def _predict_tree(self, node, X):
        """Tree prediction - use vectorized version in fast mode"""
        if self.fast_mode:
            return self._predict_tree_vectorized(node, X)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        predictions = np.zeros(len(X))
        
        for i in range(len(X)):
            current = node
            while current is not None and current.value is None:
                if current.feature is None:
                    break
                    
                if X[i, current.feature] <= current.threshold:
                    current = current.left
                else:
                    current = current.right
            
            if current is not None and current.value is not None:
                predictions[i] = current.value
        
        return predictions
    
    def _calculate_feature_importance(self):
        """Calculate feature importance - OPTIMIZED"""
        n_features = self.n_features_
        importances = np.zeros(n_features)
        
        for tree_idx, tree in enumerate(self.trees_):
            # Check if this is a list of trees (multiclass) or single tree
            if isinstance(tree, list):
                for class_tree in tree:
                    tree_weight = 1.0 + 0.5 * tree_idx / len(self.trees_)
                    
                    def traverse(node, parent_weight=1.0):
                        if node is None or node.feature is None:
                            return
                        
                        node_importance = parent_weight * node.morph_score * node.information_gain
                        importances[node.feature] += node_importance * tree_weight
                        
                        child_weight = parent_weight * 0.9
                        if node.left:
                            traverse(node.left, child_weight)
                        if node.right:
                            traverse(node.right, child_weight)
                    
                    traverse(class_tree)
            else:
                tree_weight = 1.0 + 0.5 * tree_idx / len(self.trees_)
                
                def traverse(node, parent_weight=1.0):
                    if node is None or node.feature is None:
                        return
                    
                    node_importance = parent_weight * node.morph_score * node.information_gain
                    importances[node.feature] += node_importance * tree_weight
                    
                    if not self.fast_mode:
                        for interact_feat in node.interaction_features:
                            importances[interact_feat] += node_importance * 0.3
                    
                    child_weight = parent_weight * 0.9
                    if node.left:
                        traverse(node.left, child_weight)
                    if node.right:
                        traverse(node.right, child_weight)
                
                traverse(tree)
        
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        return importances
    
    def _generate_learning_schedule(self):
        """Generate adaptive learning rate schedule - SIMPLIFIED"""
        if self.fast_mode:
            return [self.learning_rate] * self.n_estimators
        
        schedules = []
        
        warm_up = min(10, self.n_estimators // 10)
        for i in range(warm_up):
            schedules.append(self.learning_rate * (i + 1) / warm_up)
        
        remaining = self.n_estimators - warm_up
        for i in range(remaining):
            progress = i / remaining
            lr = self.learning_rate * (0.5 + 0.5 * np.cos(np.pi * progress))
            schedules.append(max(lr, self.learning_rate * 0.01))
        
        return schedules
    
    def _get_tree_depth(self, node):
        """Get maximum depth of tree"""
        if node is None or node.value is not None:
            return 0
        return 1 + max(self._get_tree_depth(node.left), 
                      self._get_tree_depth(node.right))
    
    def fit(self, X, y, sample_weight=None, eval_set=None, early_stopping_rounds=None):
        """Fit MorphBoost - FIXED FOR MULTICLASS"""
        
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        problem_type = self._detect_problem_structure(X, y)
        
        self._label_encoder = None
        if problem_type in ['classification', 'binary', 'multiclass']:
            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(y)
            self.n_classes_ = len(self._label_encoder.classes_)
            
            if self.n_classes_ == 2:
                y = 2.0 * y_encoded - 1.0
                pos_ratio = np.mean((y + 1) / 2)
                self.init_score_ = np.log(pos_ratio / (1 - pos_ratio + 1e-10))
                predictions = np.full(n_samples, self.init_score_)
            else:
                # MULTICLASS FIX: Use one-vs-rest with separate trees
                y = y_encoded
                self.init_score_ = np.zeros(self.n_classes_)
                predictions = np.zeros((n_samples, self.n_classes_))
                # Initialize with log odds
                for k in range(self.n_classes_):
                    class_ratio = np.mean(y == k)
                    self.init_score_[k] = np.log(class_ratio / (1 - class_ratio + 1e-10))
                    predictions[:, k] = self.init_score_[k]
        else:
            self.init_score_ = np.median(y)
            predictions = np.full(n_samples, self.init_score_)
        
        self._gradient_momentum = np.zeros(n_samples)
        
        self.trees_ = []
        best_score = np.inf
        rounds_without_improvement = 0
        
        if self.adaptive_learning:
            learning_rates = self._generate_learning_schedule()
        else:
            learning_rates = [self.learning_rate] * self.n_estimators
        
        for iteration in range(self.n_estimators):
            current_lr = learning_rates[iteration]
            self._boosting_rate = current_lr
            
            X_iter = X
            if self.neural_embeddings and not self.fast_mode and iteration > 10:
                X_iter = self._neural_embedding_transform(X, iteration)
            
            if problem_type == 'binary':
                predictions_clipped = np.clip(predictions, -10, 10)
                p = 1.0 / (1.0 + np.exp(-predictions_clipped))
                gradients = p - (y + 1) / 2
                hessians = p * (1 - p)
                hessians = np.maximum(hessians, 1e-8)
                
                # Build single tree for binary
                if self.subsample < 1.0:
                    sample_indices = np.random.choice(n_samples, 
                                                    int(self.subsample * n_samples), 
                                                    replace=False)
                else:
                    sample_indices = np.arange(n_samples)
                
                tree = self._build_morphing_tree(X_iter, gradients, hessians, 
                                                depth=0, iteration=iteration, 
                                                sample_indices=sample_indices)
                
                tree_predictions = self._predict_tree(tree, X_iter)
                predictions += tree_predictions
                self.trees_.append(tree)
                
            elif problem_type == 'multiclass':
                # MULTICLASS FIX: Build one tree per class (one-vs-rest)
                iteration_trees = []
                
                # Compute softmax probabilities
                exp_pred = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
                probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
                
                for k in range(self.n_classes_):
                    # One-vs-rest gradients and hessians
                    y_k = (y == k).astype(float)
                    p_k = probs[:, k]
                    gradients = p_k - y_k
                    hessians = p_k * (1 - p_k)
                    hessians = np.maximum(hessians, 1e-8)
                    
                    # Subsample
                    if self.subsample < 1.0:
                        sample_indices = np.random.choice(n_samples, 
                                                        int(self.subsample * n_samples), 
                                                        replace=False)
                    else:
                        sample_indices = np.arange(n_samples)
                    
                    # Build tree for this class
                    tree = self._build_morphing_tree(X_iter, gradients, hessians, 
                                                    depth=0, iteration=iteration, 
                                                    sample_indices=sample_indices)
                    
                    # Update predictions for this class
                    tree_predictions = self._predict_tree(tree, X_iter)
                    predictions[:, k] += tree_predictions
                    iteration_trees.append(tree)
                
                self.trees_.append(iteration_trees)
                
            else:
                # Regression
                residuals = predictions - y
                
                if self.fast_mode:
                    gradients = residuals
                    hessians = np.ones(n_samples)
                else:
                    delta = 1.35 * np.std(residuals)
                    mask = np.abs(residuals) <= delta
                    gradients = np.where(mask, residuals, delta * np.sign(residuals))
                    hessians = np.where(mask, np.ones(n_samples), np.zeros(n_samples)) + 1e-8
                
                if self.subsample < 1.0:
                    sample_indices = np.random.choice(n_samples, 
                                                    int(self.subsample * n_samples), 
                                                    replace=False)
                else:
                    sample_indices = np.arange(n_samples)
                
                tree = self._build_morphing_tree(X_iter, gradients, hessians, 
                                                depth=0, iteration=iteration, 
                                                sample_indices=sample_indices)
                
                tree_predictions = self._predict_tree(tree, X_iter)
                predictions += tree_predictions
                self.trees_.append(tree)
            
            # Early stopping
            if eval_set is not None and early_stopping_rounds is not None:
                X_val, y_val = eval_set[0]
                val_predictions = self.predict(X_val)
                
                if problem_type in ['binary', 'multiclass']:
                    val_score = np.mean((val_predictions - y_val) ** 2)
                else:
                    val_score = np.mean((val_predictions - y_val) ** 2)
                
                if val_score < best_score:
                    best_score = val_score
                    self._best_iteration = iteration
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                
                if rounds_without_improvement >= early_stopping_rounds:
                    self.trees_ = self.trees_[:self._best_iteration+1]
                    break
            
            if self.auto_morphing and not self.fast_mode and iteration > 0 and iteration % 10 == 0:
                if problem_type == 'multiclass':
                    train_loss = np.mean([np.mean(g**2) for g in [probs[:, k] - (y == k).astype(float) 
                                                                   for k in range(self.n_classes_)]])
                else:
                    train_loss = np.mean(gradients ** 2)
                    
                morph_info = {
                    'iteration': iteration,
                    'train_loss': train_loss,
                    'learning_rate': current_lr,
                    'avg_tree_depth': 5.0  # Simplified
                }
                self.morphing_history_.append(morph_info)
        
        self.feature_importances_ = self._calculate_feature_importance()
        
        return self
    
    def predict(self, X):
        """Make predictions - FIXED FOR MULTICLASS"""
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        if self.neural_embeddings and not self.fast_mode and self._feature_embeddings is not None:
            X = self._neural_embedding_transform(X, len(self.trees_))
        
        if self._label_encoder is not None:
            if self.n_classes_ == 2:
                # Binary classification
                predictions = np.full(len(X), self.init_score_)
                
                for tree in self.trees_:
                    predictions += self._predict_tree(tree, X)
                
                predictions = 1.0 / (1.0 + np.exp(-np.clip(predictions, -10, 10)))
                predictions = (predictions > 0.5).astype(int)
                
            else:
                # MULTICLASS FIX: Aggregate predictions from all trees
                predictions = np.zeros((len(X), self.n_classes_))
                for k in range(self.n_classes_):
                    predictions[:, k] = self.init_score_[k]
                
                for iteration_trees in self.trees_:
                    for k, tree in enumerate(iteration_trees):
                        predictions[:, k] += self._predict_tree(tree, X)
                
                # Use argmax to get class predictions
                predictions = np.argmax(predictions, axis=1)
            
            predictions = self._label_encoder.inverse_transform(predictions)
        else:
            # Regression
            predictions = np.full(len(X), self.init_score_)
            
            for tree in self.trees_:
                predictions += self._predict_tree(tree, X)
        
        return predictions
    
    def predict_proba(self, X):
        """Predict probabilities for classification"""
        if self._label_encoder is None:
            raise ValueError("predict_proba only available for classification")
        
        X = check_array(X, accept_sparse=False, dtype=np.float32)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        if self.neural_embeddings and not self.fast_mode and self._feature_embeddings is not None:
            X = self._neural_embedding_transform(X, len(self.trees_))
        
        if self.n_classes_ == 2:
            raw_predictions = np.full(len(X), self.init_score_)
            
            for tree in self.trees_:
                raw_predictions += self._predict_tree(tree, X)
            
            proba = 1.0 / (1.0 + np.exp(-np.clip(raw_predictions, -10, 10)))
            return np.column_stack([1 - proba, proba])
        else:
            # MULTICLASS FIX: Compute softmax probabilities
            predictions = np.zeros((len(X), self.n_classes_))
            for k in range(self.n_classes_):
                predictions[:, k] = self.init_score_[k]
            
            for iteration_trees in self.trees_:
                for k, tree in enumerate(iteration_trees):
                    predictions[:, k] += self._predict_tree(tree, X)
            
            # Softmax
            exp_pred = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            probas = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
            
            return probas
    
    def _neural_embedding_transform(self, X, iteration):
        """Neural-inspired feature transformation"""
        if self._feature_embeddings is None:
            n_features = X.shape[1]
            n_embeddings = min(20, n_features)
            self._feature_embeddings = np.random.randn(n_features, n_embeddings) * 0.1
        
        embedded = np.tanh(X @ self._feature_embeddings)
        
        if iteration > 0 and iteration % 10 == 0:
            learning_rate = 0.01 / (1 + iteration / 100)
            noise = np.random.randn(*self._feature_embeddings.shape) * learning_rate
            self._feature_embeddings += noise
            
        return np.hstack([X, embedded])