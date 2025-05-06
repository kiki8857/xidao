"""
增强版随机森林模型
包含特征重要性分析、特征选择、交叉验证等增强功能
"""

from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel, RFECV, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import os
from joblib import dump, load

logger = logging.getLogger(__name__)

def build_enhanced_random_forest(config, random_state=None):
    """构建一个增强版随机森林回归模型。
    
    Args:
        config (dict): 配置字典，应包含 models.random_forest 下的参数
        random_state (int, optional): 用于复现的随机种子
        
    Returns:
        RandomForestRegressor: 一个配置好的随机森林模型实例
    """
    rf_config = config.get('models', {}).get('random_forest', {})
    
    # 提取参数
    n_estimators = rf_config.get('n_estimators', 500)
    max_depth = rf_config.get('max_depth', 20)
    min_samples_split = rf_config.get('min_samples_split', 3)
    min_samples_leaf = rf_config.get('min_samples_leaf', 2)
    max_features = rf_config.get('max_features', 'sqrt')
    criterion = rf_config.get('criterion', 'squared_error')
    bootstrap = rf_config.get('bootstrap', True)
    max_samples = rf_config.get('max_samples', 0.8)
    oob_score = rf_config.get('oob_score', True)
    
    logger.info(f"Building Enhanced RandomForestRegressor with: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, max_features={max_features}, criterion={criterion}, bootstrap={bootstrap}, max_samples={max_samples}, oob_score={oob_score}")
    
    try:
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth if max_depth is None else int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            criterion=criterion,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=-1,
            max_samples=float(max_samples) if bootstrap else None,
            oob_score=oob_score if bootstrap else False
        )
        return model
    except Exception as e:
        logger.error(f"Failed to build Enhanced RandomForestRegressor: {e}", exc_info=True)
        return None

def select_features_with_importance(model, X, y, threshold=0.01, results_dir=None):
    """基于特征重要性选择特征
    
    Args:
        model (RandomForestRegressor): 已训练的随机森林模型
        X (DataFrame): 特征数据
        y (Series): 目标变量
        threshold (float): 重要性阈值
        results_dir (str): 结果保存目录
        
    Returns:
        tuple: (特征子集, 特征重要性)
    """
    try:
        # 拟合模型
        if not hasattr(model, 'feature_importances_'):
            model.fit(X, y)
            
        # 获取特征重要性
        feature_importances = model.feature_importances_
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"特征重要性排序:\n{importance_df}")
        
        # 选择重要特征
        selector = SelectFromModel(model, threshold=threshold, prefit=True)
        X_selected = selector.transform(X)
        selected_features = X.columns[selector.get_support()]
        
        logger.info(f"选择了 {len(selected_features)}/{X.shape[1]} 个特征: {list(selected_features)}")
        
        # 绘制特征重要性图
        if results_dir:
            plt.figure(figsize=(12, 8))
            plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
            plt.xlabel('重要性')
            plt.ylabel('特征')
            plt.title('随机森林特征重要性 (前15个)')
            plt.tight_layout()
            
            # 确保目录存在
            viz_dir = os.path.join(results_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            plt.savefig(os.path.join(viz_dir, 'feature_importance.png'))
            plt.close()
            
            # 保存特征重要性
            importance_df.to_csv(os.path.join(results_dir, 'feature_importance.csv'), index=False)
        
        return X_selected, selected_features, importance_df
    except Exception as e:
        logger.error(f"Error in feature importance selection: {e}", exc_info=True)
        return X, X.columns, None

def recursive_feature_elimination(model, X, y, n_features=None, step=1, cv=5, results_dir=None):
    """使用递归特征消除法选择特征
    
    Args:
        model (RandomForestRegressor): 随机森林模型
        X (DataFrame): 特征数据
        y (Series): 目标变量
        n_features (int, optional): 要选择的特征数量
        step (int): 每次消除的特征数量
        cv (int): 交叉验证折数
        results_dir (str): 结果保存目录
        
    Returns:
        tuple: (特征子集, 选择的特征)
    """
    try:
        # 如果未指定特征数量，使用交叉验证找到最佳数量
        if n_features is None:
            rfe = RFECV(estimator=model, step=step, cv=cv, scoring='neg_mean_absolute_error', min_features_to_select=3, n_jobs=-1)
            rfe.fit(X, y)
            logger.info(f"RFECV最佳特征数量: {rfe.n_features_}")
            
            if results_dir:
                # 绘制交叉验证分数曲线
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(rfe.cv_results_['mean_test_score']) + 1), -rfe.cv_results_['mean_test_score'])
                plt.xlabel('特征数量')
                plt.ylabel('MAE (CV)')
                plt.title('递归特征消除交叉验证')
                
                viz_dir = os.path.join(results_dir, 'visualizations')
                os.makedirs(viz_dir, exist_ok=True)
                plt.savefig(os.path.join(viz_dir, 'rfe_cv_scores.png'))
                plt.close()
        else:
            rfe = RFE(estimator=model, n_features_to_select=n_features, step=step)
            rfe.fit(X, y)
            
        # 获取选择的特征
        selected_features = X.columns[rfe.support_]
        logger.info(f"RFE选择的特征 ({len(selected_features)}): {list(selected_features)}")
        
        # 转换数据
        X_selected = rfe.transform(X)
        
        if results_dir:
            # 保存RFE结果
            pd.DataFrame({
                'feature': X.columns,
                'selected': rfe.support_,
                'ranking': rfe.ranking_
            }).sort_values('ranking').to_csv(os.path.join(results_dir, 'rfe_results.csv'), index=False)
        
        return X_selected, selected_features
    except Exception as e:
        logger.error(f"Error in recursive feature elimination: {e}", exc_info=True)
        return X, X.columns

def random_forest_grid_search(X, y, param_grid, cv=5, n_jobs=-1, results_dir=None, scoring='neg_mean_absolute_error'):
    """使用网格搜索优化随机森林超参数
    
    Args:
        X (DataFrame): 特征数据
        y (Series): 目标变量
        param_grid (dict): 参数网格
        cv (int): 交叉验证折数
        n_jobs (int): 并行作业数
        results_dir (str): 结果保存目录
        scoring (str): 评分指标，默认为负MAE
        
    Returns:
        RandomForestRegressor: 优化后的模型
    """
    try:
        # 创建基础模型
        base_model = RandomForestRegressor(random_state=42)
        
        # 创建网格搜索
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            verbose=1,
            return_train_score=True
        )
        
        # 拟合数据
        grid_search.fit(X, y)
        
        # 输出最佳参数
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳{scoring}分数: {grid_search.best_score_:.4f}")
        
        if results_dir:
            # 保存网格搜索结果
            results = pd.DataFrame(grid_search.cv_results_)
            results.to_csv(os.path.join(results_dir, 'grid_search_results.csv'), index=False)
            
            # 保存最佳模型
            dump(grid_search.best_estimator_, os.path.join(results_dir, 'best_rf_model.joblib'))
            
            # 保存最佳参数
            pd.DataFrame([grid_search.best_params_]).to_csv(os.path.join(results_dir, 'best_params.csv'), index=False)
            
            # 可视化参数影响（对于部分参数）
            try:
                plt.figure(figsize=(15, 10))
                
                # 找出有多个值的参数
                params_to_plot = [param for param, values in param_grid.items() if len(values) > 1]
                n_params = len(params_to_plot)
                
                if n_params > 0:
                    # 创建子图
                    fig, axes = plt.subplots(1, n_params, figsize=(n_params*5, 5))
                    if n_params == 1:
                        axes = [axes]  # 确保axes始终是列表
                    
                    for i, param in enumerate(params_to_plot):
                        # 从结果中提取此参数的数据
                        param_values = results['param_' + param].astype(str)
                        unique_values = param_values.unique()
                        
                        # 按参数值分组并绘制分数箱形图
                        scores = []
                        for val in unique_values:
                            mask = param_values == val
                            scores.append(results.loc[mask, 'mean_test_score'].values)
                        
                        # 绘制箱形图
                        axes[i].boxplot(scores, labels=unique_values)
                        axes[i].set_title(f'参数 {param} 对模型性能的影响')
                        axes[i].set_xlabel(param)
                        axes[i].set_ylabel(scoring)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, 'parameter_impact.png'))
                    plt.close()
            except Exception as e:
                logger.warning(f"无法创建参数影响可视化: {e}")
        
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error in random forest grid search: {e}", exc_info=True)
        return None

def build_forest_ensemble(config, random_state=None):
    """构建随机森林集成模型（支持stacking和voting）
    
    Args:
        config (dict): 配置字典，包含模型和集成设置
        random_state (int, optional): 随机种子
        
    Returns:
        object: 集成模型（StackingRegressor或VotingRegressor）
    """
    ensemble_config = config.get('models', {}).get('ensemble', {})
    
    if not ensemble_config.get('enabled', False):
        logger.info("集成学习未启用，返回标准随机森林模型")
        return build_enhanced_random_forest(config, random_state)
    
    # 获取集成方法
    method = ensemble_config.get('method', 'stacking')
    logger.info(f"创建{method}集成模型")
    
    # 创建基础模型
    base_models = []
    base_model_configs = ensemble_config.get('base_models', [])
    
    if not base_model_configs:
        # 如果没有指定基础模型，使用默认配置创建三个不同参数的随机森林
        rf_config = config.get('models', {}).get('random_forest', {})
        n_estimators = rf_config.get('n_estimators', 300)
        max_depth = rf_config.get('max_depth', 25)
        
        base_model_configs = [
            {'type': 'random_forest', 'n_estimators': n_estimators, 'max_depth': max_depth},
            {'type': 'random_forest', 'n_estimators': n_estimators + 200, 'max_depth': max_depth - 5},
            {'type': 'random_forest', 'n_estimators': n_estimators + 400, 'max_depth': max_depth - 10}
        ]
    
    # 为每个基础模型创建克隆配置
    for i, model_config in enumerate(base_model_configs):
        model_type = model_config.get('type', 'random_forest')
        
        if model_type == 'random_forest':
            # 克隆主配置并更新特定参数
            model_specific_config = {
                'models': {
                    'random_forest': {
                        'n_estimators': model_config.get('n_estimators', 300),
                        'max_depth': model_config.get('max_depth', 25),
                        'min_samples_split': model_config.get('min_samples_split', 2),
                        'min_samples_leaf': model_config.get('min_samples_leaf', 1),
                        'max_features': model_config.get('max_features', 'sqrt'),
                        'bootstrap': True,
                        'random_state': random_state + i if random_state is not None else None
                    }
                }
            }
            
            # 创建随机森林并添加到基础模型列表
            rf_model = build_enhanced_random_forest(model_specific_config, random_state + i if random_state is not None else None)
            base_models.append((f'rf_{i}', rf_model))
    
    # 根据指定的方法创建集成
    if method == 'voting':
        # 获取权重（如果有）
        weights = ensemble_config.get('weights', None)
        ensemble = VotingRegressor(estimators=base_models, weights=weights, n_jobs=-1)
        logger.info(f"创建了VotingRegressor，包含{len(base_models)}个基础模型")
        return ensemble
    else:  # stacking
        # 创建元模型
        meta_model_type = ensemble_config.get('meta_model', 'linear')
        if meta_model_type == 'linear':
            meta_model = LinearRegression()
        elif meta_model_type == 'ridge':
            meta_model = Ridge(alpha=1.0, random_state=random_state)
        else:
            # 默认也使用线性回归
            meta_model = LinearRegression()
        
        # 创建Stacking集成
        ensemble = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        logger.info(f"创建了StackingRegressor，包含{len(base_models)}个基础模型，元模型为{meta_model_type}")
        return ensemble

def create_rf_ensemble(models, X, y, weights=None):
    """创建随机森林集成模型
    
    Args:
        models (list): 模型列表
        X (DataFrame): 特征数据
        y (Series): 目标变量
        weights (list, optional): 模型权重
        
    Returns:
        callable: 集成预测函数
    """
    # 如果未提供权重，则使用均等权重
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # 标准化权重
    weights = np.array(weights) / sum(weights)
    
    # 定义集成预测函数
    def ensemble_predict(X_test):
        predictions = np.zeros(X_test.shape[0])
        for i, model in enumerate(models):
            predictions += weights[i] * model.predict(X_test)
        return predictions
    
    return ensemble_predict 