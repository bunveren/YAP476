# YAP476 Data Mining Project
## Instacart Multilevel Mining



# imports & setup
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
from IPython.display import display
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='mlxtend')

plt.style.use('seaborn-v0_8-muted')
sns.set_theme(style="whitegrid")
def get_memory_usage(df): return df.memory_usage(deep=True).sum() / 1024**2



### Loading the datasets, casting appropriate data types for memory optimization, and merging
products = pd.read_csv('products.csv')
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
order_products = pd.read_csv('order_products__prior.csv', nrows = 2e6)

aisles['aisle'] = aisles['aisle'].astype('category')
departments['department'] = departments['department'].astype('category')
products['product_name'] = products['product_name'].astype('category')

df_merged = order_products.merge(products, on='product_id', how='left')
df_merged = df_merged.merge(aisles, on='aisle_id', how='left')
df_merged = df_merged.merge(departments, on='department_id', how='left')

orders = pd.read_csv('orders.csv')
orders = orders[['order_id', 'order_dow', 'order_hour_of_day']] # order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
df_merged = df_merged.merge(orders, on='order_id', how='left')
for col in ['order_dow', 'order_hour_of_day']: df_merged[col] = df_merged[col].astype('int8')




def prepare_multidimensional_data(df):
    bins = [0, 6, 12, 18, 24]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df['hour_bin'] = pd.cut(df['order_hour_of_day'], bins=bins, labels=labels, right=False)
    df['hour_bin'] = 'Hour_' + df['hour_bin'].astype(str)
    df['day_type'] = df['order_dow'].apply(lambda x: 'Weekend' if x in [0, 1] else 'Weekday')
    basket_counts = df.groupby('order_id')['product_id'].transform('count')
    df['basket_size'] = pd.cut(basket_counts, bins=[0, 5, 15, 100], labels=['Small_Basket', 'Medium_Basket', 'Large_Basket'])
    return df
df_merged = prepare_multidimensional_data(df_merged)


print("shape:", df_merged.shape)
print("\nattributes:", df_merged.columns.tolist())
print("\nbasic stats for numeric columns:")
display(df_merged.describe())

def opt(df):
    for col in df.columns:
        if df[col].dtype == 'int64': df[col] = df[col].astype('int32')
        elif df[col].dtype == 'float64': df[col] = df[col].astype('float32')
    return df

df_merged = opt(df_merged)




#prepping 'understanding data' section in report
plt.figure(figsize=(12, 6))
basket_size = df_merged.groupby('order_id').size()
sns.histplot(
    basket_size,
    bins=50,
    kde=True,
    #color=''
)
plt.xlabel('Number of Items')
plt.ylabel('Number of Orders')
plt.xlim(0, 57)
#plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
df_merged['product_name'].value_counts()[:10].plot(kind='bar', title='Top 10 Products')
plt.subplot(1, 3, 2)
df_merged['aisle'].value_counts()[:10].plot(kind='bar', title='Top 10 Aisles')
plt.subplot(1, 3, 3)
df_merged['department'].value_counts()[:10].plot(kind='bar', title='Top 10 Departments')
plt.tight_layout()
#plt.show()




### Transaction Encoding
def get_basket(df, level_column, is_sparse=True):
    df = df.dropna(subset=[level_column, 'hour_bin', 'day_type', 'basket_size'])
    
    baskets = df.groupby('order_id')[level_column].unique().reset_index()
    meta = df.drop_duplicates('order_id')[['order_id', 'hour_bin', 'day_type', 'basket_size']]
    merged = baskets.merge(meta, on='order_id')
    
    items_list = merged[level_column].tolist()
    meta_cols = ['hour_bin', 'day_type', 'basket_size']
    meta_list = merged[meta_cols].fillna('Unknown').astype(str).values.tolist()
    
    transaction_list = [list(items) + meta for items, meta in zip(items_list, meta_list)]
    te = TransactionEncoder()
    te_ary = te.fit(transaction_list).transform(transaction_list, sparse=is_sparse)
    
    if is_sparse: 
        return pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    return pd.DataFrame(te_ary, columns=te.columns_)



### Algorithm Comparison TODO APRIORI FOR BENCHMARK
def run_mining_experiment(dimensions=['Weekend', 'Weekday'], levels=['department', 'aisle', 'product_name'], supports=[0.05, 0.02, 0.01]):
    results = []
    all_rules = {}
    level_supports = { # TODO CHANGE ACC TO METHOD PARAMS
        'department': [0.2, 0.1],
        'aisle': [0.05, 0.02],
        'product_name': [0.01, 0.005]
    }
    for dim in dimensions:
        print(f"\n=== Processing Dimension: {dim} ===")
        df_subset = df_merged[df_merged['day_type'] == dim]
        
        for level in levels:
            print(f"--- Mining Level: {level} ---")
            basket_df = get_basket(df_subset, level)
            mem_usage = basket_df.memory_usage(deep=True).sum() / 1024**2
            current_supports = level_supports.get(level, [0.05])
            for min_sup in current_supports:
                try:
                    start_time = time.time()
                    freq_items = fpgrowth(basket_df, min_support=min_sup, use_colnames=True)
                    exec_time = time.time() - start_time
                    
                    rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
                    if not rules.empty:
                        rules['kulczynski'] = (rules['support']/rules['antecedent support'] + rules['support']/rules['consequent support']) / 2
                        rules['ir'] = np.abs(rules['antecedent support'] - rules['consequent support']) / (rules['antecedent support'] + rules['consequent support'] - rules['support'])
                        robust_count = len(rules[(rules['kulczynski'] > 0.4) & (rules['ir'] < 0.6)])
                    else:
                        robust_count = 0
                        
                    all_rules[(dim, level, min_sup)] = rules
                    results.append({
                        'Dimension': dim, 'Level': level, 'Support': min_sup, 
                        'Algorithm': 'fpgrowth', 'Time': exec_time, 
                        'Memory_MB': mem_usage, 'Robust_Rules': robust_count
                    })
                    print(f"  Sup: {min_sup} | Rules Found: {len(rules)} | Time: {exec_time:.2f}s")
                except Exception as e:
                    print(f"  Error at {dim}-{level}-{min_sup}: {e}")
                
    return pd.DataFrame(results), all_rules
benchmark_df, all_rules = run_mining_experiment()

 
plt.figure(figsize=(10, 6))
sns.barplot(data=benchmark_df, x='Support', y='Time', hue='Dimension')
plt.title('Execution Time: Apriori vs FP-Growth')
plt.yscale('log')
#plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=benchmark_df, x='Level', y='Memory_MB', hue='Algorithm')
plt.title('Memory Usage by Level')
plt.ylabel('Memory (MB)')
#plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=benchmark_df, x='Support', y='Time', hue='Algorithm')
plt.title('Execution Time Comparison: Apriori vs FP-Growth')
plt.xlabel('Minimum Support Level')
plt.ylabel('Execution Time (seconds)')
plt.yscale('log')
#plt.show()



### Redundancy Filtering Across Concept Hierarchies
def filter_hierarchical_redundancy(low_rules, high_rules_set, mapping_dict):
    if low_rules.empty or not high_rules_set: return low_rules
    def check_redundancy(row):
        ant_high = tuple(sorted(set(mapping_dict.get(item, item) for item in row['antecedents'])))
        cons_high = tuple(sorted(set(mapping_dict.get(item, item) for item in row['consequents'])))
        return (ant_high, cons_high) in high_rules_set
    mask = low_rules.apply(check_redundancy, axis=1)
    return low_rules[~mask]

def plot_rules_network(rules_df, title="Association Rules Network", num_rules=30):
    if rules_df.empty:
        print(f"No rules to plot for: {title}")
        return

    G = nx.DiGraph()
    top_rules = rules_df.sort_values('lift', ascending=False).head(num_rules)
    
    for _, row in top_rules.iterrows():
        ant = ' & '.join(list(row['antecedents']))
        cons = ' & '.join(list(row['consequents']))
        G.add_edge(ant, cons, weight=row['lift'], conf=row['confidence'], supp=row['support'])

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.8, seed=42)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] * 0.5 for u, v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', arrowsize=20, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    plt.title(f"{title}\n(Edge thickness = Lift, Nodes = Items)", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    #plt.show()
    




##
def display_robust_rules(rules_df, level_name, top_n=20):
    print(f"\n--- Top {top_n} Robust Rules for {level_name} (Filtered by Kulc & IR) ---")
    temp_df = rules_df.copy()
    temp_df['antecedents'] = temp_df['antecedents'].apply(lambda x: ', '.join(list(x)))
    temp_df['consequents'] = temp_df['consequents'].apply(lambda x: ', '.join(list(x)))
    display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'kulczynski', 'ir']
    top_robust = temp_df.sort_values('kulczynski', ascending=False).head(top_n)
    display(top_robust[display_cols]) 


def get_contextual_rules(rules_df, context_items=['Weekend', 'Hour_Afternoon']):
    filtered = rules_df[rules_df['antecedents'].apply(lambda x: any(item in x for item in context_items))]
    return filtered.sort_values('lift', ascending=False)
afternoon_rules = get_contextual_rules(all_rules[('Weekend', 'product_name', 0.01)], ['Hour_Afternoon'])
display(afternoon_rules.head(10))



## rule network & redundancy filtering
##TODO Lift ve Kulczynski ye gore bir Scatter Plot 
mapping_data = df_merged[['product_name', 'aisle', 'department']].drop_duplicates()
prod_to_aisle_map = mapping_data.set_index('product_name')['aisle'].to_dict()
aisle_to_dept_map = mapping_data[['aisle', 'department']].drop_duplicates().set_index('aisle')['department'].to_dict()
def get_rule_set(rules_df):
    if rules_df.empty: return set()
    return set(
        (tuple(sorted(list(row['antecedents']))), tuple(sorted(list(row['consequents']))))
        for _, row in rules_df.iterrows()
    )

filtered_results = {}
day_types = benchmark_df['Dimension'].unique()
levels = benchmark_df['Level'].unique()
supports = benchmark_df['Support'].unique()

for d_type in day_types:
    for sup in supports:
        rules_dept = all_rules.get((d_type, 'department', sup), pd.DataFrame())
        rules_aisle = all_rules.get((d_type, 'aisle', sup), pd.DataFrame())
        rules_prod = all_rules.get((d_type, 'product_name', sup), pd.DataFrame())
        
        if not rules_aisle.empty and not rules_dept.empty:
            dept_rules_set = get_rule_set(rules_dept)
            interesting_aisle = filter_hierarchical_redundancy(rules_aisle, dept_rules_set, aisle_to_dept_map)
            filtered_results[(d_type, sup, 'aisle_vs_dept')] = interesting_aisle
        else: filtered_results[(d_type, sup, 'aisle_vs_dept')] = rules_aisle

        if not rules_prod.empty and not rules_aisle.empty:
            aisle_rules_set = get_rule_set(rules_aisle)
            interesting_prod = filter_hierarchical_redundancy(rules_prod, aisle_rules_set, prod_to_aisle_map)
            filtered_results[(d_type, sup, 'prod_vs_aisle')] = interesting_prod
        else: filtered_results[(d_type, sup, 'prod_vs_aisle')] = rules_prod
        
        
        
def summarize_context_impact(all_rules_dict):
    summary_data = []
    for key, df in all_rules_dict.items():
        if df.empty: continue
        dim, level, sup = key
        context_tags = ['Weekend', 'Weekday', 'Hour_', 'Basket']
        
        for tag in context_tags:
            mask = df['antecedents'].apply(lambda x: any(tag in str(item) for item in x))
            avg_lift = df[mask]['lift'].mean()
            rule_count = mask.sum()
            
            summary_data.append({
                'Dimension_Group': tag,
                'Level': level,
                'Min_Support': sup,
                'Avg_Lift': avg_lift,
                'Rule_Count': rule_count
            })
            
    return pd.DataFrame(summary_data).dropna()
context_summary = summarize_context_impact(all_rules)
display(context_summary.sort_values('Avg_Lift', ascending=False))


def clean_rules_df(df):
    if df.empty: return df
    df_clean = df.copy()
    df_clean['antecedents'] = df_clean['antecedents'].apply(lambda x: ', '.join(list(x)))
    df_clean['consequents'] = df_clean['consequents'].apply(lambda x: ', '.join(list(x)))
    return df_clean

##TODO Department,Aisle için Kulc > 0.5 IR < 0.5 /// Product için Kulc > 0.3 IR < 0.7 lazım

def plot_robustness_scatter(rules_df, title):
    if rules_df.empty: return
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(rules_df['kulczynski'], rules_df['ir'], 
                         c=rules_df['lift'], cmap='YlOrRd', 
                         s=rules_df['support']*10000, alpha=0.6, edgecolors='w')
    
    plt.colorbar(scatter, label='Lift')
    plt.axvline(0.5, color='green', linestyle='--', alpha=0.4, label='High Certainty')
    plt.axhline(0.5, color='blue', linestyle='--', alpha=0.4, label='Low Imbalance')
    
    plt.xlabel('Kulczynski Measure (Certainty)')
    plt.ylabel('Imbalance Ratio (IR)')
    plt.title(f"Robustness Analysis: {title}")
    plt.legend()
    plt.show()

def run_full_batch_analysis(rules_dict):
    print("\n" + "="*50)
    print("="*50)
    
    for (dim, level, sup), rules in rules_dict.items():
        if rules.empty: continue
        title_str = f"{dim} | {level} | Sup: {sup}"
        plot_robustness_scatter(rules, title_str)
        golden_rules = rules[(rules['kulczynski'] > 0.5) & (rules['ir'] < 0.5)]
        
        print(f"\n>>> kowalski: {title_str}")
        print(f"- total rule count: {len(rules)}")
        print(f"- golden rule count: {len(golden_rules)}")
        
        if not golden_rules.empty:
            print("- hottest rules acc to lift:")
            top_3 = clean_rules_df(golden_rules.sort_values('lift', ascending=False).head(3))
            print(top_3[['antecedents', 'consequents', 'lift', 'kulczynski', 'ir']].to_string(index=False))
        print("-" * 30)
run_full_batch_analysis(all_rules)


scenarios = [
    ('Weekend', 0.01, 'prod_vs_aisle', 'Product Rules (Filtered by Aisle)'),
    ('Weekday', 0.01, 'prod_vs_aisle', 'Product Rules (Filtered by Aisle)'),
    ('Weekend', 0.02, 'aisle_vs_dept', 'Aisle Rules (Filtered by Dept)')
]

for dim, sup, r_type, title_suffix in scenarios:
    key = (dim, sup, r_type)
    if key in filtered_results:
        rules_to_plot = filtered_results[key]
        if not rules_to_plot.empty:
            robust_rules = rules_to_plot.sort_values('kulczynski', ascending=False).head(50)
            full_title = f"{dim} - {title_suffix} (Sup: {sup})"
            plot_rules_network(robust_rules, title=full_title, num_rules=50)