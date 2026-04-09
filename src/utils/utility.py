import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

#-----------------------------------------------------------------------------------------------------------------------#
def extract_conditions_from_tree(tree, feature_names):
    """Extract all unique conditions (feature, threshold) from one decision tree."""
    conditions = set()
    tree_ = tree.tree_

    for i in range(tree_.node_count):
        if tree_.feature[i] != -2:  # not a leaf node
            feature = feature_names[tree_.feature[i]]
            threshold = tree_.threshold[i]
            # Add both <= and > conditions to represent the split
            conditions.add(f"({feature} <= {threshold:.5f})")
            conditions.add(f"({feature} > {threshold:.5f})")
    return conditions
#-----------------------------------------------------------------------------------------------------------------------#
def calculate_U(data_frame, label_series=None, n_trees=100, max_depth=None, random_state=42):
    """
    Calculates the 'Universe' (U) of all possible conditions by training a
    RandomForestClassifier and extracting all conditions from its decision trees.
    """
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=random_state)
    rf.fit(data_frame, label_series)

    all_conditions = set()  # Use a set to prevent duplicate conditions
    for tree in rf.estimators_:
        conditions = extract_conditions_from_tree(tree, feature_names=data_frame.columns)
        all_conditions.update(conditions)

    return all_conditions

#-----------------------------------------------------------------------------------------------------------------------#
def parse_condition(cond_str):
    """
    Converts a condition string (e.g., '(Feature <= 0.5)') into a structured tuple
    (feature_name, operator, value).
    """
    cond_str = cond_str.strip("() ").strip()
    match = re.match(r"(.+?)\s*(<=|>=|==|!=|<|>)\s*(\S+)", cond_str)
    if match:
        feature, operator, value = match.groups()
        try:
            value = float(value) # Attempt to convert value to float
        except ValueError:
            pass # Keep as string if conversion fails (e.g., for categorical features)
        return feature.strip(), operator, value
    else:
        raise ValueError(f"Condition '{cond_str}' not recognized.")
#-----------------------------------------------------------------------------------------------------------------------#
def parse_conditions_to_structured(cond_strings):
    structured_conditions = []
    pattern = re.compile(r"(.+?)\s*(<=|>=|<|>)\s*(.+)")  # feature, operator, value
    for cond_string in cond_strings:
        cond_string = cond_string.strip("()")  # remove parentheses if present
        match = pattern.match(cond_string)
        if not match:
            raise ValueError(f"Invalid condition format: {cond_string}")
        
        feat, op, val_str = match.groups()
        try:
            val = float(val_str)
        except ValueError:
            raise ValueError(f"Invalid numeric value in condition: {cond_string}")
        
        structured_conditions.append((feat.strip(), op, val))
    return structured_conditions
#-----------------------------------------------------------------------------------------------------------------------#
def get_rule_coverage_mask(structured_conditions, X):
    """
    Vectorized function to get a boolean mask of samples covered by a rule.
    X is a pandas DataFrame.
    structured_conditions: list of tuples (feature, operator, value)
    """
    # Start with a mask where all instances are True
    combined_mask = np.ones(len(X), dtype=bool)
    
    # Apply each condition
    for feat, op, val in structured_conditions:
        feature_data = X[feat].values
        
        if op == '<=':
            new_mask = feature_data <= val
        elif op == '>=':
            new_mask = feature_data >= val
        elif op == '<':
            new_mask = feature_data < val
        elif op == '>':
            new_mask = feature_data > val
        elif op == '==':
            new_mask = feature_data == val
        elif op == '!=':
            new_mask = feature_data != val
        else:
            raise ValueError(f"Unknown operator '{op}' in rule condition for feature '{feat}'")
        
        # Combine with the existing mask
        combined_mask = np.logical_and(combined_mask, new_mask)
    
    return combined_mask
#-----------------------------------------------------------------------------------------------------------------------#
def apply_ruleset_v2(ruleset, X, y, default_class = None):
    """
    Apply a set of rules to a dataset.
    ruleset: list of tuples (conditions_set, predicted_class, ...)
    X: DataFrame of features
    y: True labels (Series or array) — needed for per-rule accuracy
    default_class: class to predict when no rule matches
    """
    y_pred = []
    rule_stats = []  # Store per-rule coverage & accuracy
    rule_matches = []  # Store matched indices for overlap

    # --- Per-rule stats & overlaps ---
    for rule_idx, rule in enumerate(ruleset, start=1):
        conditions, pred_class = rule[:2]  # only unpack first two elements

        # Find samples where this rule applies
        mask = X.apply(
            lambda row: all(
                eval(single_cond.replace("AND", "and").replace("OR", "or"),
                     {}, {col: row[col] for col in X.columns})
                for single_cond in conditions
            ),
            axis=1
        )

        matched_indices = np.where(mask)[0]
        rule_matches.append(set(matched_indices))

        if len(matched_indices) > 0:
            correct_matches = np.sum(y.iloc[matched_indices] == pred_class)
            accuracy = correct_matches / len(matched_indices)
        else:
            accuracy = np.nan  # No coverage means accuracy is undefined

        coverage = len(matched_indices) / len(X)
        rule_stats.append({
            "Rule #": rule_idx,
            "Predicted Class": pred_class,
            "Coverage (%)": coverage * 100,
            "Accuracy (%)": accuracy * 100 if not np.isnan(accuracy) else None
        })

    # Final predictions 
    for _, row in X.iterrows():
        matched_class = None
        for rule in ruleset:
            conditions, pred_class = rule[:2]  # only unpack first two elements
            if all(
                eval(single_cond.replace("AND", "and").replace("OR", "or"),
                     {}, {col: row[col] for col in X.columns})
                for single_cond in conditions
            ):
                matched_class = pred_class
                break
        y_pred.append(matched_class if matched_class is not None else default_class)

    overlaps = {}
    for i in range(len(rule_matches)):
        for j in range(i + 1, len(rule_matches)):
            overlap_count = len(rule_matches[i].intersection(rule_matches[j]))
            if overlap_count > 0:
                overlaps[(i+1, j+1)] = overlap_count

    return np.array(y_pred), pd.DataFrame(rule_stats), overlaps
#-----------------------------------------------------------------------------------------------------------------------#
def prune_rules_verbose(rules, X, y, default_class):
    """
    Iteratively prune rules that do not improve performance.
    Prints reasoning for each pruning step.
    """
    best_rules = rules.copy()
    best_pred, _, _ = apply_ruleset_v2(best_rules, X, y, default_class)
    best_score = f1_score(y, best_pred, average="macro")
    print(f"Initial Macro F1: {best_score:.4f}\n")

    pruned = True
    while pruned:
        pruned = False
        for i, (conds, pred_class, fitness, coverage, n_removed) in enumerate(best_rules):
            candidate_rules = best_rules[:i] + best_rules[i+1:]
            y_pred, _, _ = apply_ruleset_v2(candidate_rules, X, y, default_class)
            score = f1_score(y, y_pred, average="macro")

            # Heuristics for reasoning
            accuracy_est = fitness  # Assuming fitness ~ accuracy
            reason = None
            if accuracy_est < 0.5:
                reason = f"low accuracy ({accuracy_est:.2f})"
            elif score >= best_score:
                reason = "redundant (later rules cover instances equally well or better)"
            else:
                continue  # skip pruning this one

            # If pruning improves or maintains performance, prune
            if score >= best_score or accuracy_est < 0.5:
                print(f"Pruning Rule #{i+1} → Reason: {reason}")
                print(f" New Macro F1: {score:.4f} (previous {best_score:.4f})\n")
                best_rules = candidate_rules
                best_score = score
                pruned = True
                break  # restart loop after pruning

    return best_rules, best_score

#-----------------------------------------------------------------------------------------------------------------------#