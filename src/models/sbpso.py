import numpy as np
import pandas as pd
from random import sample, uniform
from copy import deepcopy
import re
from sklearn.tree import _tree
from math import floor

from utility import (
    parse_condition,
    parse_conditions_to_structured,
    get_rule_coverage_mask
)

#-----------------------------------------------------------------------------------------------------------------------#
def single_rule_fitness(
    conditions, predicted_class, X, y,
    alpha=0.7,
    delta=0.2,
    redundancy_overlap_threshold=0.46,
    beta=0.2,
    gamma=0.5,              
    previous_rules=None,
    max_possible_conditions=None
):

    """
    Evaluate the quality of a single candidate rule.

    This fitness function is used by SBPSO to score candidate IF-THEN rules.
    A good rule should:
    1. classify the samples it covers correctly,
    2. cover a meaningful portion of the data,
    3. remain relatively short and interpretable,
    4. avoid being overly general,
    5. avoid excessive overlap with previously extracted rules.

    Parameters
    ----------
    conditions : set or list
        The antecedent of the rule, represented as a set/list of condition strings,

    predicted_class : int or str
        The class label predicted by the rule.

    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        True class labels corresponding to X.

    alpha : float, default=0.7
        Controls the trade-off between rule accuracy and coverage in the base score.
        Higher alpha places more emphasis on accuracy.

    delta : float, default=0.2
        Gives additional weight to precision-like correctness in the final base score.

    redundancy_overlap_threshold : float, default=0.46
        Threshold above which overlap with previous rules is considered excessive.

    beta : float, default=0.2
        Strength of the rule-length penalty. Larger beta penalizes longer rules more.

    gamma : float, default=0.5
        Strength of the penalty for overly general rules with very high coverage.

    previous_rules : list, default=None
        Previously extracted rules. Used to discourage redundant or conflicting rules.

    max_possible_conditions : int, default=None
        Normalization factor for the rule-length penalty. If None, defaults to the
        number of features in X.

    Returns
    -------
    float
        The final fitness score for the candidate rule. Higher is better.
    """

    # If the rule has no conditions, it is meaningless and returns 0
    if not conditions:
        return 0.0

    structured_conds = parse_conditions_to_structured(list(conditions))
    mask = get_rule_coverage_mask(structured_conds, X)

    if not np.any(mask):
        return 0.0

    #If the rule covers no samples, it is meaningless
    covered_y = y.loc[mask]
    total_covered = len(covered_y)
    total_samples = len(y)

    correct_classifications = np.sum(covered_y == predicted_class)
    coverage = total_covered / total_samples
    accuracy = correct_classifications / total_covered if total_covered > 0 else 0.0
    precision = accuracy

    # discourage overly general rules
    coverage_penalty = gamma * max(0, coverage - 0.4)  

    #Rule length penalty 
    if max_possible_conditions is None:
        max_possible_conditions = X.shape[1]
    length_penalty = 1 - (len(conditions) / max_possible_conditions) * beta
    length_penalty = max(length_penalty, 0)  

    # Fitness calculation
    base_fitness = (alpha * accuracy + (1 - alpha) * coverage)
    base_fitness = (1 - delta) * base_fitness + delta * precision

    # Apply penalties
    fitness = base_fitness * length_penalty - coverage_penalty

    # Overlap/conflict penalty 
    if previous_rules:
        for prev_conds, prev_class in previous_rules:
            prev_structured_conds = parse_conditions_to_structured(list(prev_conds))
            prev_mask = get_rule_coverage_mask(prev_structured_conds, X)
            
            overlap_ratio = np.sum(mask & prev_mask) / total_covered if total_covered > 0 else 0
            
            if overlap_ratio > redundancy_overlap_threshold:
                if prev_class == predicted_class:
                    fitness *= 0.9
                else:
                    fitness *= 0.7

    return fitness

#-----------------------------------------------------------------------------------------------------------------------#

class DesirabilityMetrics:
     """
    Tracks how beneficial it is to ADD or REMOVE each condition during SBPSO.

    The idea is to learn from previous particle moves:
    - If adding a condition improves fitness → that condition becomes "desirable"
    - If removing a condition improves fitness → that condition becomes "undesirable"

    This introduces a form of adaptive learning into the SBPSO search process.
    """  
    def __init__(self, U, desirability_threshold=0.65):
        """
        Initializes desirability scores for all conditions in the universal set U.
        U: A set of all possible conditions (strings).
        """
        self.U = sorted(list(U))
        self.scores = {cond: {'add': 0.0, 'remove': 0.0, 'total_actions': 0} for cond in self.U}
        self.U_map = {cond: i for i, cond in enumerate(self.U)}
        self.desirability_threshold = desirability_threshold
        
        self.desirable = set()
        self.undesirable = set()
    
    def update_scores(self, conditions_added, conditions_removed, score_improvement):
        """
        Updates the desirability scores based on the outcome of a particle's move.
        - conditions_added: a list of conditions that were added.
        - conditions_removed: a list of conditions that were removed.
        - score_improvement: the improvement in fitness score from the particle's move.
        """
        for cond in conditions_added:
            if cond in self.scores:
                self.scores[cond]['add'] += score_improvement
                self.scores[cond]['total_actions'] += 1
                
        for cond in conditions_removed:
            if cond in self.scores:
                self.scores[cond]['remove'] += score_improvement
                self.scores[cond]['total_actions'] += 1

    def get_add_desirability(self, cond):
        """Returns the desirability score for adding a condition."""
        if self.scores[cond]['total_actions'] > 0:
            return self.scores[cond]['add'] / self.scores[cond]['total_actions']
        return 0.0

    def get_remove_desirability(self, cond):
        """Returns the desirability score for removing a condition."""
        if self.scores[cond]['total_actions'] > 0:
            return self.scores[cond]['remove'] / self.scores[cond]['total_actions']
        return 0.0

    def update_desirable_sets(self):
        """
        Updates the desirable and undesirable sets based on current desirability scores.
        This method should be called periodically (e.g., at the end of each SBPSO iteration).
        """
        self.desirable = set()
        self.undesirable = set()
        for cond, scores in self.scores.items():
            add_score = self.get_add_desirability(cond)
            remove_score = self.get_remove_desirability(cond)
            
            if add_score > self.desirability_threshold and remove_score <= 0.2:
                self.desirable.add(cond)
            
            if remove_score > self.desirability_threshold and add_score <= 0.2:
                self.undesirable.add(cond)

#-----------------------------------------------------------------------------------------------------------------------#


class SBParticle:
    """
    Langeveld-style Set-Based Particle for PSO rule extraction
    Represents one particle in the Set-Based PSO search space.

    In this implementation, a particle corresponds to one candidate rule.
    Its position is a set of rule conditions, and its goal is to search for
    a high-quality, interpretable rule that can be used in the final ruleset.

    Each particle maintains:
    - a current rule (its current position),
    - a personal best rule found so far,
    - access to the global best rule found by the swarm,
    - fitness values used to compare candidate rules.
    """
    
    def __init__(self, c1, c2, c3, c4, k, U, data_frame, label_frame, f,
                 class_labels, desirability_obj=None, previous_rules=None, **kwargs):
        """
        Initialize a particle.

        Parameters
        ----------
        c1, c2, c3, c4 : float
            SBPSO coefficients controlling the search behaviour.

        k : int
            Tournament size / search control parameter used in set-based operators.

        U : set
            Universal set of all possible candidate conditions.

        data_frame : pandas.DataFrame
            Feature matrix used to evaluate candidate rules.

        label_frame : pandas.Series
            Class labels corresponding to the feature matrix.

        f : function
            Fitness function used to score a candidate rule.

        class_labels : array-like
            Unique class labels present in the current training subset.

        desirability_obj : DesirabilityMetrics, optional
            Tracks how useful it is to add or remove specific conditions.

        previous_rules : list, optional
            Previously extracted rules. Used to discourage redundant or conflicting rules.

        kwargs : dict
            Additional parameters passed to fitness evaluation and greedy optimization.
        """
        self.c1, self.c2, self.c3, self.c4, self.k = c1, c2, c3, c4, k
        self.U = U
        self.data_frame = data_frame
        self.label_frame = label_frame
        self.f = f
        self.previous_rules = previous_rules
        self.kwargs = kwargs

        self.class_labels = class_labels
        self.desirability_obj = desirability_obj

        initial_conditions = set(sample(list(self.U), k=np.random.randint(1, 5)))

        self.predicted_class = self.get_best_class_for_position(initial_conditions)
        self.position = [(initial_conditions, self.predicted_class)]

        self.v_add, self.v_sub = set(), set()

        self.f_val = self.f(
            self.position[0][0],
            self.predicted_class,
            self.data_frame,
            self.label_frame,
            previous_rules=self.previous_rules
        )

        self.personal_best = deepcopy(self.position)
        self.f_personal_best = self.f_val
        self.global_best = None
        self.f_global_best = -1

    def get_best_class_for_position(self, conditions):
        """
        Assign the most appropriate predicted class to the current rule.

        The rule may cover samples from multiple classes. This method chooses
        the majority class among the covered samples and uses it as the
        consequent of the rule.

        Returns
        -------
        class label or None
            Majority class among covered samples, or None if the rule covers no samples.
        """
        if not conditions:
            return None

        structured_conds = parse_conditions_to_structured(list(conditions))
        mask = get_rule_coverage_mask(structured_conds, self.data_frame)

        if not np.any(mask):
            return None

        covered_labels = self.label_frame.loc[mask]
        class_counts = covered_labels.value_counts()

        return class_counts.idxmax()

    def is_redundant(self, candidate, existing_conditions):
        """
        Check whether a candidate condition is already implied by the current rule.

        Example:
        - If the rule already contains "feature > 5", then adding "feature > 4"
          is redundant because it does not make the rule stricter.
        - If the rule already contains "feature <= 3", then adding "feature <= 4"
          is also redundant.

        Parameters
        ----------
        candidate : str
            Candidate condition to test.

        existing_conditions : set
            Current rule conditions.

        Returns
        -------
        bool
            True if the candidate condition is redundant, False otherwise.
        """

        feat, op, thresh = parse_condition(candidate)  
    
        for cond in existing_conditions:
            f, existing_op, existing_thresh = parse_condition(cond)
            if f != feat:
                continue
            # Only compare same operators
            if op == ">" and existing_op == ">":
                if thresh <= existing_thresh:  # Candidate is looser → redundant
                    return True
            elif op == "<=" and existing_op == "<=":
                if thresh >= existing_thresh:  # Candidate is looser → redundant
                    return True
        return False

    def remove_elements(self, beta):
        """
        Select conditions to remove from the current rule.

        This operator focuses on conditions shared by:
        - the current position,
        - the personal best,
        - and optionally the global best.

        A tournament-selection strategy is used to choose which conditions
        are most suitable for removal.

        Parameters
        ----------
        beta : float
            Controls how many conditions are removed.

        Returns
        -------
        set
            A set of tagged removal operations: {("-", condition), ...}
        """

        S = self.position[0][0].intersection(self.personal_best[0][0])
        if self.global_best:
            S = S.intersection(self.global_best[0][0])
        if len(S) == 0: return set()
        
        to_subtract = set()
        onebool = 1 if uniform(0,1) < beta - floor(beta) else 0
        nbs = min(len(S), floor(beta) + onebool)

        # Tournament selection
        for i in range(nbs):
            k = min(len(S), self.k)
            ejs = sample(list(S), k=k)
            scores = [self.f(self.position[0][0].difference({e}),
                             self.predicted_class,
                             self.data_frame,
                             self.label_frame,
                             previous_rules=self.previous_rules)
                      for e in ejs]
            tw = ejs[np.argmax(scores)]
            to_subtract.add(tw)
            S = S.difference({tw})
        return set([("-", s) for s in to_subtract])

    def add_elements(self, beta):
        """
        Select conditions to add to the current rule.

        Candidate additions come from the universal set U, excluding conditions
        already present in the current rule, personal best, or global best.
        Redundant candidates are discarded.

        Parameters
        ----------
        beta : float
            Controls how many conditions are added.

        Returns
        -------
        set: A set of tagged addition operations: {("+", condition), ...}
        """
        A = self.U.difference(self.position[0][0].union(self.personal_best[0][0]))
        if self.global_best:
            A = A.difference(self.global_best[0][0])
        if len(A) == 0: 
            return set()
    
        to_add = set()
        onebool = 1 if uniform(0,1) < beta - floor(beta) else 0
        nba = min(len(A), floor(beta) + onebool)
    
        for i in range(nba):
            k = min(len(A), self.k)
            ejs = sample(list(A), k=k)
            # Filter out redundant candidates
            ejs = [e for e in ejs if not self.is_redundant(e, self.position[0][0])]
            if not ejs:
                continue
    
            scores = [self.f(self.position[0][0].union({e}),
                             self.predicted_class,
                             self.data_frame,
                             self.label_frame,
                             previous_rules=self.previous_rules)
                      for e in ejs]
            tw = ejs[np.argmax(scores)]
            to_add.add(tw)
            A = A.difference({tw})
        return set([("+", a) for a in to_add])

    def update_particle(self, t):
        """
        Perform one particle update step.

        This method:
        1. samples random numbers used by SBPSO,
        2. updates velocity,
        3. updates the particle position,
        4. reassigns the best class label,
        5. greedily simplifies the rule,
        6. evaluates the new fitness,
        7. updates the particle's personal best if improved.
        """

        self.t = t
        self.r1 = uniform(0,1)
        self.r2 = uniform(0,1)
        self.r3 = uniform(0,1)
        self.r4 = uniform(0,1)

        current_conditions = self.position[0][0]

        # Compute velocities
        self.calculate_velocity(current_conditions)
        self.calculate_position()

        # Update predicted class
        new_conditions = self.position[0][0]
        self.predicted_class = self.get_best_class_for_position(new_conditions)

        # Greedy optimization
        optimized_conditions, _ = self.greedy_rule_optimization(new_conditions, self.predicted_class)
        self.position = [(optimized_conditions, self.get_best_class_for_position(optimized_conditions))]

        # Evaluate fitness safely
        self.f_val = self.f(
            self.position[0][0],
            self.position[0][1],
            self.data_frame,
            self.label_frame,
            previous_rules=self.previous_rules
        )

        if self.f_val > self.f_personal_best:
            self.personal_best = deepcopy(self.position)
            self.f_personal_best = self.f_val

    def calculate_velocity(self, current_conditions):
        """
        Compute the velocity update for a particle in Set-Based PSO.
    
        In this formulation:
        - A particle represents a rule (set of conditions)
        - Velocity is represented as:
            v_add → conditions to add
            v_sub → conditions to remove
    
        The update is driven by two components:
        1. Cognitive component → move toward personal best rule
        2. Social component → move toward global best rule
        """
        # Cognitive component: pull toward personal best
        pb_conds = self.personal_best[0][0]
        cog_add = pb_conds.difference(current_conditions)
        cog_sub = current_conditions.difference(pb_conds)
    
        # Social component: pull toward global best
        gb_conds = self.global_best[0][0] if self.global_best else set()
        soc_add = gb_conds.difference(current_conditions)
        soc_sub = current_conditions.difference(gb_conds)
        
        # v_add: conditions to add
        # v_sub: conditions to remove
        self.v_add = set()
        self.v_sub = set()
    
        # Add missing personal-best conditions with probability c1 * r1
        for cond in cog_add:
            if uniform(0, 1) < self.c1 * self.r1:
                self.v_add.add(cond)
    
        # Remove extra conditions not in personal best
        for cond in cog_sub:
            if uniform(0, 1) < self.c2 * self.r2:
                self.v_sub.add(cond)

        # Add missing global-best conditions
        for cond in soc_add:
            if uniform(0, 1) < self.c3 * self.r3:
                self.v_add.add(cond)
    
        # Remove conditions not in global best
        for cond in soc_sub:
            if uniform(0, 1) < self.c4 * self.r4:
                self.v_sub.add(cond)

    def calculate_position(self):
        # Apply additions/removals
        current_conditions = self.position[0][0].union(self.v_add).difference(self.v_sub)
        self.position = [(current_conditions, None)]
    
    def greedy_rule_optimization(self, conditions, pred_class):
        """
        The goal is to remove unnecessary conditions while maintaining or improving
        the rule's quality. This improves interpretability and avoids overfitting.
    
        The method works iteratively:
        - At each step, try removing each condition one at a time
        - Keep the removal that improves the rule the most
        - Repeat until no further improvement is possible

        """
        # Minimum improvement required to accept a change
        improvement_threshold = self.kwargs.get("improvement_threshold_greedy", 0.01)
        current_conditions = deepcopy(conditions)
        
        # If rule has fewer than 2 conditions, it cannot be simplified further. This is to prevent 0 conditions from happening.
        if len(current_conditions) < 2:
            return current_conditions, conditions
    
        improved = True
        current_score = self.evaluate_single_rule(current_conditions, pred_class)
    
        # Continue pruning while improvements are found
        while improved and len(current_conditions) > 1:
            improved = False
            best_pruned_conds = None
            best_pruned_score = current_score
    
            for cond in list(current_conditions):
                test_conditions = current_conditions - {cond}
    
                test_score = self.evaluate_single_rule(test_conditions, pred_class)
                
                if test_score >= best_pruned_score + improvement_threshold:
                    best_pruned_score = test_score
                    best_pruned_conds = test_conditions
    
            # If a better rule was found, update the current rule
            if best_pruned_conds is not None:
                current_conditions = best_pruned_conds
                current_score = best_pruned_score
                improved = True

        return current_conditions, conditions

    def evaluate_single_rule(self, conditions, pred_class):
        """
        Compute a lightweight score for a rule during greedy pruning.
    
        This is NOT the full SBPSO fitness function.
        It is a simplified, faster scoring function used only during
        local optimization to decide whether removing a condition is beneficial.
    
        The score rewards:
        - high accuracy on covered samples
        - reasonable coverage
    
        The score penalizes:
        - overly complex (long) rules
        """

        if not conditions:
            return -1
        structured_conds = [parse_condition(c) for c in list(conditions)]
        mask = get_rule_coverage_mask(structured_conds, self.data_frame)
        if not np.any(mask):
            return 0.0

        correct_matches = np.sum(self.label_frame.loc[mask] == pred_class)
        coverage = len(np.where(mask)[0]) / len(self.data_frame)
        accuracy = correct_matches / len(np.where(mask)[0])

        alpha = self.kwargs.get('alpha', 0.7)
        base_score = alpha * accuracy + (1-alpha) * coverage
        penalty_conditions = max(0, len(conditions)-6) * 0.01

        return base_score - penalty_conditions

#-----------------------------------------------------------------------------------------------------------------------#

class SBPSO:
    """
    Set-Based Particle Swarm Optimization (SBPSO) for rule extraction.

    This class manages:
    - the swarm of particles (candidate rules),
    - the global best solution,
    - the iterative optimization process,
    - performance tracking.

    Each particle searches for a high-quality rule, and the swarm
    collectively converges toward the best rule.
    """
    def __init__(self, c1, c2, c3, c4, k, swarm_size, max_iterations,
                 data_frame, label_frame, U, f, previous_rules=None,
                 run_simulation_on_init=False, verbosity=1, **kwargs):
        """
        Initialize the SBPSO optimizer.

        Parameters
        ----------
        c1, c2, c3, c4 : float
            PSO coefficients controlling cognitive and social updates.

        k : int
            Tournament parameter used in set-based operators.

        swarm_size : int
            Number of particles (rules) in the swarm.

        max_iterations : int
            Number of optimization iterations.

        data_frame : DataFrame
            Feature matrix.

        label_frame : Series
            Target labels.

        U : set
            Universal set of candidate rule conditions.

        f : function
            Fitness function used to evaluate rules.

        previous_rules : list, optional
            Previously extracted rules (used for redundancy control).

        run_simulation_on_init : bool
            Whether to immediately run optimization.

        verbosity : int
            Controls printing level.

        kwargs : dict
            Additional parameters passed to particles.
        """

        self._run_sim_on_init = run_simulation_on_init

        self.VERBOSITY = verbosity

        self.c1, self.c2, self.c3, self.c4 = c1, c2, c3, c4
        self.k = k
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.data_frame = data_frame
        self.label_frame = label_frame
        self.U = U
        self.f = f
        self.previous_rules = previous_rules
        self.kwargs = kwargs

        # Adaptive penalty parameters
        self.gamma_min = 0.1
        self.gamma_max = 1.0
        self.beta_min = 0.05
        self.beta_max = 0.3

        # Desirability object
        self.desirability = DesirabilityMetrics(U)

        # Initialize performance metrics
        self.metrics = SBPSOmetrics(self.swarm_size, self.max_iterations)

        # Initialize swarm
        self.particles = [None] * self.swarm_size
        for i in range(self.swarm_size):
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop("class_labels", None)
            kwargs_copy.pop("desirability_obj", None)

            self.particles[i] = SBParticle(
                c1=self.c1,
                c2=self.c2,
                c3=self.c3,
                c4=self.c4,
                k=self.k,
                U=self.U,
                max_iterations=self.max_iterations,
                data_frame=self.data_frame,
                label_frame=self.label_frame,
                f=self.f,
                class_labels=np.unique(self.label_frame),
                desirability_obj=self.desirability,
                previous_rules=self.previous_rules,
                **kwargs_copy
            )

        # Initialize global best
        self.global_best = self.particles[0].personal_best
        self.f_global_best = self.particles[0].f_personal_best
        self.global_best_class = self.particles[0].personal_best[0][1]

        for i in range(self.swarm_size):
            if self.particles[i].f_personal_best > self.f_global_best:
                self.f_global_best = self.particles[i].f_personal_best
                self.global_best = self.particles[i].personal_best
                self.global_best_class = self.particles[i].personal_best[0][1]

        for i in range(self.swarm_size):
            self.particles[i].f_global_best = self.f_global_best
            self.particles[i].global_best = self.global_best

        # Run simulation if requested
        if self._run_sim_on_init:
            self.run_simulation()

    def run_simulation(self):
        for i in range(self.max_iterations):
            self.t = i

            # Adaptive penalty schedule 
            gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * (i / self.max_iterations)
            beta = self.beta_min + (self.beta_max - self.beta_min) * (i / self.max_iterations)

            # Evaluate each particle using adaptive gamma and beta
            for particle in self.particles:
                particle.f_val = single_rule_fitness(
                    conditions=particle.position[0][0],
                    predicted_class=particle.position[0][1],
                    X=self.data_frame,
                    y=self.label_frame,
                    alpha=self.kwargs.get("alpha", 0.7),
                    delta=self.kwargs.get("delta", 0.2),
                    gamma=gamma,
                    beta=beta,
                    previous_rules=self.previous_rules
                )

            self.next_iteration()
            self.update_performance_metrics()

            if self.VERBOSITY == 2:
                print(f"\tt({i:4d}): {self.f_global_best}, "
                      f"{self.metrics.ave_personal_best_ff_vals[self.t]}, "
                      f"{self.metrics.ave_ff_vals[self.t]}, "
                      f"{self.metrics.global_best_size[self.t]}, "
                      f"{self.metrics.ave_position_size[self.t]:.4f}")

        if self.VERBOSITY >= 1:
            print("\nOptimization completed.")
            print("Global best (conditions):", self.global_best[0][0])
            print("Global best class:", self.global_best_class)
            print("Global best fitness:", self.f_global_best)

        return deepcopy(self.global_best[0][0]), deepcopy(self.global_best_class), self.f_global_best

    def initialise_performance_metrics(self):
        self.metrics = SBPSOmetrics(self.swarm_size, self.max_iterations)

    def update_performance_metrics(self):
        self.metrics.update_global_best_ff_vals(self.t, self.f_global_best)
        
        self.metrics.update_global_best_pos(self.t, self.global_best[0][0])
        self.metrics.update_global_best_size(self.t, len(self.global_best[0][0]))

        self.metrics.update_average_personal_best_ff_vals(
            self.t, self.calculate_average_personal_best_ff_vals()
        )
        self.metrics.update_average_personal_best_size(
            self.t, self.calculate_average_personal_best_size()
        )
        self.metrics.update_average_personal_best_jaccard(
            self.t, self.calculate_average_personal_best_jaccard()
        )

        self.metrics.update_average_ff_vals(self.t, self.calculate_average_ff_vals())
        self.metrics.update_average_position_size(
            self.t, self.calculate_average_pos_size()
        )
        self.metrics.update_average_jaccard_index(
            self.t, self.calculate_average_jaccard()
        )

        self.metrics.update_average_jaccard_to_global_best(self.t, self.calculate_average_jaccard_to_global_best())
        self.metrics.update_average_jaccard_to_personal_best(self.t, self.calculate_average_jaccard_to_personal_best())

        for j in range(self.swarm_size):
            self.metrics.particle_positions[j][self.t] = self.particles[j].position[0][0]
            self.metrics.particle_velocities_add[j][self.t] = self.particles[j].v_add
            self.metrics.particle_velocities_sub[j][self.t] = self.particles[j].v_sub

    def next_iteration(self):
        for i in range(self.swarm_size):
            if self.particles[i].f_personal_best > self.f_global_best:
                self.f_global_best = self.particles[i].f_personal_best
                self.global_best = self.particles[i].personal_best
                self.global_best_class = self.particles[i].personal_best[0][1]

        for i in range(self.swarm_size):
            self.particles[i].global_best = self.global_best
            self.particles[i].f_global_best = self.f_global_best

        self.update_particles()

    def update_particles(self):
        for i in range(self.swarm_size):
            self.particles[i].update_particle(self.t)

    # --- metric helpers (UNCHANGED) ---
    def jaccard(self, A, B):
        conditions_A = A[0][0]
        conditions_B = B[0][0]
        if len(conditions_A) == 0 and len(conditions_B) == 0:
            return 1
        return len(conditions_A.intersection(conditions_B)) / len(conditions_A.union(conditions_B))

    def calculate_average_personal_best_ff_vals(self):
        return np.average([self.particles[i].f_personal_best for i in range(self.swarm_size)])

    def calculate_average_personal_best_jaccard(self):
        sum_jaccard = 0
        for i in range(self.swarm_size - 1):
            particle_jaccard = 0
            for j in range(i + 1, self.swarm_size):
                particle_jaccard += self.jaccard(
                    self.particles[i].personal_best, self.particles[j].personal_best
                )
            sum_jaccard += particle_jaccard / self.swarm_size
        return sum_jaccard / self.swarm_size

    def calculate_average_personal_best_size(self):
        return np.average([len(self.particles[i].personal_best[0][0]) for i in range(self.swarm_size)])

    def calculate_average_ff_vals(self):
        return np.average([self.particles[i].f_val for i in range(self.swarm_size)])

    def calculate_average_jaccard(self):
        sum_jaccard = 0
        count = 0
        for i in range(self.swarm_size):
            for j in range(self.swarm_size):
                if i != j:
                    sum_jaccard += self.jaccard(self.particles[i].position, self.particles[j].position)
                    count += 1
        return sum_jaccard / count if count > 0 else 0

    def calculate_average_jaccard_to_global_best(self):
        return np.average([self.jaccard(self.global_best, self.particles[i].position) for i in range(self.swarm_size)])

    def calculate_average_jaccard_to_personal_best(self):
        return np.average([self.jaccard(self.particles[i].personal_best, self.particles[i].position) for i in range(self.swarm_size)])

    def calculate_average_pos_size(self):
        return np.average([len(self.particles[i].position[0][0]) for i in range(self.swarm_size)])


#-----------------------------------------------------------------------------------------------------------------------#

def separate_and_conquer(c1, c2, c3, c4, k, swarm_size, max_iterations,
                         data_frame, label_frame, U, f, **kwargs):
    """
    Implements the separate-and-conquer strategy to discover a set of classification rules.
    This function iteratively calls the SBPSO algorithm to find the best rule on a
    shrinking dataset.

    Parameters:
        c1, c2, c3, c4, k: SBPSO-specific parameters.
        swarm_size, max_iterations: SBPSO swarm parameters.
        data_frame (pd.DataFrame): The feature data.
        label_frame (pd.Series): The target labels.
        U (set): The universal set of all possible conditions.
        f (function): The fitness function for evaluating rules.
        kwargs: Additional keyword arguments for S&C and SBPSO.

    Returns:
        list: A list of discovered rules, each a tuple of (conditions, class, fitness, coverage, n_removed).
    """
    remaining_X = data_frame.copy()
    remaining_y = label_frame.copy()
    discovered = []
    rule_count = 0

    previous_rules_list = []
    
    # Extract S&C-specific kwargs so they are not passed to SBPSO
    max_rules = kwargs.pop('max_rules', 50)
    min_coverage = kwargs.pop('min_coverage', 0.01)
    remove_only_correct = kwargs.pop('remove_only_correct', True)
    stop_on_no_removal = kwargs.pop('stop_on_no_removal', True)
    verbosity = kwargs.pop('verbosity', 1)

    while (len(remaining_X) > 0) and (rule_count < max_rules):
        rule_count += 1
        if verbosity >= 1:
            print(f"\n=== Extracting rule #{rule_count} from {len(remaining_X)} remaining instances ===")

        # Call SBPSO for the current subset
        sbpso_run = SBPSO(
            c1, c2, c3, c4, k,
            swarm_size, max_iterations,
            remaining_X, remaining_y,
            U,
            f,
            previous_rules=previous_rules_list,
            run_simulation_on_init=True,
            **kwargs
        )
        
        best_conditions = sbpso_run.global_best[0][0]
        best_class = sbpso_run.global_best_class
        best_f = sbpso_run.f_global_best

        if verbosity >= 1:
            print("--- Final Rule Found ---")
            print(f"Rule (Fitness {best_f:.4f}): {list(best_conditions)}")
        
        if not best_conditions:
            coverage = 0.0
            n_removed = 0
        else:
            structured = parse_conditions_to_structured(list(best_conditions))
            mask_covered = get_rule_coverage_mask(structured, remaining_X)  # instances satisfying rule
            coverage = mask_covered.mean()

            # Determine which instances to remove
            if remove_only_correct:
                mask_correct = mask_covered & (remaining_y.values == best_class)
                remove_mask = mask_correct
            else:
                remove_mask = mask_covered

            n_removed = int(remove_mask.sum())

        if verbosity >= 1:
            print(f"Candidate rule covers {coverage*100:.2f}% of remaining data (fitness={best_f:.5f})")

        if coverage < min_coverage or n_removed == 0:
            if verbosity >= 1:
                print("Stopping: coverage below min_coverage or no instances were correctly removed.")
            if stop_on_no_removal:
                break

        # Remove selected instances
        remaining_X = remaining_X.iloc[~remove_mask]
        remaining_y = remaining_y.iloc[~remove_mask]

        previous_rules_list.append((best_conditions, best_class))
        discovered.append((deepcopy(best_conditions), deepcopy(best_class), best_f, coverage, n_removed))

        if verbosity >= 1:
            print(f"Accepted rule #{rule_count}: removed {n_removed} instances. {len(remaining_X)} remain.")

    return discovered


#-----------------------------------------------------------------------------------------------------------------------#


class SBPSOmetrics:
    def __init__(self, swarm_size, max_iterations):
        """Initializes data structures to store performance metrics over iterations."""
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        
        # Global best metrics
        self.global_best_pos = [None] * self.max_iterations
        self.global_best_ff_vals = np.zeros(max_iterations)
        self.global_best_size = np.zeros(max_iterations)
        self.global_test_accuracy = [None] * self.max_iterations
        self.global_test_coverage = [None] * self.max_iterations

        # Average personal best metrics
        self.ave_personal_best_ff_vals = np.zeros(max_iterations)
        self.ave_personal_best_size = np.zeros(max_iterations)
        self.ave_personal_best_jaccard = np.zeros(max_iterations)

        # Average current swarm metrics
        self.ave_ff_vals = np.zeros(max_iterations)
        self.ave_position_size = np.zeros(max_iterations) 
        self.ave_jaccard = np.zeros(max_iterations)

        # Average Jaccard to bests
        self.ave_jaccard_to_global_best = np.zeros(max_iterations)
        self.ave_jaccard_to_personal_best = np.zeros(max_iterations)

        # Individual particle trajectories
        self.particle_positions = [
            [None for _ in range(self.max_iterations)] for _ in range(self.swarm_size)
        ]
        
        # This part is correct, no changes needed.
        self.particle_velocities_add = [
            [None for _ in range(self.max_iterations)] for _ in range(self.swarm_size)
        ]
        self.particle_velocities_sub = [
            [None for _ in range(self.max_iterations)] for _ in range(self.swarm_size)
        ]
        
        # Individual particle accuracy/coverage on test set (if applicable)
        self.accuracy = [[None for _ in range(self.max_iterations)] for _ in range(self.swarm_size)]
        self.coverage = [[None for _ in range(self.max_iterations)] for _ in range(self.swarm_size)]

    def update_accuracy(self, i, j, acc):
        self.accuracy[j][i] = acc

    def update_coverage(self, i, j, cov):
        self.coverage[j][i] = cov

    def update_global_best_ff_vals(self, i, f_global_best):
        self.global_best_ff_vals[i] = f_global_best

    def update_global_best_pos(self, i, global_best):
        self.global_best_pos[i] = global_best

    def update_global_best_size(self, i, global_best_len):
        self.global_best_size[i] = global_best_len

    def update_global_test_accuracy(self, i, acc):
        self.global_test_accuracy[i] = acc

    def update_global_test_coverage(self, i, cov):
        self.global_test_coverage[i] = cov

    def update_average_personal_best_ff_vals(self, i, ave_personal_best_fvals):
        self.ave_personal_best_ff_vals[i] = ave_personal_best_fvals

    def update_average_personal_best_size(self, i, ave_personal_best_size):
        self.ave_personal_best_size[i] = ave_personal_best_size

    def update_average_personal_best_jaccard(self, i, ave_personal_best_jaccard):
        self.ave_personal_best_jaccard[i] = ave_personal_best_jaccard

    def update_average_ff_vals(self, i, ave_ff_val):
        self.ave_ff_vals[i] = ave_ff_val
        
    def update_average_position_size(self, i, ave_pos_size):
        self.ave_position_size[i] = ave_pos_size

    def update_average_jaccard_index(self, i, ave_jaccard_index):
        self.ave_jaccard[i] = ave_jaccard_index

    def update_average_jaccard_to_global_best(self, i, ave_jacc_to_gbest):
        self.ave_jaccard_to_global_best[i] = ave_jacc_to_gbest
        
    def update_average_jaccard_to_personal_best(self, i, ave_jacc_to_pbest):
        self.ave_jaccard_to_personal_best[i] = ave_jacc_to_pbest
