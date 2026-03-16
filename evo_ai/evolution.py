import torch
import copy
import numpy as np

class GeneticAlgorithm:
    def __init__(self, template_model, pop_size=50, mutation_rate=0.01, mutation_power=0.02, device='cpu'):
        self.template_model = template_model
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_power = mutation_power
        self.device = device
        
        # Initialize population
        # We store the state_dicts.
        self.population = []
        base_state = template_model.state_dict()
        
        for _ in range(pop_size):
            self.population.append(copy.deepcopy(base_state))
            
    def perturb(self, state_dict):
        # Mutate weights
        new_state = copy.deepcopy(state_dict)
        for key in new_state:
            if 'weight' in key or 'bias' in key:
                # Add noise
                # Identify tensor type
                tensor = new_state[key]
                if tensor.is_floating_point():
                   noise = torch.randn_like(tensor) * self.mutation_power
                   # Apply mask based on mutation rate? 
                   # Classical GA: Randomly change some gene.
                   # ES: Change all genes slightly.
                   # Let's go with ES-style small perturbation to all for smoother landscapes.
                   # Or Sparse mutation.
                   
                   # Let's simple add noise to everything
                   new_state[key] = tensor + noise
        return new_state

    def get_population_models(self):
        # Return list of models with loaded weights
        # Actually to save memory, we might load them one by one in the eval loop.
        # But this function is for convenience if needed.
        pass

    def evolve(self, scores):
        # scores: list of (index, score)
        # Select top k
        # Simple Elitism: Top 10% survive. The rest are mutated versions of Top 10%.
        
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_k = int(self.pop_size * 0.2) # Top 20%
        if top_k < 1: top_k = 1
        
        elites = [self.population[i] for i, score in sorted_scores[:top_k]]
        
        # Print best score
        print(f"Best Score: {sorted_scores[0][1]:.4f}")
        
        next_gen = []
        # Elitism: keep elites
        next_gen.extend([copy.deepcopy(e) for e in elites])
        
        # Fill the rest with mutated elites
        while len(next_gen) < self.pop_size:
            parent = random.choice(elites)
            child = self.perturb(parent)
            next_gen.append(child)
            
        self.population = next_gen
        
        return sorted_scores[0][1] # Return best score

import random
