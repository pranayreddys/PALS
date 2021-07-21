# {Init State {"variable": DistributionObj(DistType, Params)}, population_size }
# {Effect Profile {("cause" (variable) & "effect" (variable)): Effect}}
# {Nudge Cycle Config(num_actions, run_period, on_period, washout_time)}
# Generator(prev_state, last_few_nudges) return new_state (state: All Variables)
# Effect: lag, curve_type, min, max (unimodal hardcode)
# 
from pydantic import BaseModel, parse_file_as
import enum
from typing import Dict, List, Tuple, Optional
import numpy as np
import random
from copy import deepcopy
import pandas as pd
import argparse

random.seed(0)
np.random.seed(0)
@enum.unique
class VariableName(str, enum.Enum):
    sleep = 'sleep'
    salt = 'salt'

@enum.unique
class DistributionType(str, enum.Enum):
	normal = 'normal'
	bernoulli = 'bernoulli'
	beta = 'beta'
	gamma = 'gamma'
	binomial = 'binomial'
	multinomial = 'multinomial'

@enum.unique
class CurveType(str, enum.Enum):
    unimodal = 'unimodal'

class VariableSpec(BaseModel):
    name: VariableName
    init_distribution_type: DistributionType
    init_params: Dict[str, float]
    min_bound: float
    max_bound: float
    noise_type: DistributionType
    noise_params: Dict[str, float]
    
    def sample(self, n: int = 1):
        params = self.init_params.copy()
        params["size"] = n
        return np.clip(eval("np.random."+self.init_distribution_type)(**params), 
                        self.min_bound, self.max_bound).squeeze()

class Variable:
    def __init__(self, var_spec: VariableSpec):
        self.min_bound = var_spec.min_bound
        self.max_bound = var_spec.max_bound
        self.value = var_spec.sample()
        self.noise_type = var_spec.noise_type
        self.noise_params = var_spec.noise_params
    
    def update(self, inc_value: float):
        self.value = min(max(self.min_bound, self.value+inc_value), self.max_bound)
    
    def observe(self):
        return min(max(self.min_bound, self.value 
                        + (eval("np.random."+self.noise_type)(**self.noise_params)))
                        ,self.max_bound)
    
class Effect(BaseModel):
    lag: int
    curve_type: CurveType = CurveType.unimodal
    min_bound: float
    max_bound: float
    return_to_mean: bool = True

    @property
    def effect(self):
        effect = self.__dict__.get('effect')
        if effect is None:
            if self.curve_type==CurveType.unimodal:
                lag = self.lag-1 if self.return_to_mean else self.lag 
                e = np.random.uniform(self.min_bound, self.max_bound, lag)
                max_val = e.max()
                elems = e[e<max_val]
                elems = np.random.permutation(elems)
                assert (elems.shape[0]+1)==e.shape[0], "Floating point error"
                peak = np.random.randint(lag)
                ret_array = np.zeros(lag+1)
                ret_array[peak+1]= max_val
                ret_array[1:(peak+1)] = np.sort(elems[0:peak])
                if peak+2 < ret_array.shape[0]:
                    assert ret_array[(peak+2):].shape[0] == elems[peak:].shape[0]
                    ret_array[(peak+2):] = np.sort(elems[peak:])[::-1]
                if self.return_to_mean:
                    ret_array = np.append(ret_array, 0)
                assert ret_array.shape[0]== (self.lag+1)
                effect = ret_array[1:] - ret_array[0:-1] 
                self.__dict__['effect'] = effect
        return effect        

class EffectProfile(BaseModel):
    profiles: Dict[int, Dict[VariableName, Effect]]
    max_lag: int = 0
    def init_effect_profile(self):
        for v in self.profiles.values():
            for e in v.values():
                self.max_lag = max(self.max_lag, e.lag)
    
    def update_state(self, nudges, state: Dict[VariableName, Variable]):
        nudges = nudges[-min(self.max_lag, len(nudges)):]
        updated_variables : Dict[VariableName, float] = {}
        for i, nudge in enumerate(nudges[::-1]):
            if not nudge==0:
                for variable, effect in self.profiles[nudge].items():
                    if effect.lag > i:
                        if variable in updated_variables:
                            updated_variables[variable] += effect.effect[i]
                        else:
                            updated_variables[variable] = effect.effect[i]
        
        for variable, increment in updated_variables.items():
            state[variable].update(increment)

class NudgeGeneratorSpec(BaseModel):
    num_actions: int
    init_period: int
    cycle_period: int
    washout_period: int

    def get_nudge_perm(self):
        actions = list(range(1,self.num_actions+1))
        random.shuffle(actions)
        ret_nudges = [] 
        for _ in range(self.init_period):
            ret_nudges.append(0)
        for a in actions:
            for _ in range(self.cycle_period):
                ret_nudges.append(a)
            for _ in range(self.washout_period):
                ret_nudges.append(0)        
        return ret_nudges
      
class Person:
    def __init__(self, variable_specs: List[VariableSpec], nudge_spec: NudgeGeneratorSpec,
                cause_effect: EffectProfile, user_id: int):
        self.variables : Dict[VariableName,Variable] = \
            {spec.name: Variable(spec) for spec in variable_specs}
        self.gen_data: List[Dict[VariableName, float]] = [] 
        self.nudges = nudge_spec.get_nudge_perm()
        self.cause_effect = cause_effect
        self.user_id = user_id

    def observe_variables(self):
        return {str(k): v.observe() for k,v in self.variables.items()}

    def gen(self):
        i = 0
        while self.nudges[i]==0:
            self.gen_data.append(self.observe_variables())
            i += 1
        
        while i<len(self.nudges):
            self.cause_effect.update_state(self.nudges[:(i+1)], self.variables)
            self.gen_data.append(self.observe_variables())
            i += 1
        

        ret_df = pd.DataFrame(self.gen_data)
        ret_df["nudge"] = self.nudges
        ret_df["time"] = np.array(list(range(ret_df.shape[0])))
        ret_df["user_id"] = self.user_id
        return ret_df


class Main(BaseModel):
    variable_specs: List[VariableSpec]
    population_size: int
    nudge_spec: NudgeGeneratorSpec
    cause_effect: EffectProfile
    def generate_data(self):
        self.cause_effect.init_effect_profile()
        people = [Person(self.variable_specs, self.nudge_spec, self.cause_effect, user_id) \
                     for user_id in range(self.population_size)]
        
        return pd.concat([person.gen() for person in people])

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    r = parse_file_as(Main, args.config_path)
    t = r.generate_data()
    t.to_csv(args.output_path)
