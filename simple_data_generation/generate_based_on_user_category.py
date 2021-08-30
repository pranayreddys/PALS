"""
This file is almost exactly the same as gen.py, but with slight modifications (introducing a category variable for each user).
The category variable determines the psychological response of a user to different nudges.
The current design is quite hardcoded, can be improved upon if this is a common use-case by introducing
category itself as a variable in the config file.
This code was primarily used to generate data and test out the reward predictor & nudge optimizer modules.
Please refer gen.py to understand the basic generation process and implementation.
"""
from pydantic import BaseModel, parse_file_as
import enum
from typing import Dict, List, Tuple, Optional
import numpy as np
import random
from copy import deepcopy
import pandas as pd
import argparse
import os

random.seed(0)
np.random.seed(0)

def sigmoid(x):
    return 1/(1+np.exp(-x))
# Assumptions Nudge[t] -> UAT[t] -> BP[t]
@enum.unique
class VariableName(str, enum.Enum):
    sleep_duration = 'sleep_duration'
    salt_intake = 'salt_intake'
    inertia = 'inertia'
    physiological_response = 'physiological_response'
    dbp = 'dbp'
    sbp = 'sbp'
    step_count = 'step_count'

@enum.unique
class DistributionType(str, enum.Enum):
    normal = 'normal'
    bernoulli = 'bernoulli'
    beta = 'beta'
    gamma = 'gamma'
    binomial = 'binomial'
    multinomial = 'multinomial'
    uniform = 'uniform'

@enum.unique
class CurveType(str, enum.Enum):
    unimodal = 'unimodal'

class VariableSpec(BaseModel):
    """
    This class describes configuration parameters required for UAT/BP generation. Key params:
    name: a string of type VariableName (which is an enum)
    init_distribution_type: Specifies the population-level distribution that a person's baseline numbers
    need to be sampled from
    init_params: Specifies the params required for init_distribution_type
    min_bound: The minimum possible value that can be taken by this variable
    max_bound: The maximum possible value that can be taken by this variable
    healthy_baseline: This is compulsory for UAT variables and optional otherwise. For UAT variables,
    this number is required for computing delayed effect of changes in UAT w.r.t BP.
    observation_noise_type: Note that even though a person has a certain UAT (e.g. 2000 steps),
    there could be day-to-day fluctuations through measurement noise and simple fluctuations in the person's behavior
    which cannot be predicted. This variable captures that distribution type.
    observation_noise_params: The parameters that need to be passed into the observation noise distribution. Noise will 
    be sampled with these params. 
    """
    name: VariableName
    init_distribution_type: DistributionType
    init_params: Dict[str, float]
    min_bound: float
    max_bound: float
    healthy_baseline: float = 0 # Compulsory parameter to be specified in config for UAT variables
    observation_noise_type: Optional[DistributionType] # Needs to be specified for UAT and BP
    observation_noise_params: Optional[Dict[str, float]] # Needs to be specified for UAT and BP
    
    def sample(self, n: int = 1):
        params = self.init_params.copy()
        params["size"] = n
        return np.clip(eval("np.random."+self.init_distribution_type)(**params), 
                        self.min_bound, self.max_bound).squeeze()

class Variable:
    """
    Create a variable object from the configuration provided in var_spec.
    These objects are used throughout while generating data.
    """
    def __init__(self, var_spec: VariableSpec):
        self.min_bound = var_spec.min_bound
        self.max_bound = var_spec.max_bound
        self.value = var_spec.sample()
        self.noise_type = var_spec.observation_noise_type
        self.noise_params = var_spec.observation_noise_params
        self.healthy_baseline = var_spec.healthy_baseline 
    
    def update(self, inc_value: float):
        self.value = min(max(self.min_bound, self.value+inc_value), self.max_bound)
    
    def observe(self):
        if self.noise_type == DistributionType.normal:
            sampled_value = self.value + (eval("np.random."+self.noise_type)(**self.noise_params))
        elif self.noise_type == DistributionType.beta:
            a = self.value*self.noise_params["sum_alpha_beta"]
            b = self.noise_params["sum_alpha_beta"] - a
            sampled_value = np.random.beta(a,b)
        else:
            raise NotImplementedError
        return min(max(self.min_bound, sampled_value),self.max_bound)
    
class Effect(BaseModel):
    """
    This class has the delayed effect model curve parameters for either
    a) Nudge - UAT
    b) UAT - BP
    The class is currently being initialized to generate a random unimodal
    curve. For more realistic curves, more curve types can be added, along
    with the required behaviour/params for the same.
    """
    lag: int
    curve_type: CurveType = CurveType.unimodal
    min_bound: float # assumed positive
    max_bound: float # assumed positive, represents only magnitude. Sign represented by negative
    return_to_mean: bool = True
    negative: Optional[bool] = False

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
                if self.negative:
                    effect = -effect 
                self.__dict__['effect'] = effect
        return effect

class EffectProfile(BaseModel):
    """
    This class maintains the Nudge, UAT -> Delayed Effect curve mapping.
    In addition to maintaining this mapping, it also performs the computation
    to generate new UAT/BP given older values and the Delayed Effect Model parameters
    """
    nudge_uat: Dict[int, Dict[VariableName, Effect]] # Action -> Dictionary[VariableName, Effect] mapping
    uat_bps: Dict[VariableName, Dict[VariableName, Effect]] 
    max_lag_nudge_uat: Optional[int] = 0 #HACK, Internal Parameter
    max_lag_uat_bp: Optional[int] = 0 #HACK, Internal Parameter
    uat_bps_array: Optional[str] = None #HACK, This is an internal numpy array param
    variable_specs: Optional[str] = None #HACK, This is an internal list of variablespec
    final_output_specs: Optional[str] = None #HACK, This is an internal list of variablespec


    def init_effect_profile(self, variable_specs: List[VariableSpec], final_output_specs: List[VariableSpec]):
        for v in self.nudge_uat.values():
            for e in v.values():
                self.max_lag_nudge_uat= max(self.max_lag_nudge_uat, e.lag)
        
        for v in self.uat_bps.values():
            for e in v.values():
                self.max_lag_uat_bp = max(self.max_lag_uat_bp, e.lag)
        
        # (UAT * lag uncoiled needs to be multiplied by final x (UAT * lag) 
        self.variable_specs = variable_specs
        self.final_output_specs = final_output_specs
        self.uat_bps_array = np.zeros((len(final_output_specs), len(variable_specs)*self.max_lag_uat_bp))
        for i,output_spec in enumerate(final_output_specs):
            for j,var_spec in enumerate(variable_specs): 
                if output_spec.name in self.uat_bps[var_spec.name]:
                    self.uat_bps_array[i,j::len(variable_specs)] = self.uat_bps[var_spec.name][output_spec.name].effect
    
    def update_uat(self, nudges: List[int], state: Dict[VariableName, Variable], inertia: Dict[int, float]):
        nudges = nudges[-min(self.max_lag_nudge_uat, len(nudges)):]
        updated_variables : Dict[VariableName, float] = {}
        for i, nudge in enumerate(nudges[::-1]):
            if not nudge==0:
                for variable, effect in self.nudge_uat[nudge].items():
                    if effect.lag > i:
                        if variable in updated_variables:
                            updated_variables[variable] += (inertia[nudge]* effect.effect[i])
                        else:
                            updated_variables[variable] = (inertia[nudge]*effect.effect[i])
        
        for variable, increment in updated_variables.items():
            state[variable].update(increment)
    
    def update_bp(self, uats: List[Dict[VariableName, Variable]], bps: Dict[VariableName, Variable]):
        updated_variables : Dict[VariableName, float] = {}
        uat_array = np.zeros((self.max_lag_uat_bp* len(uats[0])))
        for idx, uat in enumerate(uats[-1:-(self.max_lag_uat_bp+1):-1]):
            start_ind = idx*len(uat)
            for idx2, spec in enumerate(self.variable_specs):
                uat_array[start_ind+idx2] =  uat[spec.name].value - uat[spec.name].healthy_baseline
        increments = (self.uat_bps_array @ uat_array)
        assert len(increments.shape)==1 and increments.shape[0]==len(bps) 
        for idx, spec in enumerate(self.final_output_specs):
            bps[spec.name].update(increments[idx])
        
    def ret_effect_profile(self):
        ret_profile = []
        for nudge, cause_effect in self.nudge_uat.items():
            for variable, effect in cause_effect.items():
                profile = np.cumsum(effect.effect)
                for idx,curve_val in enumerate(list(profile)):
                    ret_profile.append({"time": idx, "cause": nudge, "effect_on": str(variable), "effect": curve_val})
        for uat,v in self.uat_bps.items():
            for bp, effect in v.items():
                profile = np.cumsum(effect.effect)
                for idx, curve_val in enumerate(list(profile)):
                    ret_profile.append({"time": idx, "cause": uat, "effect_on": str(bp), "effect": curve_val})
        return pd.DataFrame(ret_profile)
    
class NudgeGeneratorSpec(BaseModel):
    """
    Simply generates Study 1 like cycles,
    with number of actions, initial baseline period, cycling period and washout period as parameters
    """
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
    """
    Brings all the above class components together for a single user.
    """
    def __init__(self,  user_id: int, variable_specs: List[VariableSpec], 
                nudge_spec: NudgeGeneratorSpec,cause_effect: EffectProfile, 
                final_output_spec: List[VariableSpec],category: bool):
        self.variables : Dict[VariableName,Variable] = \
            {spec.name: Variable(spec) for spec in variable_specs}
        self.gen_data: List[Dict[VariableName, float]] = [] 
        self.nudges = nudge_spec.get_nudge_perm()
        self.cause_effect = cause_effect
        self.user_id = user_id
        self.final_outputs : Dict[VariableName, Variable] = \
            {spec.name: Variable(spec) for spec in final_output_spec}
        self.category = category

    def observe_variables(self):
        output1 =  {str(k): v.observe() for k,v in self.variables.items()}
        output1.update({(str(k)+"_noiseless"): v.value for k, v in self.variables.items()})
        output2 =  {str(k): v.observe() for k,v in self.final_outputs.items()}
        output2.update({(str(k)+"_noiseless"): v.value for k, v in self.final_outputs.items()})
        return output1, output2
    
    def gen(self):
        i = 0
        init_state = {str(k): v.value for k, v in self.variables.items()}
        inertia_array = [{1: 1.0, 2: 0.2}, {1: 0.1, 2: 1.0}]
        category_var = 0 if not self.category else 1
        uat_data = []
        bp_data = []
        past_uat= []
        past_bp = []
        while self.nudges[i]==0:
            uat, bp = self.observe_variables()
            past_uat.append(deepcopy(self.variables))
            uat_data.append(uat)
            bp_data.append(bp)
            i += 1
        
        while i<len(self.nudges):
            self.cause_effect.update_uat(self.nudges[:(i+1)], self.variables, inertia_array[category_var])
            past_uat.append(deepcopy(self.variables))
            self.cause_effect.update_bp(past_uat, self.final_outputs)
            uat, bp = self.observe_variables()
            uat_data.append(uat)
            bp_data.append(bp)
            i += 1
        

        for idx, d in enumerate(uat_data):
            d.update(bp_data[idx])
        ret_df = pd.DataFrame(uat_data)
        ret_df["nudge"] = self.nudges
        ret_df["time"] = np.array(list(range(ret_df.shape[0])))
        ret_df["user_id"] = self.user_id
        for k, v in init_state.items():
            ret_df["init_state_"+k] = v
        ret_df["category"] = category_var 
        return ret_df


class Main(BaseModel):
    variable_specs: List[VariableSpec]
    population_size: int
    nudge_spec: NudgeGeneratorSpec
    cause_effect: EffectProfile
    final_output_specs: List[VariableSpec]
    def generate_data(self):
        self.cause_effect.init_effect_profile(self.variable_specs, self.final_output_specs)
        people = [Person(user_id, self.variable_specs, self.nudge_spec, self.cause_effect, self.final_output_specs, user_id<(self.population_size//2)) \
                     for user_id in range(self.population_size)]
        
        return pd.concat([person.gen() for person in people]), self.cause_effect.ret_effect_profile()

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="scaled_config.json")
    parser.add_argument('--output_folder', default="user_features2")
    args = parser.parse_args()
    r = parse_file_as(Main, args.config_path)
    os.makedirs(args.output_folder, exist_ok=False)
    user_data, effect_profile = r.generate_data()
    user_data.to_csv(os.path.join(args.output_folder, "data.csv"))
    effect_profile.to_csv(os.path.join(args.output_folder, "effect_profile.csv"))
