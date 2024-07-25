# UTILITIES FOR THE ASSOCIATION RULES APP
import pandas as pd
import numpy as np
import math
import modules.constants as const

# Returns a table to be used as binarization mapping
def data_bin_template(df_int: pd.DataFrame)->pd.DataFrame:
    """
    df_int: a dataframe with all integer values, it can include nan entries

    Returns a dataftame with:
    index = columns names
    columns = [nan][min_df]....[max_df]

    where min_df = min(df), max_df = max(df)

    Cells: boolean
    """

    df = pd.DataFrame()
    df[const.col_varname] = df_int.columns.tolist()
    df[const.col_nan] = len(df)*[False]
    
    min_int = df_int.min().min()
    max_int = df_int.max().max()
    
    for i in range(min_int,max_int+1):
        df[const.suffix_colnum%(i)] = len(df)*[False]

    df = df.set_index(const.col_varname)

    return df    

# Binarize a dataframe of int to a dataframe with bool according to a specified mapping
def recode_int_bin(df_int, map_bin : pd.DataFrame)->pd.DataFrame:
    """
    df_int: the dtaframe containing int type only
    map_bin: the data frame with bool entry True for the modalitie (e.g. NaN, 1.. 10), and with df_int column names as index

    Return a data frame with 0,1 entries corresponding to the binarization in map_bin
    """

    data_bin = df_int.copy()

    # get the list of modes in the map_bin
    modeslist = list(map_bin.columns)
    # the first entry is for NaN, it it treated separately, so we only keep the int modes
    modeslist = modeslist[1:]
    # the numerical modes are stored as column names with "[x]": we drop both braces and convert back to int
    modeslist = [int(x.replace("[","").replace("]","")) for x in modeslist]

    varslist = list(map_bin.index.values)

    for idx, var in enumerate(varslist):
        check_modes = map_bin.loc[var,:]
        nan_map = check_modes[0]
        check_modes = check_modes[1:]
        var_dict = dict(zip(modeslist,check_modes))
        data_bin[var] = data_bin[var].map(var_dict)
        data_bin[var] = data_bin[var].replace(np.nan,nan_map)
    
    data_bin = data_bin.astype(int)

    return data_bin


def nr_rules(n: int, k: int)->int:
    '''
    The number of rules with k antecedents out of n variables: it s the binomial coeff: n!/(k!*(n-k)!)
    it returns
    '''
    return math.comb(n,k)

def nr_rules_total(max_antecedents: int, nr_vars: int)-> int:
    '''
    It returns the total number of rules obtained by considering from 1 up to max_antecedents
    '''
    N=0
    for k in range(1,max_antecedents+1):
        N+=nr_rules(nr_vars,k)
    
    return N

def rule_precondition(A_df: pd.DataFrame, antecedents_dict: dict)->np.array:
    '''
    Evaluates the precondition on a set of antecedents A, where
    A_df : the dataset restricted to variables {A} in the precondition.
    antecedetsn_dict: a dictionary containing dict[antecents_name]=target

    It returns an array of 0/1 that flags the records firing the precondition 

    '''
    # loop over variables in the precondition
    pre_arr = np.ones(A_df.shape[0],dtype=int)
    for A in A_df.columns:
        t = antecedents_dict[A]
        A_arr = np.array(A_df[A]==t)
        pre_arr = pre_arr*A_arr

    return pre_arr

def set_new_rule(max_antecedents,antecedents_list, consequent, rule_len, target_val, support_n, support_val, confidence_val, conviction_val, lift_val)->dict:
    '''
    Takes the results from a rule computation and store them into a dictionary
    '''
    rule_dict = {}
    for i in range(max_antecedents):
        key = f"antecedent_{i+1}"
        if i in range(len(antecedents_list)):
            value = antecedents_list[i]
        else:
            value = ''
        rule_dict[key] = value
    rule_dict[const.col_consequent]=consequent
    rule_dict[const.col_length]=rule_len
    rule_dict[const.col_target]=target_val
    rule_dict[const.col_supportN]=support_n
    rule_dict[const.col_support]=support_val
    rule_dict[const.col_confidence]=confidence_val
    rule_dict[const.col_conviction]=conviction_val
    rule_dict[const.col_lift]=lift_val

    return rule_dict


def get_items_freqency(rules_df: pd.DataFrame, antecedents_list: list)->pd.DataFrame:
    '''
    From the DataFrame containing the rules, it returns a data frame with the frequency each antecedent appears throughiut the whole set of rules
    '''
    freq_ant_df=pd.DataFrame()
    Ntot=rules_df.shape[0]

    # Find the columns that contain the substring antecedent_col
    cols_to_search = [col for col in rules_df.columns if const.col_antecedent in col]
    for antecedent_to_find in antecedents_list:
        rules_containing_antecedent = rules_df[rules_df[cols_to_search].isin([antecedent_to_find]).any(axis=1)]
        nrules=rules_containing_antecedent.shape[0]
        mean_lift=0
        if nrules>0:
            mean_lift=rules_containing_antecedent[const.col_lift].mean()
        antecedent_values={const.col_antecedent:antecedent_to_find,const.col_freq:nrules/Ntot,const.col_avglift:mean_lift}
        antecedent_df = pd.DataFrame([antecedent_values])
        freq_ant_df=pd.concat([freq_ant_df, antecedent_df], ignore_index=True)
        freq_ant_df.sort_values(by=[const.col_freq], inplace=True,ascending=False)
        freq_ant_df = freq_ant_df.reset_index(drop=True)
    return freq_ant_df
