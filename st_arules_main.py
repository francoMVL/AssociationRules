# ASSOCIATION RULES APPLICATION

import pandas as pd
import streamlit as st
import numpy as np
import itertools
import math
import altair as alt
import io

import os

import logging
from logging.handlers import RotatingFileHandler

import modules.constants as const
import modules.utils_rules as urules

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================
NAN_REPLACE = -999
EMPTY_DF = pd.DataFrame(data=[],columns=[])
KEYS_INIT = {'keyid':None, 'numvars':None}
EMPTY_INPUT = {'meta':pd.DataFrame,'int':pd.DataFrame}
RULES_INIT = {'max_ant':1,'min_support':0.2,'min_lift':1.1,'top_pctile':0.1}

# ------------------------------------------------------------------------------
# SET UP LOGGING
# ------------------------------------------------------------------------------
# Set up logging with rotation: set up for local deployment and debugging
#logging.basicConfig(level=logging.INFO, 
#                    format='%(asctime)s - %(levelname)s - %(message)s',
#                    handlers=[
#                        RotatingFileHandler("app.log", maxBytes=10000, backupCount=5),  # Rotate logs
#                        logging.StreamHandler()  # Log to the console
#                    ])

# Set up logging: set up for web deployment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# INITiALIZATION SESSION STATE
# ------------------------------------------------------------------------------
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"]=None
if "data_raw" not in st.session_state:
    st.session_state["data_raw"]=pd.DataFrame()
if "data_int" not in st.session_state:
    st.session_state["data_int"]=pd.DataFrame()
if "map_bin" not in st.session_state:
    st.session_state["map_bin"]=pd.DataFrame()
if "data_bin" not in st.session_state:
    st.session_state["data_bin"]=pd.DataFrame()
if "rules_params" not in st.session_state:
    st.session_state["rules_params"]=RULES_INIT
if "rules" not in st.session_state:
    st.session_state["rules"] = pd.DataFrame()




# ------------------------------------------------------------------------------
# RESET FUNCTIONS
# ------------------------------------------------------------------------------
def reset_status():
    st.session_state["uploaded_file"]=None
    st.session_state["data_raw"]=pd.DataFrame()
    st.session_state["data_int"]=pd.DataFrame()
    st.session_state["map_bin"]=pd.DataFrame()
    st.session_state["data_bin"]=pd.DataFrame()
    st.session_state["rules_params"]=RULES_INIT
    st.session_state["rules"] = pd.DataFrame()
    logging.info("CLEARED CACHE")

def reset_map_bin():
    st.session_state["map_bin"] = urules.data_bin_template(df_int=st.session_state["data_int"])

# On_change call-back to be used to refresh a dataframe edited in a st.data_editor
def on_change_map(df: pd.DataFrame):
    st.session_state["map_bin"]=df.copy()


# ------------------------------------------------------------------------------
# CHECK FUNCTIONS
# ------------------------------------------------------------------------------
# check whether a file has been uploaded
def check_uploaded_file():
    return not st.session_state["uploaded_file"]==None
# check whether the data have been loaded
def check_data_load():
    return not st.session_state["data_raw"].empty
# check whether the conversion to numerical int exists
def check_data_int():
    return not st.session_state["data_int"].empty
# chec whether the data binarized is available
def check_data_bin():
    return not st.session_state["data_bin"].empty
# check whether the rules have been computed
def check_rules_computed():
    return not st.session_state["rules"].empty



# ------------------------------------------------------------------------------
# USER FUNCTIONS
# ------------------------------------------------------------------------------
# load the dataset from the uploaded_file info
def load_data():
    filename, file_extension = os.path.splitext(st.session_state["uploaded_file"].name)
    try:
        if file_extension==".dta":
            st.session_state["data_raw"]=pd.read_stata(st.session_state["uploaded_file"])
        elif file_extension==".xlsx":
            st.session_state["data_raw"]=pd.read_excel(st.session_state["uploaded_file"])
        elif file_extension==".csv":
            st.session_state["data_raw"]=pd.read_csv(st.session_state["uploaded_file"])
        else:
            st.session_state["data_raw"]=pd.DataFrame()
        logging.info(f"data loaded from {file_extension} file")
    except Exception as e:
        st.error(f"An error occurred while loading file: {e}")
        logging.error(f"An error occurred while loading file: {e}")

# convert all variables passed into int format: raise an exception when the conversion fails
# Return: the list of variables that could not be converted to int
def typecast_to_int(varslist: list[str])->pd.DataFrame:
    df = st.session_state["data_raw"].copy()
    df_int = pd.DataFrame()
    int_err = []
    for var in varslist:
        values = df[var].values
        try:
            int_values = np.array(values, dtype='int')
            df_int[var]=int_values
            logging.info(f"variable: {var} converted to int and added to df")
        except Exception as e:
            int_err.append(var)
            logging.info(f"An error occurred in int conversion for {var}: {e}")
    
    st.session_state["data_int"]=df_int.copy()
    st.session_state["map_bin"] = urules.data_bin_template(st.session_state["data_int"]).copy()
    return int_err

# data binarization: apply the current map_bin to the data_in
def data_binarization():
    data_bin = urules.recode_int_bin(df_int=st.session_state["data_int"],map_bin=st.session_state["map_bin"])
    st.session_state["data_bin"] = data_bin.copy()

# Rules comutation loop: updates the rules and a progress bar on the total number of rules to be computed
def rule_computation(oucome_dict: dict, antecedents_dict: dict):
    '''
    It reads the rules_params settings and the variables spectifications in the two dictionaries outcome_dict, antecedents_dict
    and compute all the expected rules.

    It returs an increment to be used for a progress bar
    '''
    rules_df = EMPTY_DF

    antecedents_list = list(antecedents_dict.keys())
    max_ant = st.session_state['rules_params']['max_ant']

    data_bin = st.session_state['data_bin']

    Ntot = data_bin.shape[0]

    # rule: A -> B where B = outcome and A = {subset of antecedents list}

    # POSTCONDITION subset of data corresponding to outcome = target
    B = outcome_dict['out_var']
    tB = outcome_dict['out_target']
    B_arr = np.array(data_bin[B]==tB)
    # support of B: P(B)
    sB = B_arr.sum()/Ntot

    # set te progress bar parameter
    rule_id = 0
    rule_increment = 1/(urules.nr_rules_total(max_ant,len(antecedents_list))+1)
    # loop over all the combinations containing 1,2,... max_ant antecedents
    for i in range(1,max_ant+1):
        comb = itertools.combinations(antecedents_list,i)
        list_comb = list(comb)
        # te length of the rule = # antecedents considered: i:1...max_ant
        rule_len = i
        # loop within the set of combinations: each is a list of antecedents
        for k in range(len(list_comb)):
            rule_support = 0
            rule_confidence = 0
            rule_conviction = 0
            rule_list = 0
            # Analysing the current rule: k is the current set of antecedents
            A = list(list_comb[k])
            # Temp dataframe with columns = current antecedents: a subset of the Dataframe where B = target
            # PRECONDITION : subset of data on the subset {A}
            A_df = data_bin[A]
            A_arr = urules.rule_precondition(A_df,antecedent_dict)
            # data for Pre and Post condition be true
            nA = A_arr.sum()
            sA = nA/Ntot
            AB_arr = A_arr*B_arr
            nAB = AB_arr.sum()
            sAB = nAB/Ntot
            rule_support = sAB
            # continue the computation only if the rule suport is above the min fixed in the paramemeters
            if sAB>=st.session_state['rules_params']['min_support']:
                rule_confidence = rule_support/sA
                rule_conviction = np.inf
                rule_lift = rule_confidence/sB
                # conviction is calculated only if lift > threshold
                if rule_lift>=st.session_state['rules_params']['min_lift']:
                    if rule_confidence<1:
                        rule_conviction = (1-sB)/(1-rule_confidence)
                    # append the results with the current rule
                    newrule_dict = urules.set_new_rule(max_ant,A,B,rule_len,tB,nAB,rule_support,rule_confidence,rule_conviction,rule_lift)
                    new_rule_df = pd.DataFrame([newrule_dict])
                    rules_df = pd.concat([rules_df,new_rule_df],ignore_index=True)
            rule_id+=rule_increment
            rules_bar.progress(rule_id, text=const.progress_text)
    if rules_df.shape[0]>0:
        rules_df.index.name = 'ruleid'
        # select rules within the top_percentile_lift: sort by lift in descending order
        rules_df.sort_values(by=[const.col_lift], inplace=True,ascending=False)
        nr=math.ceil(rules_df.shape[0]*st.session_state['rules_params']['top_pctile'])
        rules_df=rules_df.head(nr)

    st.session_state['rules'] = rules_df.copy()
    

        
# ------------------------------------------------------------------------------
# PAGE TITLE
st.set_page_config(page_title='Associaiton Rules', page_icon=":microscope:",layout="wide")
st.title('Association Rules Analysis')



# ------------------------------------------------------------------------------
# APPLICATION TABS
# ------------------------------------------------------------------------------

tab_datamap, tab_rules, tab_download = st.tabs(["Data mapping","Rules","Save results"])

# ------------------------------------------------------------------------------
# TAB: DATA MAPPING

with tab_datamap:
    st.button("CLEAR CACHE",on_click=reset_status)
    with st.expander("Load data"):
        # file_uploader
        st.session_state['uploaded_file'] = st.file_uploader("Browse for data", type=["dta","xlsx","csv"])
        logging.info("File uploaded into app")
        # load button
        st.button("Load data", on_click=load_data,disabled=not check_uploaded_file())
        n_rec, n_vars = st.session_state["data_raw"].shape
        st.write(f"loaded {n_rec} rows and {n_vars} columns from file")
        logging.info(f"loaded {n_rec} rows and {n_vars} columns from file")
        st.data_editor(st.session_state["data_raw"],disabled=True,key="edit_raw")

    with st.expander("Select variables for association rules"):
        var_select = []
        if not check_data_load():
            st.write("No data was loaded.")
        else: 
            var_select = st.multiselect("Select variables for rules (must be numerical int)",options=st.session_state["data_raw"].columns.tolist())
            n_vars = len(var_select)
            st.write("Variables selected: ",n_vars)
            if len(var_select)>0:
                logging.info(f"selected {n_vars} variables for rules")
                if st.button("Convert selection to numerical"):
                    # type-cast the selected variables to numerical (automatically intialize the map_bin template)
                    err_list = typecast_to_int(varslist=var_select)
                    st.write("Errors in numerical conversion: ",len(err_list))
                    st.write("The following variables could not be converted to int:")
                    st.write(err_list)
                    st.write("Data with int variables:")
                    st.data_editor(st.session_state["data_int"],disabled=True,key="edit_int")

    with st.expander("Variables Binarization"):
        if not check_data_int():
            st.write("No variable was selected.")
        else:
            st.button("Reset binarization",on_click=reset_map_bin)
            col1, col2 = st.columns([10,8])
            with col1:
                st.write("Check modalities to binarize (checked->1, unckecked->0)")
                data_map_df = st.session_state["map_bin"]
                data_map_df = st.data_editor(st.session_state["map_bin"],on_change=on_change_map,args=[data_map_df])
                st.session_state["map_bin"] = data_map_df
            with col2:
                st.write("Frequency on chosen binarization")
                data_binarization()
                data_bin = st.session_state["data_bin"]
                df =  pd.DataFrame({const.col_varname:data_bin.columns,const.col_freq:data_bin.sum().values/data_bin.shape[0],
                                    const.col_target:data_bin.shape[1]*[True]})
                edited_bin_df = st.data_editor(df,hide_index=True,disabled=[const.col_varname,const.col_freq],
                                        column_config={
                                            const.col_freq:st.column_config.ProgressColumn(const.col_freq,min_value=0,max_value=1,help="Frequency is the support for the current binarization"),
                                            const.col_target:st.column_config.CheckboxColumn(const.col_target,help='If unchecked, target is 0.')})

# ------------------------------------------------------------------------------
# TAB: RULES COMPUTATION
with tab_rules:
    if check_data_bin():
        col1, col2, col3, col4  = st.columns([2,5,7,5])
        with col1:
            st.write('SET RULES PARAMETERS')
            st.session_state['rules_params']['max_ant']= st.number_input('Max number of antecedents:',min_value=1,max_value=edited_bin_df.shape[0]-1,help='Min number of antecedents: 1 Max < number of variables in the data.')
            st.session_state['rules_params']['min_support']=st.slider('Min rules support:',min_value=0.0,max_value=1.0,value=st.session_state['rules_params']['min_support'],help='Rule support is the occurrence of A->B in the dataset: Pr(A & B). Only rules above the min support will be retained.')
            st.session_state['rules_params']['min_lift'] = st.slider('Min rules lift:',min_value=1.0,max_value=2.0,value=st.session_state['rules_params']['min_lift'],help='Lift for the rule A->B is a measure of likelihood of association: Pr(A & B)/[Pr(A)Pr(B)]. Only rules with lift above the min lift will be retained.')
            st.session_state['rules_params']['top_pctile']  = st.slider('Lift top percentile:',min_value=0.01,max_value=0.8,value=st.session_state['rules_params']['top_pctile'],help='Rules with lift above the top percentile will be retained')

        with col2:
            st.write('OUTCOME SELECTION')
            outcome_var = st.selectbox('Select the outcome variable:',options=edited_bin_df[const.col_varname].values.tolist())
            outcome_target = int(edited_bin_df[edited_bin_df[const.col_varname]==outcome_var][const.col_target].values[0])
            antecedent_df = edited_bin_df[edited_bin_df[const.col_varname]!=outcome_var]
            cols = [const.col_varname,const.col_freq,const.col_target]
            antecedent_df[const.col_target]=antecedent_df[const.col_target].apply(lambda x:int(x))
            antecedent_df[const.col_antecedent]=[True]*antecedent_df.shape[0]
            st.write('Choose the set of antecedents:')
            antecedent_select = st.data_editor(antecedent_df,hide_index=True,disabled=[const.col_varname,const.col_freq,const.col_target],
                                                column_config={
                                                    const.col_freq:st.column_config.ProgressColumn(const.col_freq,min_value=0,max_value=1),
                                                    const.col_antecedent:st.column_config.CheckboxColumn(const.col_antecedent+'?')})
            st.write('Nr. possible antecedents: ',antecedent_select[const.col_antecedent].sum())
            st.write('Nr. rules analyzed: %d'%urules.nr_rules_total(max_antecedents=st.session_state['rules_params']['max_ant'],nr_vars=antecedent_select[const.col_antecedent].sum()))
            
            antecedents_list = list(antecedent_select[antecedent_select[const.col_antecedent]==True][const.col_varname].values)
            antecedents_targets = list(antecedent_select[antecedent_select[const.col_antecedent]==True][const.col_target].values)
            # put together all vars: outcome + selected antecedents and the selected target: put it into a dictionary structure
            # the 2 dscionaries are passed as input for the rules evaluation
            outcome_dict = {'out_var':outcome_var, 'out_target' :outcome_target}
            antecedent_dict = dict(zip(antecedents_list,antecedents_targets))

        with col3:
            run_rules = st.button('RUN RULES')
            rules_bar = st.progress(0,text=const.progress_text)
            if run_rules:
                rule_computation(oucome_dict=outcome_dict,antecedents_dict=antecedent_dict)
                rules_bar.empty()
            if check_rules_computed():
                st.write(st.session_state['rules'])
                st.text('Computed %d rules'%st.session_state['rules'].shape[0])

        with col4:
            if check_rules_computed():
                st.write('Antecedents frequency')
                freq_df = urules.get_items_freqency(st.session_state['rules'],antecedents_list)
                freq_df = freq_df.sort_values(by=[const.col_freq],ascending=False)
                #bars_freq = (alt.Chart(freq_df).mark_bar().encode(y=const.col_antecedent,x=const.col_freq))
                #st.altair_chart(bars_freq,use_container_width=True)
                edited_freq_df = st.data_editor(freq_df,hide_index=True,disabled=[const.col_varname,const.col_freq,const.col_avglift],
                                       column_config={
                                           const.col_freq:st.column_config.ProgressColumn(const.col_freq,min_value=0,max_value=1,help="Frequency of variable recurrence over the rules set"),
                                           })

        
# ------------------------------------------------------------------------------
# TAB: RULES COMPUTATION            
with tab_download:
    if not check_rules_computed():
        st.write("No results to download")
    else:
        st.write('Download the following results as Excel workbook')
        sheet_mapping='data_map'
        sheet_support='vars_support'
        sheet_rules='rules'
        sheet_freqs='var_freq'

        st.write(f"Data mapping to sheet: ............... {sheet_mapping}")
        st.write(f"Variables support to sheet: .......... {sheet_support}")
        st.write(f"Association rules to sheet: .......... {sheet_rules}")
        st.write(f"Variables frequencies to sheet: ... {sheet_freqs}")
        st.write("")

        filename = st.text_input('Enter file name (default: rules.xlsx)','rules.xlsx')

        # Create a BytesIO buffer to save the Excel file in memory
        buffer = io.BytesIO()

        # Write the dataframes to separate sheets in the same Excel file
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state["map_bin"].to_excel(writer, index=True, sheet_name=sheet_mapping)
            edited_bin_df.to_excel(writer, index=False, sheet_name=sheet_support)
            st.session_state['rules'].to_excel(writer, index=False, sheet_name=sheet_rules)
            freq_df.to_excel(writer, index=False, sheet_name=sheet_freqs)

        # Move the buffer position to the beginning
        buffer.seek(0)

        # Provide a download button for the Excel file
        if st.download_button(
            label="Download Results as Excel with Multiple Sheets",
            data=buffer,
            file_name=filename,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ):
            st.write("Results downloaded.")






