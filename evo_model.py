import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed


def load_and_split_replicates(file_path='Chr5 and Chr6 aneuploidies instability.xlsx', well_id_sheet = 'Chr5 CNV BFP'):
    """
    Loads data for a specific strain (sheet) and splits it into replicates
    formatted for ABC inference.

    Parameters:
    -----------
    file_path : str
        Path to the Excel file.
    well_id_sheet : str
        The specific sheet name (e.g., 'Chr5 CNV BFP').
        This is used to identify the dataset AND determine BFP vs GFP logic.

    Returns:
    --------
    replicate_names : list of str
        Simplified replicate IDs (e.g., 'C9_1').
    replicate_data : np.ndarray
        Array of shape (N_replicates, 6, 3) containing [Tri, WT, LOH].
        Normalized so each timepoint sums to 1.0.
    """
    
    # 1. Load the specific sheet directly
    try:
        df = pd.read_excel(file_path, sheet_name=well_id_sheet)
    except Exception as e:
        print(f"Error loading sheet '{well_id_sheet}': {e}")
        return None, None

    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Determine Logic based on the 'well' argument (Sheet Name)
    # If "BFP" is in the name -> ABB Strain (Tri=BFP, LOH=BFP Only)
    # If "GFP" is in the name -> AAB Strain (Tri=GFP, LOH=GFP Only)
    if 'BFP' in well_id_sheet:
        tri_col = 'CNV BFP'
        wt_col  = '1:1 Ratio'
        loh_col = 'BFP Only' # Loss of Green -> BB (BFP Only)
    elif 'GFP' in well_id_sheet:
        tri_col = 'CNV GFP'
        wt_col  = '1:1 Ratio'
        loh_col = 'GFP Only' # Loss of Blue -> AA (GFP Only)
    else:
        raise ValueError("Sheet name must contain 'BFP' or 'GFP' to determine aneuploidy type.")

    # 3. Define Time Points (Passages)
    # We enforce this exact order. Missing passages will be NaNs (or handled).
    target_passages = [0, 1, 3, 5, 7, 10]
    
    replicate_names = []
    data_list = []
    
    # 4. Split by well_ID
    # The 'well_ID' column in the CSV usually looks like "Chr5 CNV BFP C9_1"
    for well, group in df.groupby('well_ID'):
        
        # Simplify Name: Take last part "C9_1"
        short_name = str(well).split(' ')[-1]
        
        # Pivot: Index=Passage, Col=Population
        pivot = group.pivot_table(index='Passage', columns='Population', values='population fraction')
        
        # Reindex to force our specific time structure
        pivot = pivot.reindex(target_passages)
        
        # Extract columns strictly in [Tri, WT, LOH] order
        # Use .get(col, 0) to return 0s if a column is missing (e.g. no LOH yet)
        tri_vals = pivot.get(tri_col, pd.Series(0, index=target_passages)).fillna(0)
        wt_vals  = pivot.get(wt_col,  pd.Series(0, index=target_passages)).fillna(0)
        loh_vals = pivot.get(loh_col, pd.Series(0, index=target_passages)).fillna(0)
        
        # Stack into a (6, 3) matrix
        raw_matrix = np.column_stack((tri_vals, wt_vals, loh_vals))
        
        # 5. Normalize (CRITICAL STEP)
        # Ensure every row sums to exactly 1.0
        row_sums = raw_matrix.sum(axis=1)
        
        # Avoid division by zero (if a row is all 0s, keep it 0s or handle)
        # We replace 0 sums with 1 to avoid NaN, resulting in a [0,0,0] row
        safe_sums = np.where(row_sums == 0, 1.0, row_sums)
        
        norm_matrix = raw_matrix / safe_sums[:, np.newaxis]
        
        replicate_names.append(short_name)
        data_list.append(norm_matrix)
        
    return replicate_names, np.array(data_list)



def load_fitness_constants(file_path='Relative fitness of aneuploidy strains.xlsx', chromosome='Chr5', strain_type='BFP'):
    """
    Parses the fitness file to extract the 12 specific fitness values for a given 
    Chromosome and Strain (BFP/GFP).
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV or Excel file.
    chromosome : str
        'Chr5' or 'Chr6' (looks for this string in the first column).
    strain_type : str
        'BFP' (Cols 6-17) or 'GFP' (Cols 18-29).
        
    Returns:
    --------
    np.array
        Array of 12 fitness values.
    """
    try:
        # Load data (handle CSV or Excel)
        df = pd.read_excel(file_path, header=None, index_col=0)
            
        # 1. Find the row corresponding to the chromosome
        # The first column (index 0) contains names like "Chr5 " or "Chr6 "
        # We strip whitespace to be safe.
        mask = df.index.astype(str).str.strip() == chromosome
        
        if not mask.any():
            raise ValueError(f"Chromosome '{chromosome}' not found in file '{file_path}'. Available: {df.index.unique()}")
            
        # Get the specific row (series)
        row_data = df.loc[mask].iloc[0]
        
        # 2. Select columns based on strain type
        # BFP: Columns 6 to 17 (12 cols)
        # GFP: Columns 18 to 29 (12 cols)
        if 'BFP' in strain_type:
            fitness_values = row_data[6:18].values
        elif 'GFP' in strain_type:
            fitness_values = row_data[18:30].values
        else:
            raise ValueError("strain_type must be 'BFP' or 'GFP'")
            
        # Convert to numeric, forcing errors to NaN (then check)
        fitness_values = pd.to_numeric(fitness_values, errors='coerce')
        
        if np.isnan(fitness_values).any():
            print("Warning: NaNs found in fitness values. Filling with mean.")
            fitness_values = np.nan_to_num(fitness_values, nan=np.nanmean(fitness_values))
            
        return fitness_values

    except Exception as e:
        print(f"Error loading fitness file: {e}")
        return None
    

   

def simulator(mu_tri, mu_loh, w_tri, w_loh, p_0 = [1,0,0], N_e=100_000, generations = 100, ret_gens = [0,10,30,50,70,100], noise=0):
    
    mu_tri = 10**mu_tri # for optimization

    N = np.array(np.array(p_0) * N_e, dtype=int)
    S = np.diag([w_tri, 1, w_loh])
    M = np.array([[1-mu_tri, 0, 0],
                 [2/3 * mu_tri, 1, 0],
                 [1/3 * mu_tri, 0, 1]])
    G = S@M # mutate then select
    
    p = [N / N.sum()]
    for i in range(generations+1):
        N_t = G @ N
        N_t = np.random.multinomial(N_e, N_t / N_t.sum()) / N_e
        p.append(N_t)
        N = np.array(N_t * N_e, dtype=int)

    ret = np.array(p)[ret_gens]
    if noise>0:
        ret = ret + np.random.normal(0, noise, ret.shape)
    
    return ret


def mutation_simulator(mu_tri, mu_loh, w_loh, w_tri, p_0 = [1,0,0], N_e=100_000, generations = 100, ret_gens = [0,10,30,50,70,100], noise=0):
    
    mu_tri = 10**mu_tri # for optimization
    mu_loh = 10**mu_loh # for optimization

    N = np.array(np.array(p_0) * N_e, dtype=int)
    S = np.diag([w_tri, 1, w_loh])
    M = np.array([[1-mu_tri-mu_loh, 0, 0],
                 [mu_tri, 1, 0],
                 [mu_loh, 0, 1]])
    G = S@M # mutate then select
    
    p = [N / N.sum()]
    for i in range(generations+1):
        N_t = G @ N
        N_t = np.random.multinomial(N_e, N_t / N_t.sum()) / N_e
        p.append(N_t)
        N = np.array(N_t * N_e, dtype=int)

    ret = np.array(p)[ret_gens]
    if noise>0:
        ret = ret + np.random.normal(0, noise, ret.shape)
    
    return ret



def _worker_single_dataset(replicate_name, target_data, n_simulations, quantile):
    """
    Worker function that runs the complete ABC inference for ONE replicate.
    This runs on a single core.
    """
    # 1. Setup Priors
    # Sampling in Log10 space as indicated by your bounds (-5 to -2)
    prior_log_mu = np.random.uniform(low=-5, high=-2, size=n_simulations)
    prior_w_tri  = np.random.uniform(low=0.8, high=1.05, size=n_simulations)
    prior_w_loh  = np.random.uniform(low=0.8, high=1.05, size=n_simulations)
    
    # 2. Get Starting Conditions specific to this replicate
    # Normalize P0 just in case
    p_0 = target_data[0] / target_data[0].sum()
    
    distances = np.zeros(n_simulations)
    
    # 3. Run Simulation Loop (Serial execution for this core)
    for i in range(n_simulations):
        # Transform Log10 mu -> Linear mu for the simulator
        
        
        sim_output = simulator(
            mu=prior_log_mu[i], 
            w_tri=prior_w_tri[i], 
            w_loh=prior_w_loh[i], 
            p_0=p_0,  # Specific to this replicate
            N_e=100_000, 
            generations=100, 
            ret_gens=[0,10,30,50,70,100], 
            noise=0
        )
        
        # Calculate RMSE
        diff = sim_output - target_data
        rmse = np.sqrt(np.mean(diff**2))
        distances[i] = rmse
    
    # 4. Filter Results (Rejection Step)
    cutoff = np.quantile(distances, quantile)
    accepted_mask = distances <= cutoff
    
    # Create the Posterior DataFrame
    posterior = pd.DataFrame({
        'Replicate': replicate_name,      # Tag results with replicate name
        'mu': prior_log_mu[accepted_mask],
        'w_tri': prior_w_tri[accepted_mask],
        'w_loh': prior_w_loh[accepted_mask],
        'rmse': distances[accepted_mask]
    })
    
    return posterior

def run_abc_batch_parallel(replicate_names, replicate_data_array, n_simulations=50_000, quantile=0.002, n_jobs=12):
    """
    Distributes 12 different ABC tasks across 12 cores.
    
    Parameters:
    - replicate_names: List of names (e.g., ['C9_1', 'C9_2', ...])
    - replicate_data_array: np.array of shape (12, 6, 3)
    """
    print(f"Starting parallel batch inference on {len(replicate_names)} replicates...")
    print(f"Configuration: {n_simulations} sims/rep | {n_jobs} cores | Quantile: {quantile}")

    # Use joblib to map the worker function to the datasets
    # zip() pairs the name with the data matrix
    results_list = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_worker_single_dataset)(name, data, n_simulations, quantile)
        for name, data in zip(replicate_names, replicate_data_array)
    )
    
    # Combine all individual posteriors into one giant DataFrame
    # This makes plotting with seaborn (hue='Replicate') extremely easy later
    full_posterior = pd.concat(results_list, ignore_index=True)
    
    print("Batch inference complete.")
    return full_posterior





def _abc_worker(replicate_name, target_data, simulator_func, n_sims, quantile, fixed_w_tri=None, is_mutation_model=False):
    """
    Runs ABC for a single replicate with flexible parameter handling.
    
    Parameters:
    - fixed_w_tri: If float, w_tri is fixed. If None, w_tri is inferred.
    - is_mutation_model: If True, samples mu_loh (mu2) and calls 
                         simulator(mu_tri, mu_loh...).
                         If False, calls simulator(mu...).
    """
    
    # --- 1. Common Priors ---
    
    # A. Primary Loss Rate (mu / mu_tri): Log-Uniform 10^-5 to 10^-2
    log_mu = np.random.normal(-3, 0.5, n_sims)
    prior_mu_1 = log_mu
    
    # B. LOH Fitness (w_loh): Uniform 0.8 to 1.05
    prior_w_loh = np.random.uniform(0.8, 1.05, n_sims)
    
    # C. Aneuploid Fitness (w_tri): Either Fixed or Inferred
    if fixed_w_tri is not None:
        # Fixed Case: Create array of constant values
        prior_w_tri = np.full(n_sims, fixed_w_tri)
    else:
        # Inferred Case: Uniform 0.8 to 1.05
        # prior_w_tri = np.random.normal(0.92, 0.02, n_sims)
        prior_w_tri = np.random.uniform(0.8, 1.05, n_sims)

    # --- 2. Model-Specific Handling ---
    
    distances = np.zeros(n_sims)
    p_0 = target_data[0] / target_data[0].sum()
    
    if is_mutation_model:
        # === MUTATION SIMULATOR ===
        # Requires: mu_tri, mu_loh, w_tri, w_loh
        
        # Sample Secondary Rate (Formation): Log-Uniform 10^-5 to 10^-2 (Same dist as loss)
        log_mu2 = np.random.normal(-3, 0.5, n_sims)
        prior_mu_2 = log_mu2
        
        for i in range(n_sims):
            sim_out = simulator_func(
                mu_tri=prior_mu_1[i],
                mu_loh=prior_mu_2[i],
                w_tri=prior_w_tri[i],
                w_loh=prior_w_loh[i],
                p_0=p_0,
                generations=100
            )
            distances[i] = np.sqrt(np.mean((sim_out - target_data)**2))
            
        res_mu_2 = prior_mu_2 # Save for output

    else:
        # === STANDARD SIMULATOR ===
        # Requires: mu, w_tri, w_loh
        
        for i in range(n_sims):
            sim_out = simulator_func(
                mu=prior_mu_1[i],
                w_tri=prior_w_tri[i],
                w_loh=prior_w_loh[i],
                p_0=p_0,
                generations=100
            )
            distances[i] = np.sqrt(np.mean((sim_out - target_data)**2))
        
        res_mu_2 = None # Not used

    # --- 3. Rejection & Packaging ---
    cutoff = np.quantile(distances, quantile)
    accepted_mask = distances <= cutoff
    
    data_dict = {
        'Replicate': replicate_name,
        'mu': prior_mu_1[accepted_mask],         # mu_tri
        'w_tri': prior_w_tri[accepted_mask],     
        'w_loh': prior_w_loh[accepted_mask],
        'rmse': distances[accepted_mask],
        'model_type': 'Mutation' if is_mutation_model else 'Standard'
    }
    
    # Add optional columns
    if is_mutation_model:
        data_dict['mu_formation'] = res_mu_2[accepted_mask]
        
    if fixed_w_tri is not None:
        data_dict['is_w_tri_fixed'] = True
    else:
        data_dict['is_w_tri_fixed'] = False

    return pd.DataFrame(data_dict)


def run_abc_inference(replicate_names, data_array, simulator_func, fitness_constants=None, is_mutation_model=False, n_sims=50_000, quantile=0.002, n_jobs=12):
    """
    Main ABC Engine.
    
    Parameters:
    -----------
    fitness_constants : list/array or None
        - If provided: w_tri is FIXED to these values.
        - If None: w_tri is INFERRED (Uniform Prior).
        
    is_mutation_model : bool
        - If True: Samples 'mu_formation' and calls simulator(mu_tri, mu_loh...)
        - If False: Calls simulator(mu...)
    """
    
    print(f"Starting ABC ({n_sims} sims, {n_jobs} cores)...")
    
    # Logging Configuration
    mode_str = "MUTATION" if is_mutation_model else "STANDARD"
    param_str = "FIXED w_tri" if fitness_constants is not None else "INFERRED w_tri"
    print(f"Model: {mode_str} | Parameter Mode: {param_str}")
    
    if fitness_constants is not None and len(fitness_constants) != len(replicate_names):
        raise ValueError(f"Mismatch: {len(fitness_constants)} fitness values vs {len(replicate_names)} replicates.")

    tasks = []
    for i, (name, data) in enumerate(zip(replicate_names, data_array)):
        # Determine fixed w_tri for this specific replicate (if any)
        f_val = fitness_constants[i] if fitness_constants is not None else None
        
        # Args: (Name, Data, Func, Sims, Quantile, Fixed_W, Is_Mut_Model)
        tasks.append((name, data, simulator_func, n_sims, quantile, f_val, is_mutation_model))

    results_dfs = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_abc_worker)(*t) for t in tasks
    )
    
    return pd.concat(results_dfs, ignore_index=True)