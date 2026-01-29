"""
Preprocessing script for linguistic morphology data.
Converts R preprocessing.Rmd logic to Python.
"""

import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path


def split_gloss(gl_vec):
    """
    Parse glosses and extract prefix, stem, other, suffix components.
    
    Args:
        gl_vec: pandas Series of gloss strings
        
    Returns:
        DataFrame with columns: prefix, stem, other, suffix
    """
    results = []
    
    for gloss in gl_vec:
        # 1. Tokenize by whitespace
        tokens = re.split(r'\s+', str(gloss).strip())
        
        # 2. Identify star tokens (3+ asterisks)
        star_mask = [bool(re.match(r'^\*{3,}$', tok)) for tok in tokens]
        
        # 3. Get core (non-star) tokens
        core_tokens = [tok for tok, is_star in zip(tokens, star_mask) if not is_star]
        core = ' '.join(core_tokens) if core_tokens else ''
        
        # 4. Extract prefix and suffix using regex
        prefix_list = re.findall(r'\S+-\s', core)
        suffix_list = re.findall(r'-\S+', core)
        
        # Find positions
        prefix_matches = list(re.finditer(r'\S+-\s', core))
        suffix_matches = list(re.finditer(r'-\S+', core))
        
        prefix_end = max([m.end() for m in prefix_matches]) if prefix_matches else None
        suffix_start = min([m.start() for m in suffix_matches]) if suffix_matches else None
        
        # 5. Extract stem_raw
        stem_start = (prefix_end if prefix_end is not None else 0)
        stem_end = (suffix_start if suffix_start is not None else len(core))
        
        stem_raw = core[stem_start:stem_end].strip() if stem_start < stem_end else ''
        stem_raw = stem_raw if stem_raw else None
        
        # 6. Validate stem (must contain lowercase letter)
        stem = stem_raw if (stem_raw and re.search(r'[a-z]', stem_raw)) else None
        
        # 7. Get star tokens
        star_tokens = [tok for tok, is_star in zip(tokens, star_mask) if is_star]
        
        # 8. Construct other
        other_base = stem_raw if stem is None else None
        other = []
        if other_base:
            other.append(other_base)
        other.extend(star_tokens)
        other = other if other else None
        
        results.append({
            'prefix': prefix_list if prefix_list else None,
            'stem': stem,
            'other': other,
            'suffix': suffix_list if suffix_list else None
        })
    
    return pd.DataFrame(results)


def partition_ps(ps_vec):
    """
    Partition POS strings by whitespace.
    
    Args:
        ps_vec: pandas Series of POS strings
        
    Returns:
        List of tokenized POS lists
    """
    return [re.split(r'\s+', str(ps).strip()) if pd.notna(ps) else [] for ps in ps_vec]


def combine_gloss_units(prefixes, stems, others, suffixes):
    """
    Combine gloss components back together.
    
    Args:
        prefixes: List of prefix lists
        stems: List of stems (can be None)
        others: List of other lists (can be None)
        suffixes: List of suffix lists
        
    Returns:
        List of combined gloss units
    """
    results = []
    
    for pref, stem, other, suf in zip(prefixes, stems, others, suffixes):
        parts = []
        
        if pref:
            parts.extend(pref)
        if stem:
            parts.append(stem)
        if other:
            if isinstance(other, list):
                parts.extend(other)
            else:
                parts.append(other)
        if suf:
            parts.extend(suf)
        
        results.append(parts if parts else None)
    
    return results


def explode_row(row_id, ps_vec, gl_vec, prefix, stem, other, suffix):
    """
    Expand a single row into morphological components.
    
    Args:
        row_id: Original row ID
        ps_vec: List of POS tokens
        gl_vec: List of gloss tokens
        prefix, stem, other, suffix: Morphological components
        
    Returns:
        DataFrame with exploded rows
    """
    rows = []
    
    # If NA or length mismatch, return NA row
    if not ps_vec or not gl_vec or len(ps_vec) != len(gl_vec):
        return pd.DataFrame({
            'id': [row_id],
            'morph_label': [None],
            'gloss': [None],
            'morph_type': [None]
        })
    
    # Generate morph_type for each gloss token
    for ps, gl in zip(ps_vec, gl_vec):
        morph_type = None
        
        if prefix and gl in prefix:
            morph_type = 'prefix'
        elif suffix and gl in suffix:
            morph_type = 'suffix'
        elif stem and gl == stem:
            morph_type = 'stem'
        elif other:
            if isinstance(other, list):
                if gl in other:
                    morph_type = 'other'
            elif gl == other:
                morph_type = 'other'
        
        rows.append({
            'id': row_id,
            'morph_label': ps,
            'gloss': gl,
            'morph_type': morph_type
        })
    
    return pd.DataFrame(rows)


def preprocess_data(words_csv_path, output_pickle_path):
    """
    Main preprocessing pipeline.
    
    Args:
        words_csv_path: Path to words.csv
        output_pickle_path: Path to save processed data
    """
    # Load data
    df = pd.read_csv(words_csv_path)
    
    # Filter valid entries
    invalid_ps = ['', '<p:>', '****']
    df_valid = df[~df['ps'].isin(invalid_ps)].copy()
    df_valid = df_valid.reset_index(drop=True)
    
    print(f"Loaded {len(df_valid)} valid records")
    
    # Step A: Parse glosses
    df_gl = split_gloss(df_valid['gl'])
    
    # Step B: Partition POS
    df_valid['ps_partition'] = partition_ps(df_valid['ps'])
    
    # Step C: Partition glosses
    df_valid['gl_partition'] = combine_gloss_units(
        df_gl['prefix'],
        df_gl['stem'],
        df_gl['other'],
        df_gl['suffix']
    )
    
    # Step D: Length information
    df_valid['len_ps'] = df_valid['ps_partition'].apply(len)
    df_valid['len_gl'] = df_valid['gl_partition'].apply(lambda x: len(x) if x else 0)
    
    # Step E: Explode rows (alignment)
    df_aligned_list = []
    for i in range(len(df_valid)):
        df_aligned_list.append(
            explode_row(
                row_id=i,
                ps_vec=df_valid.iloc[i]['ps_partition'],
                gl_vec=df_valid.iloc[i]['gl_partition'],
                prefix=df_gl.iloc[i]['prefix'],
                stem=df_gl.iloc[i]['stem'],
                other=df_gl.iloc[i]['other'],
                suffix=df_gl.iloc[i]['suffix']
            )
        )
    
    df_aligned = pd.concat(df_aligned_list, ignore_index=True)
    
    # Step F: Merge and save
    df_valid['id'] = range(len(df_valid))
    df_merged = df_valid.merge(df_aligned, on='id', how='left')
    
    # Save as pickle
    df_merged.to_pickle(output_pickle_path)
    print(f"Saved merged data to {output_pickle_path}")
    
    return df_merged


def process_stem_labels(merged_pickle_path, output_pickle_path, pos_mapping_path):
    """
    Post-process to identify stem labels and add POS mapping.
    
    Args:
        merged_pickle_path: Path to merged data pickle
        output_pickle_path: Path to save stem-labeled data
        pos_mapping_path: Path to POS mapping CSV
    """
    df_merged = pd.read_pickle(merged_pickle_path)
    
    # Load POS mapping
    pos_mapping = pd.read_csv(pos_mapping_path)
    
    # Merge POS mapping with data
    df_merged = df_merged.merge(
        pos_mapping,
        left_on='morph_label',
        right_on='morph_label',
        how='left'
    )
    
    print(f"Added coarse_pos column; {df_merged['coarse_pos'].notna().sum()} rows have mapping")
    
    # Save the stem-labeled data
    df_merged.to_pickle(output_pickle_path)
    print(f"Saved stem-labeled data to {output_pickle_path}")
    
    return df_merged


if __name__ == '__main__':
    # Set paths relative to this script
    data_dir = Path(__file__).parent
    
    words_csv = data_dir / 'words.csv'
    pos_mapping_csv = data_dir / 'pos_mapping.csv'
    merged_pkl = data_dir / 'df_merged.pkl'
    stem_labeled_pkl = data_dir / 'df_stem_labeled.pkl'
    
    # Run preprocessing
    if words_csv.exists():
        print("Starting preprocessing...")
        df_merged = preprocess_data(str(words_csv), str(merged_pkl))
        print("Preprocessing complete!")
        
        # Process stem labels
        print("\nProcessing stem labels...")
        process_stem_labels(str(merged_pkl), str(stem_labeled_pkl), str(pos_mapping_csv))
        print("Done!")
    else:
        print(f"Error: {words_csv} not found!")
